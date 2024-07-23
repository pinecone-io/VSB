import json
from abc import ABC
from collections.abc import Iterator
from typing import Generator

import pandas
import numpy as np
import pyarrow

from ..base import VectorWorkload
from ..dataset import Dataset
from ...vsb_types import SearchRequest, RecordList, Record, DistanceMetric, Vector
from ...databases.pgvector.filter_util import FilterUtil


class ParquetWorkload(VectorWorkload, ABC):
    """A static workload which is implemented by reading records and query from
    two sets of parquet files.
    The initial records for the workload are loaded from one set of parquet
    files, then the run phase of the workload consists of queries loaded
    from a second set of parquet files.
    """

    def __init__(
        self,
        name: str,
        dataset_name: str,
        cache_dir: str,
        limit: int = 0,
        query_limit: int = 0,
    ):
        super().__init__(name)
        self.dataset = Dataset(dataset_name, cache_dir=cache_dir, limit=limit)

        self.dataset.setup_queries(query_limit=query_limit)
        self.queries = self.dataset.queries

    def get_sample_record(self) -> Record:
        iter = self.get_record_batch_iter(1, 0, 1)
        (_, batch) = next(iter)
        sample = Record.model_validate(batch[0])
        return sample

    def get_record_batch_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, RecordList]]:

        batch_iter = self.dataset.get_batch_iterator(num_users, user_id, batch_size)

        # Need to convert the pyarrow RecordBatch into a pandas DataFrame with
        # correctly formatted fields for consumption by the database.
        def recordbatch_to_dataframe(
            iter: Iterator[pyarrow.RecordBatch],
        ) -> Generator[tuple[str, list[dict]], None, None]:
            for batch in iter:
                # Note: RecordBatch does have a to_pylist() method itself, however it's
                # much slower to use that than convert to pandas and then convert to a
                # dict (!).
                # See: https://github.com/apache/arrow/issues/28694
                # TODO: Add multiple tenant support.
                records: pandas.DataFrame = batch.to_pandas()
                self._decode_metadata(records)
                yield "", RecordList(records.to_dict("records"))

        return recordbatch_to_dataframe(batch_iter)

    def get_query_iter(
        self, num_users: int, user_id: int
    ) -> Iterator[tuple[str, SearchRequest]]:
        # Calculate start / end for this query chunk, then split the table
        # and create an iterator over it.
        quotient, remainder = divmod(len(self.queries), num_users)
        chunks = [quotient + (1 if r < remainder else 0) for r in range(num_users)]
        # Determine start position based on sum of size of all chunks prior
        # to ours.
        start = sum(chunks[:user_id])
        end = start + chunks[user_id]
        user_chunk = self.queries.iloc[start:end]

        # Return an iterator for the Nth chunk.
        def make_query_iter(queries):
            for query in queries.itertuples(index=False):
                assert query
                # neighbors are nested inside a `blob` field, need to unnest them
                # to pass to SearchRequest.
                args = query._asdict()
                assert "values" in args
                assert args["values"] is not None
                args.update(args.pop("blob"))
                # TODO: Add multiple tenant support.
                yield "", SearchRequest(**args)

        return make_query_iter(user_chunk)

    @staticmethod
    def _decode_metadata(records):
        # Metadata is encoded as a string, need to convert to JSON dict
        if "metadata" in records:
            records["metadata"] = records["metadata"].map(json.loads)


class ParquetSubsetWorkload(ParquetWorkload, ABC):
    """A subclass of ParquetWorkload that reads a subset of records and queries
    from the original dataset.
    The expected results of the queries are incorrect with respect to the new
    subset of records, so we recalculate correct expected results in-memory during
    each request.
    """

    def __init__(
        self,
        name: str,
        dataset_name: str,
        cache_dir: str,
        limit: int = 0,
        query_limit: int = 0,
    ):
        super().__init__(
            name,
            dataset_name,
            cache_dir=cache_dir,
            limit=limit,
            query_limit=query_limit,
        )
        # Store records in memory; we run a k-NN search to find correct top-k
        # neighbors for each request.
        self.records: pandas.DataFrame = self._get_records()

        # Recalculate neighbors for each query in the dataset
        neighbors = self.recalculate_neighbors(
            self.records, self.queries, self.metric()
        )

        def update_blob(blob, neighbors):
            blob["neighbors"] = neighbors
            return blob

        self.queries["blob"] = self.queries.apply(
            lambda row: update_blob(row["blob"], neighbors[row.name]),
            axis=1,
        )

    def _get_records(self) -> pandas.DataFrame:
        batch_iter = self.dataset.get_batch_iterator(1, 0, self.record_count())
        all_records: pandas.DataFrame = pandas.DataFrame()
        for batch in batch_iter:
            records = batch.to_pandas()
            self._decode_metadata(records)
            all_records = pandas.concat([all_records, records])
        return all_records

    def _get_topk(self, req: SearchRequest) -> list[str]:
        return self.calc_k_nearest_neighbors(self.records, self.metric(), req)

    @staticmethod
    def recalculate_neighbors(
        records: pandas.DataFrame, queries: pandas.DataFrame, metric: DistanceMetric
    ) -> pandas.Series:
        """Recalculate the neighbors field for each query in the given set of
        queries, using the given set of records.
        """

        # Convert the list of embeddings to a numpy array for faster computation
        embeddings = np.stack(records["values"].values)

        def calculate_distances(query_vector, embeddings):
            match metric:
                case DistanceMetric.Euclidean:
                    # Compute the Euclidean distances
                    return np.linalg.norm(embeddings - query_vector, axis=1)
                case DistanceMetric.DotProduct:
                    # Compute the negative dot-product (to use argsort for
                    # descending order)
                    return -np.dot(embeddings, query_vector)
                case DistanceMetric.Cosine:
                    # Compute the cosine distances (1 - cosine similarity) to use
                    # argsort for descending order.
                    query_norm = np.linalg.norm(query_vector)
                    embeddings_norm = np.linalg.norm(embeddings, axis=1)
                    cosine_similarities = np.dot(embeddings, query_vector) / (
                        embeddings_norm * query_norm
                    )
                    return 1 - cosine_similarities

        # Function to find top k nearest neighbors for a single query
        def get_top_k_nearest(query_vector, top_k, q_filter):
            distances = calculate_distances(query_vector, embeddings)

            # Filter by metadata tags if provided (e.g. yfcc)
            if q_filter is not None:
                filters = FilterUtil.to_set(q_filter)

                def filt(metadata: dict):
                    if metadata is not None:
                        assert "tags" in metadata  # We only support yfcc tags for now.
                        return filters <= set(metadata["tags"])
                    return filters <= set()

                # np.nonzero returns a tuple of arrays with the non-zero (true) indices of an ndarray
                # Apply filter to the 'metadata' Series to avoid iterating over DataFrame rows.
                filtered_indices = np.nonzero(records["metadata"].apply(filt).array)[0]
            else:
                filtered_indices = range(len(records))

            # Get the distances of the filtered indices
            filtered_distances = distances[filtered_indices]

            # Get the indices of the top_k smallest distances
            top_k_filtered_indices = np.argsort(filtered_distances)[:top_k]

            # Map the filtered indices back to the original indices
            top_k_original_indices = [
                filtered_indices[i] for i in top_k_filtered_indices
            ]

            return records.iloc[top_k_original_indices]["id"].tolist()

        # Apply the function to each query
        return queries.apply(
            lambda row: get_top_k_nearest(
                row["values"], row["top_k"], row["filter"] if "filter" in row else None
            ),
            axis=1,
        )
