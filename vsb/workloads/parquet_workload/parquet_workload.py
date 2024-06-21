import json
from abc import ABC
from collections.abc import Iterator
from typing import Generator

import pandas
import pyarrow

from ..base import VectorWorkload
from ..dataset import Dataset
from ...vsb_types import SearchRequest, RecordList, Record


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
                # Metadata is encoded as a string, need to convert to JSON dict
                if "metadata" in records:
                    records["metadata"] = records["metadata"].map(json.loads)
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
