from abc import ABC
from collections.abc import Iterator

import numpy
import pandas

from ..base import VectorWorkload, RecordBatchIterator
from ..dataset import Dataset
from ...vsb_types import Record, SearchRequest


class ParquetWorkload(VectorWorkload, ABC):
    """A static workload which is implemented by reading records and query from
    two sets of parquet files.
    The initial records for the workload are loaded from one set of parquet
    files, then the run phase of the workload consists of queries loaded
    from a second set of parquet files.
    """

    def __init__(
        self, dataset_name: str, cache_dir: str, limit: int = 0, query_limit: int = 0
    ):
        self.dataset = Dataset(dataset_name, cache_dir=cache_dir, limit=limit)
        self.dataset.load_documents()
        # TODO: At parquet level should probably just iterate across entire row
        # groups, if the DB wants to split further they can chose to.
        self.records = Dataset.split_dataframe(self.dataset.documents, 200)

        self.dataset.setup_queries(load_queries=True, query_limit=query_limit)
        self.queries = self.dataset.queries.itertuples(index=False)

    def get_record_batch_iter(
        self, num_users: int, user_id: int
    ) -> RecordBatchIterator:
        # Need split the documents into `num_users` subsets, then return an
        # iterator over the `user_id`th subset.
        total_docs = self.dataset.documents.shape[0]
        quotient, remainder = divmod(total_docs, num_users)
        chunks = [quotient + (1 if r < remainder else 0) for r in range(num_users)]
        # Determine start position based on sum of size of all chunks prior
        # to ours.
        start = sum(chunks[:user_id])
        # Note: For pandas DataFrames, slicing is *inclusive*.
        end = start + chunks[user_id]
        user_chunk = self.dataset.documents[start:end]

        # TODO: Add multiple tenant support.
        def batch_dataframe(
            df: pandas.DataFrame, batch_size
        ) -> Iterator[pandas.DataFrame]:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i : i + batch_size]
                records = [r for r in batch.to_dict("records")]
                yield "", records

        # TODO: make batch size configurable.
        return batch_dataframe(user_chunk, 200)

    def next_request(self) -> (str, SearchRequest | None):
        try:
            query = next(self.queries)
            # TODO: Add multiple tenant support.
            return "", SearchRequest(**query._asdict())
        except StopIteration:
            return None, None
