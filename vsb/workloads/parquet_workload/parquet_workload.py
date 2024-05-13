from abc import ABC
from collections.abc import Iterator

import numpy
import pandas
import pyarrow

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
        self,
        name: str,
        dataset_name: str,
        cache_dir: str,
        limit: int = 0,
        query_limit: int = 0,
    ):
        super().__init__(name)
        self.dataset = Dataset(dataset_name, cache_dir=cache_dir, limit=limit)

        self.dataset.setup_queries(load_queries=True, query_limit=query_limit)
        self.queries = self.dataset.queries.itertuples(index=False)

    def get_record_batch_iter(
        self, num_users: int, user_id: int
    ) -> RecordBatchIterator:

        # TODO: Make batch size configurable.
        batch_iter = self.dataset.get_batch_iterator(num_users, user_id, 200)

        # Need to convert the pyarrow RecordBatch into a python dict for
        # consumption by the database.
        def recordbatch_to_pylist(iter: Iterator[pyarrow.RecordBatch]):
            for batch in iter:
                # Note: RecordBatch does have a do_pylist() method itself, however it's
                # much slower to use that than convert to pandas and then convert to a
                # dict (!).
                # See: https://github.com/apache/arrow/issues/28694
                # TODO: Add multiple tenant support.
                yield "", batch.to_pandas().to_dict("records")

        return recordbatch_to_pylist(batch_iter)

    def next_request(self) -> (str, SearchRequest | None):
        try:
            query = next(self.queries)
            # TODO: Add multiple tenant support.
            return "", SearchRequest(**query._asdict())
        except StopIteration:
            return None, None
