from abc import ABC

from ..base import VectorWorkload
from ..dataset import Dataset
from ...databases.base import Index
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

        self.dataset.setup_queries()
        if query_limit:
            self.dataset.queries = self.dataset.queries[:query_limit]
        self.queries = self.dataset.queries.itertuples(index=False)

        # Map of index tenants to the Index instance. Lazily populated as
        # requests are accessed for a particular tenant.
        self.indexes = dict()

    def next_record_batch(self) -> (str, list[Record]):
        try:
            batch_df = next(self.records)
            records = [r for r in batch_df.to_dict("records")]
            # TODO: Add multiple tenant support.
            return "", records
        except StopIteration:
            return None, None

    def next_request(self) -> (str, SearchRequest | None):
        try:
            query = next(self.queries)
            # TODO: Add multiple tenant support.
            return "", SearchRequest(**query._asdict())
        except StopIteration:
            return None, None
