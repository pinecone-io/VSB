from abc import ABC
from collections.abc import Iterator

from ..parquet_workload.parquet_workload import ParquetWorkload
from ...vsb_types import DistanceMetric, SearchRequest, Record

import numpy as np


class MnistBase(ParquetWorkload, ABC):
    @property
    def dimensions(self) -> int:
        return 784

    @property
    def metric(self) -> DistanceMetric:
        return DistanceMetric.Euclidean


class Mnist(MnistBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist", cache_dir=cache_dir)

    @property
    def record_count(self) -> int:
        return 60000

    @property
    def request_count(self) -> int:
        return 10_000


class MnistTest(MnistBase):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 20 queries)."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist", cache_dir=cache_dir, limit=600, query_limit=20)
        # Store records in memory; we run a k-NN search to find correct top-k neighbors for each request.
        batch_iter = self.get_record_batch_iter(1, 0, self.record_count)
        (_, batch) = next(batch_iter)
        self.records = batch

    @property
    def record_count(self) -> int:
        return 600

    @property
    def request_count(self) -> int:
        return 20

    def _get_topk(self, req: SearchRequest) -> list[str]:
        # Run a k-NN search to find the top-k nearest neighbors.
        def dist(v: Record):
            # Euclidean distance between v and the query vector.
            return np.linalg.norm(np.array(v.values) - np.array(req.values))

        return list(
            map((lambda r: r.id), sorted(self.records.root, key=dist)[: req.top_k])
        )

    def get_query_iter(
        self, num_users: int, user_id: int
    ) -> Iterator[tuple[str, SearchRequest]]:
        """
        MnistTest only uses the first P passages from the original mnist dataset,
        as such the neighbors field is not necessarily the correct top-k nearest
        neighbors from the truncated passages set.
        As such, replace by re-calculating from the actual records in the dataset.
        """
        base_iter = super().get_query_iter(num_users, user_id)

        def update_neighbors():
            for tenant, request in base_iter:
                request.neighbors = self._get_topk(request)
                yield tenant, request

        return update_neighbors()
