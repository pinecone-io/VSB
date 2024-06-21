from abc import ABC

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
        batch_iter = self.get_record_batch_iter(1, 0, self.record_count)
        (_, batch) = next(batch_iter)
        self.records = batch

    @property
    def record_count(self) -> int:
        return 600

    @property
    def request_count(self) -> int:
        return 20

    def get_topk(self, args: dict) -> list[str]:
        # Run a k-NN search to find the top-k nearest neighbors.
        def dist(v: Record):
            # Euclidean distance between v and the query vector.
            return np.linalg.norm(np.array(v.values) - np.array(args["values"]))

        return list(
            map((lambda r: r.id), sorted(self.records.root, key=dist)[: args["top_k"]])
        )

    def next_request(self) -> (str, SearchRequest | None):
        try:
            query = next(self.queries)
            args = query._asdict()
            args.pop("blob")
            # Replace the original neighbors field with the correct top-k nearest neighbors from the truncated dataset.
            args["neighbors"] = self.get_topk(args)
            return "", SearchRequest(**args)

        except StopIteration:
            return None, None
