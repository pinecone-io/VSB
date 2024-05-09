from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload
from ...vsb_types import DistanceMetric


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


class MnistTest(MnistBase):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 20 queries)."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist", cache_dir=cache_dir, limit=600, query_limit=20)
