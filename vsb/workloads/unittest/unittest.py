from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload
from ...vsb_types import DistanceMetric


class UnitTestBase(ParquetWorkload, ABC):
    @property
    def dimensions(self) -> int:
        return 784

    @property
    def metric(self) -> DistanceMetric:
        return DistanceMetric.Euclidean


class UnitTest(UnitTestBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "unittest", cache_dir=cache_dir)

    @property
    def record_count(self) -> int:
        return 100

    @property
    def request_count(self) -> int:
        return 1


class MnistTest(MnistBase):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 20 queries)."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist", cache_dir=cache_dir, limit=600, query_limit=20)

    @property
    def record_count(self) -> int:
        return 600

    @property
    def request_count(self) -> int:
        return 20
