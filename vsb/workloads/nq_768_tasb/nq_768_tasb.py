from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric


class Nq768TasbBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 768

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.DotProduct


class Nq768Tasb(Nq768TasbBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "nq-768-tasb", cache_dir=cache_dir)

    @staticmethod
    def record_count() -> int:
        return 2_680_893

    @staticmethod
    def request_count() -> int:
        return 3_452


class Nq768TasbTest(ParquetSubsetWorkload, Nq768TasbBase):
    """Reduced, "test" variant of nq768; with ~1% of the full dataset."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name, "nq-768-tasb", cache_dir=cache_dir, limit=26809, query_limit=35
        )

    @staticmethod
    def record_count() -> int:
        return 26809

    @staticmethod
    def request_count() -> int:
        return 35
