from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload
from ...vsb_types import DistanceMetric


class Nq768TasbBase(ParquetWorkload, ABC):
    @property
    def dimensions(self) -> int:
        return 768

    @property
    def metric(self) -> DistanceMetric:
        return DistanceMetric.DotProduct


class Nq768Tasb(Nq768TasbBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "nq-768-tasb", cache_dir=cache_dir)

    @property
    def record_count(self) -> int:
        return 2_680_893

    @property
    def request_count(self) -> int:
        3_452


class Nq768TasbTest(Nq768TasbBase):
    """Reduced, "test" variant of nq768; with ~1% of the full dataset."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name, "nq-768-tasb", cache_dir=cache_dir, limit=26809, query_limit=35
        )

    @property
    def record_count(self) -> int:
        return 26809

    @property
    def request_count(self) -> int:
        35
