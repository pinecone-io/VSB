from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload
from ...vsb_types import DistanceMetric


class YFCCBase(ParquetWorkload, ABC):
    _dataset_name = "yfcc-10M-filter-euclidean-formatted"

    @property
    def dimensions(self) -> int:
        return 192

    @property
    def metric(self) -> DistanceMetric:
        return DistanceMetric.Euclidean


class YFCC(YFCCBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, self._dataset_name, cache_dir=cache_dir)

    @property
    def record_count(self) -> int:
        return 10_000_000


class YFCCTest(YFCCBase):
    """Reduced, "test" variant of YFCC; with ~1% of the full dataset."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name, self._dataset_name, cache_dir=cache_dir, limit=10_000, query_limit=100
        )

    @property
    def record_count(self) -> int:
        return 10_000
