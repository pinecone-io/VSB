from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric


class YFCCBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 192

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Euclidean


class YFCC(YFCCBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name, "yfcc-10M-filter-euclidean-formatted-multipart", cache_dir=cache_dir
        )

    @staticmethod
    def record_count() -> int:
        return 10_000_000

    @staticmethod
    def request_count() -> int:
        return 100_000


class YFCCTest(ParquetSubsetWorkload, YFCCBase):
    """Reduced, "test" variant of YFCC; with ~0.1% of the full dataset / 0.5%
    of queries"""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name,
            "yfcc-100K-filter-euclidean-formatted",
            limit=10_000,
            query_limit=500,
            cache_dir=cache_dir,
        )

    @staticmethod
    def record_count() -> int:
        return 10_000

    @staticmethod
    def request_count() -> int:
        return 500
