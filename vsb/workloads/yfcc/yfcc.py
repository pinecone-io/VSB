from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from vsb.workloads.base import VectorWorkloadSequence, VectorWorkload
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


class YFCCCheese(YFCCBase):
    """A subset of YFCC with only the records that do not exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name,
            "yfcc-10M-filter-euclidean-formatted-multipart-cheese",
            cache_dir=cache_dir,
        )

    @staticmethod
    def record_count() -> int:
        return 9_264_264

    @staticmethod
    def request_count() -> int:
        return 0


class YFCCHoles(YFCCBase):
    """A subset of YFCC with only the records that exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name,
            "yfcc-10M-filter-euclidean-formatted-multipart-holes",
            cache_dir=cache_dir,
        )

    @staticmethod
    def record_count() -> int:
        return 735_736

    @staticmethod
    def request_count() -> int:
        return 100_000


class YFCCSplit(VectorWorkloadSequence):
    """Drift sequence for mnist that loads cheese values,
    builds index, loads holes, and queries."""

    def __init__(self, name: str, cache_dir: str):
        self._name = name
        self.cheese = YFCCCheese("cheese", cache_dir)
        self.holes = YFCCHoles("holes", cache_dir)
        self.workloads = [self.cheese, self.holes]

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def workload_count() -> int:
        return 2

    def __getitem__(self, index: int) -> VectorWorkload:
        if index < 0 or index >= len(self.workloads):
            raise IndexError
        return self.workloads[index]
