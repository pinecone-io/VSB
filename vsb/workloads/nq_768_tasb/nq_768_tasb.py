from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from vsb.workloads.base import VectorWorkloadSequence, VectorWorkload
from ...vsb_types import DistanceMetric


class Nq768TasbBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 768

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.DotProduct


class Nq768Tasb(Nq768TasbBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "nq-768-tasb", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 2_680_893

    @staticmethod
    def request_count() -> int:
        return 3_452


class Nq768TasbTest(ParquetSubsetWorkload, Nq768TasbBase):
    """Reduced, "test" variant of nq768; with ~1% of the full dataset."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "nq-768-tasb", cache_dir=cache_dir, limit=26809, query_limit=35
        )

    @staticmethod
    def record_count() -> int:
        return 26809

    @staticmethod
    def request_count() -> int:
        return 35


class Nq768TasbCheese(Nq768TasbBase):
    """A subset of nq768 with only the records that do not exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True):
        super().__init__(
            name, "nq-768-tasb-cheese", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 2_393_343

    @staticmethod
    def request_count() -> int:
        return 0


class Nq768TasbHoles(Nq768TasbBase):
    """A subset of nq768 with only the records that exist in
    the top-k neighbors of every query."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True):
        super().__init__(
            name, "nq-768-tasb-holes", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 287_550

    @staticmethod
    def request_count() -> int:
        return 3_452


class Nq768TasbSplit(VectorWorkloadSequence):
    """Drift sequence for nq768 that loads cheese values,
    builds index, loads holes, and queries."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        self._name = name
        self.cheese = Nq768TasbCheese("cheese", cache_dir, load_on_init)
        self.holes = Nq768TasbHoles("holes", cache_dir, load_on_init)
        self.workloads = [self.cheese, self.holes]

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def workload_count() -> int:
        return 2
