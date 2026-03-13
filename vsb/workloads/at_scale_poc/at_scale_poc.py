"""
At Scale POC dataset from Amazon Reviews dataset - multiple sizes, 500K/10M/100M
"""

from abc import ABC
from vsb.workloads.base import VectorWorkload, VectorWorkloadSequence
from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric


class AtScalePocBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 1024

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Cosine


class AtScalePoc500K(AtScalePocBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "at-scale-poc/at-scale-poc-500k", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 500_000

    @staticmethod
    def request_count() -> int:
        return 5_000
        
class AtScalePoc10M(AtScalePocBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "at-scale-poc/at-scale-poc-10m", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 10_000_000

    @staticmethod
    def request_count() -> int:
        return 10_000

class AtScalePoc100M(AtScalePocBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(
            name, "at-scale-poc/at-scale-poc-100m", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 100_000_000

    @staticmethod
    def request_count() -> int:
        return 10_000