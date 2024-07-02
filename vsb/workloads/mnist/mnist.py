from abc import ABC
from collections.abc import Iterator

from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric, SearchRequest, Record

import numpy as np


class MnistBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 784

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Euclidean


class Mnist(MnistBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist", cache_dir=cache_dir)

    @staticmethod
    def record_count() -> int:
        return 60000

    @staticmethod
    def request_count() -> int:
        return 10_000


class MnistTest(ParquetSubsetWorkload, MnistBase):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 20 queries)."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist", cache_dir=cache_dir, limit=600, query_limit=20)

    @staticmethod
    def record_count() -> int:
        return 600

    @staticmethod
    def request_count() -> int:
        return 20
