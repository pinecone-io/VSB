from abc import ABC
from collections.abc import Iterator

from vsb.workloads.base import VectorWorkload, VectorWorkloadSequence
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


class MnistCheese(MnistBase):
    """A subset of mnist with only the records that do not exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist-cheese", cache_dir=cache_dir)

    @staticmethod
    def record_count() -> int:
        return 672

    @staticmethod
    def request_count() -> int:
        return 0


class MnistHoles(MnistBase):
    """A subset of mnist with only the records that exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "mnist-holes", cache_dir=cache_dir)

    @staticmethod
    def record_count() -> int:
        return 59_328

    @staticmethod
    def request_count() -> int:
        return 10_000


class MnistSplit(VectorWorkloadSequence):
    """Drift sequence for mnist that loads cheese values,
    builds index, loads holes, and queries."""

    def __init__(self, name: str, cache_dir: str):
        self._name = name
        self.cheese = MnistCheese("cheese", cache_dir)
        self.holes = MnistHoles("holes", cache_dir)
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


class MnistDoubleTest(VectorWorkloadSequence):
    """Reduced variant of mnist that reruns the test workload twice.
    Primarily used for testing multi-iteration workloads."""

    def __init__(self, name: str, cache_dir: str):
        self._name = name
        self.test1 = MnistTest("test1", cache_dir)
        self.test2 = MnistTest("test2", cache_dir)
        # We have to "trick" pinecone's iteration helper to think we have 600 records
        # by setting the record count to 0 for the second test.
        # Otherwise, finalize will "wait" for 1200 records forever, even though only 600
        # are actually available.
        self.test2.record_count = lambda: 0
        self.workloads = [self.test1, self.test2]

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
