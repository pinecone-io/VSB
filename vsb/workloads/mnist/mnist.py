from abc import ABC
from collections.abc import Iterator

from vsb.workloads.base import VectorWorkload, VectorWorkloadSequence
from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric, RecordList

import numpy as np


class MnistBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 784

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Euclidean


class Mnist(MnistBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(name, "mnist", cache_dir=cache_dir, load_on_init=load_on_init)

    @staticmethod
    def record_count() -> int:
        return 60000

    @staticmethod
    def request_count() -> int:
        return 10_000


class MnistTest(ParquetSubsetWorkload, MnistBase):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 20 queries)."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(name, "mnist", cache_dir=cache_dir, limit=600, query_limit=20)

    @staticmethod
    def record_count() -> int:
        return 600

    @staticmethod
    def request_count() -> int:
        return 20


class MnistSecondTest(ParquetSubsetWorkload, MnistBase):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 20 queries). IDs are appended with a prefix to avoid
    conflicts with the first test."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        super().__init__(name, "mnist", cache_dir=cache_dir, limit=600, query_limit=20)

    @staticmethod
    def record_count() -> int:
        return 600

    @staticmethod
    def request_count() -> int:
        return 20

    def get_record_batch_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, RecordList]]:
        for batch in super().get_record_batch_iter(num_users, user_id, batch_size):
            record_list = batch[1]
            for record in record_list:
                record.id += "_2"
            yield (batch[0], record_list)


class MnistCheese(MnistBase):
    """A subset of mnist with only the records that do not exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True):
        super().__init__(
            name, "mnist-cheese", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 672

    @staticmethod
    def request_count() -> int:
        return 0


class MnistHoles(MnistBase):
    """A subset of mnist with only the records that exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True):
        super().__init__(
            name, "mnist-holes", cache_dir=cache_dir, load_on_init=load_on_init
        )

    @staticmethod
    def record_count() -> int:
        return 59_328

    @staticmethod
    def request_count() -> int:
        return 10_000


class MnistSplit(VectorWorkloadSequence):
    """Drift sequence for mnist that loads cheese values,
    builds index, loads holes, and queries."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        self._name = name
        self.cheese = MnistCheese("cheese", cache_dir, load_on_init)
        self.holes = MnistHoles("holes", cache_dir, load_on_init)
        self.workloads = [self.cheese, self.holes]

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def workload_count() -> int:
        return 2


class MnistDoubleTest(VectorWorkloadSequence):
    """Reduced variant of mnist that reruns the test workload twice.
    Primarily used for testing multi-iteration workloads."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True):
        # load_on_init is ignored; ParquetSubsetWorkload does not support it
        self._name = name
        self.test1 = MnistTest("test1", cache_dir)
        self.test2 = MnistSecondTest("test2", cache_dir)
        self.workloads = [self.test1, self.test2]

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def workload_count() -> int:
        return 2
