"""
Cohere-768 dataset - 10M records from en wikipedia, embedded using Cohere
(https://huggingface.co/datasets/Cohere/wikipedia-22-12/tree/main/en)
"""

from abc import ABC
from vsb.workloads.base import VectorWorkload, VectorWorkloadSequence
from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric


class CohereBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 768

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Cosine


class Cohere768(CohereBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "cohere-768", cache_dir=cache_dir)

    @staticmethod
    def record_count() -> int:
        return 10_000_000

    @staticmethod
    def request_count() -> int:
        return 1_000


class Cohere768Test(ParquetSubsetWorkload, CohereBase):
    """Reduced, "test" variant of cohere-768; with ~0.1% of the full dataset (100,000
    passages and 100 queries)."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name,
            "cohere-768",
            cache_dir=cache_dir,
            limit=self.record_count(),
            query_limit=self.request_count(),
        )

    @staticmethod
    def record_count() -> int:
        return 100_000

    @staticmethod
    def request_count() -> int:
        return 100


class Cohere768Cheese(CohereBase):
    """A subset of mnist with only the records that do not exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "cohere-768-cheese", cache_dir=cache_dir)

    @staticmethod
    def record_count() -> int:
        return 6_379_955

    @staticmethod
    def request_count() -> int:
        return 0


class Cohere768Holes(CohereBase):
    """A subset of cohere-768 with only the records that exist in
    the top-k neighbors of any query."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "cohere-768-holes", cache_dir=cache_dir)

    @staticmethod
    def record_count() -> int:
        return 3_620_045

    @staticmethod
    def request_count() -> int:
        return 1_000


class Cohere768Split(VectorWorkloadSequence):
    """Drift sequence for cohere-768 that loads cheese values,
    builds index, loads holes, and queries."""

    def __init__(self, name: str, cache_dir: str):
        self._name = name
        self.cheese = Cohere768Cheese("cheese", cache_dir)
        self.holes = Cohere768Holes("holes", cache_dir)
        self.workloads = [self.cheese, self.holes]

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def workload_count() -> int:
        return 2

    def __next__(self) -> VectorWorkload:
        if not self.workloads:
            raise StopIteration
        return self.workloads.pop(0)

    def dimensions(self) -> int:
        return 768

    def metric(self) -> DistanceMetric:
        return DistanceMetric.Cosine

    def record_count(self) -> int:
        return 6_379_955 + 3_620_045

    def request_count(self) -> int:
        return 1_000
