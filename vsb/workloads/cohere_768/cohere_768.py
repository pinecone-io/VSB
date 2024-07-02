"""
Cohere-768 dataset - 10M records from en wikipedia, embedded using Cohere
(https://huggingface.co/datasets/Cohere/wikipedia-22-12/tree/main/en)
"""

from abc import ABC

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
