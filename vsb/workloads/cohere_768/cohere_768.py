"""
Cohere-768 dataset - 10M records from en wikipedia, embedded using Cohere
(https://huggingface.co/datasets/Cohere/wikipedia-22-12/tree/main/en)
"""

from abc import ABC

from ..parquet_workload.parquet_workload import ParquetWorkload
from ...vsb_types import DistanceMetric


class CohereBase(ParquetWorkload, ABC):
    @property
    def dimensions(self) -> int:
        return 768

    @property
    def metric(self) -> DistanceMetric:
        return DistanceMetric.Cosine


class Cohere768(CohereBase):
    def __init__(self, name: str, cache_dir: str):
        super().__init__(name, "cohere-768", cache_dir=cache_dir)

    @property
    def record_count(self) -> int:
        return 10_000_000

    @property
    def request_count(self) -> int:
        return 1_000


class Cohere768Test(CohereBase):
    """Reduced, "test" variant of cohere-768; with ~0.1% of the full dataset (100,000
    passages and 100 queries)."""

    def __init__(self, name: str, cache_dir: str):
        super().__init__(
            name,
            "cohere-768",
            cache_dir=cache_dir,
            limit=self.record_count,
            query_limit=self.request_count,
        )

    @property
    def record_count(self) -> int:
        return 100_000

    @property
    def request_count(self) -> int:
        return 100
