"""
MSMarco-V2-Ada dataset - 138M records from Microsoft's MSMARCOv2 dataset, embedded with Ada
(https://microsoft.github.io/msmarco/)
"""

from abc import ABC
from vsb.workloads.base import VectorWorkload, VectorWorkloadSequence
from ..parquet_workload.parquet_workload import ParquetWorkload, ParquetSubsetWorkload
from ...vsb_types import DistanceMetric


class MsMarcoV2AdaBase(ParquetWorkload, ABC):
    @staticmethod
    def dimensions() -> int:
        return 1536

    @staticmethod
    def metric() -> DistanceMetric:
        return DistanceMetric.Cosine

    def supports_bulk_import(self) -> bool:
        return True

    def get_import_uri(self) -> str:
        return "gs://pinecone-datasets-dev/msmarco-v2-ada"

    def get_import_namespace(self) -> str:
        return "passages"


class MsMarcoV2Ada(MsMarcoV2AdaBase):
    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        # Check if bulk import is enabled via options
        options = kwargs.get("options")
        skip_passages = False
        if options and getattr(options, "pinecone_bulk_import", False):
            skip_passages = True
        super().__init__(
            name,
            "msmarco-v2-ada",
            cache_dir=cache_dir,
            load_on_init=load_on_init,
            skip_passages=skip_passages,
        )

    @staticmethod
    def record_count() -> int:
        return 138_364_198

    @staticmethod
    def request_count() -> int:
        return 8_184


class MsMarcoV2AdaTest(ParquetSubsetWorkload, MsMarcoV2AdaBase):
    """Reduced, "test" variant of ms-marco-v2; with ~0.1% of the full dataset (100,000
    passages and 100 queries)."""

    def __init__(self, name: str, cache_dir: str, load_on_init: bool = True, **kwargs):
        # Check if bulk import is enabled via options
        options = kwargs.get("options")
        skip_passages = False
        if options and getattr(options, "pinecone_bulk_import", False):
            skip_passages = True
        super().__init__(
            name,
            "msmarco-v2-ada",
            cache_dir=cache_dir,
            limit=self.record_count(),
            query_limit=self.request_count(),
            skip_passages=skip_passages,
        )

    @staticmethod
    def record_count() -> int:
        return 100_000

    @staticmethod
    def request_count() -> int:
        return 100
