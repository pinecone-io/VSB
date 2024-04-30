from ..parquet_workload.parquet_workload import ParquetWorkload


class Mnist(ParquetWorkload):
    def __init__(self, cache_dir: str):
        super().__init__("mnist", cache_dir=cache_dir)

    def name(self) -> str:
        return "mnist"


class MnistTest(ParquetWorkload):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 100 queries)."""

    def __init__(self, cache_dir: str):
        super().__init__("mnist", cache_dir=cache_dir, limit=600, query_limit=100)

    def name(self) -> str:
        return "mnist-test"
