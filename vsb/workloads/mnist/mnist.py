from ..dataset import Dataset
from ..parquet_workload.parquet_workload import ParquetWorkload


class Mnist(ParquetWorkload):
    def __init__(self):
        super().__init__("mnist")

    def name(self) -> str:
        return "mnist"


class MnistTest(ParquetWorkload):
    """Reduced, "test" variant of mnist; with 1% of the full dataset (600
    passages and 100 queries)."""

    def __init__(self):
        super().__init__("mnist", 600, 100)

    def name(self) -> str:
        return "mnist-test"
