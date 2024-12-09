import pandas
import pytest

from vsb.vsb_types import DistanceMetric, SearchRequest
from vsb.workloads.parquet_workload.parquet_workload import ParquetSubsetWorkload


class TestSubsetWorkloadKNN:
    records = pandas.DataFrame(
        [
            {"id": "a", "values": [2, 0, 0]},
            {"id": "b", "values": [0, 1.9, 0]},
            {"id": "c", "values": [5, 5, 0]},
            {"id": "d", "values": [-1, 0, 2]},
            {"id": "e", "values": [-0.1, 5, 5.1]},
            {"id": "f", "values": [0, 2, 2.1]},
        ]
    )

    queries = pandas.DataFrame(
        [
            {"values": [10, 0.5, 0], "top_k": 3},
            {"values": [-1, 0.2, 1], "top_k": 3},
            {"values": [0, 5.2, 2.1], "top_k": 2},
        ]
    )

    def calc_knn(self, metric: DistanceMetric, q: SearchRequest) -> list[str]:
        return ParquetSubsetWorkload.calc_k_nearest_neighbors(self.records, metric, q)

    def recalculate_neighbors(self, metric: DistanceMetric) -> pandas.DataFrame:
        return ParquetSubsetWorkload.recalculate_neighbors(
            self.records, self.queries, metric
        )

    def test_cosine(self):
        # Test KNN recalculation for cosine distance metric.
        query_results = self.recalculate_neighbors(DistanceMetric.Cosine)

        assert query_results.iloc[0] == ["a", "c", "b"]
        assert query_results.iloc[1] == ["d", "e", "f"]
        assert query_results.iloc[2] == ["b", "e"]

    def test_dot_product(self):
        # Test KNN recalculation for cosine dot product metric.
        query_results = self.recalculate_neighbors(DistanceMetric.DotProduct)

        assert query_results.iloc[0] == ["c", "a", "e"]
        assert query_results.iloc[1] == ["e", "d", "f"]
        assert query_results.iloc[2] == ["e", "c"]

    def test_euclidean(self):
        # Test KNN recalculation for euclidean metric.
        query_results = self.recalculate_neighbors(DistanceMetric.Euclidean)

        assert query_results.iloc[0] == ["c", "a", "b"]
        assert query_results.iloc[1] == ["d", "b", "f"]
        assert query_results.iloc[2] == ["e", "f"]
