import pandas
import pytest

import vsb
from vsb.vsb_types import DistanceMetric, Record, RecordList, SearchRequest
from vsb.workloads.parquet_workload.parquet_workload import ParquetSubsetWorkload


class TestSubsetWorkloadKNN:
    records = pandas.DataFrame(
        [
            {"id": "a", "values": [2, 0, 0]},
            {"id": "b", "values": [0, 2, 0]},
            {"id": "c", "values": [5, 5, 0]},
            {"id": "d", "values": [-1, 0, 2]},
            {"id": "e", "values": [0, 5, 5]},
            {"id": "f", "values": [0, 2, 2]},
        ]
    )

    queries = [
        SearchRequest(values=[10, 0.5, 0], top_k=3),
        SearchRequest(values=[-1, 0.1, 1], top_k=3),
        SearchRequest(values=[0, 5, 2], top_k=2),
    ]

    def calc_knn(self, metric: DistanceMetric, q: SearchRequest) -> list[str]:
        return ParquetSubsetWorkload.calc_k_nearest_neighbors(self.records, metric, q)

    def test_cosine(self):
        # Test KNN recalculation for cosine distance metric.
        metric = DistanceMetric.Cosine

        assert self.calc_knn(metric, self.queries[0]) == ["a", "c", "b"]
        assert self.calc_knn(metric, self.queries[1]) == ["d", "e", "f"]
        assert self.calc_knn(metric, self.queries[2]) == ["b", "e"]

    def test_dot_product(self):
        # Test KNN recalculation for cosine dot product metric.
        metric = DistanceMetric.DotProduct

        assert self.calc_knn(metric, self.queries[0]) == ["c", "a", "e"]
        assert self.calc_knn(metric, self.queries[1]) == ["e", "d", "f"]
        assert self.calc_knn(metric, self.queries[2]) == ["e", "c"]

    def test_euclidean(self):
        # Test KNN recalculation for euclidean metric.
        metric = DistanceMetric.Euclidean

        assert self.calc_knn(metric, self.queries[0]) == ["c", "a", "b"]
        assert self.calc_knn(metric, self.queries[1]) == ["d", "b", "f"]
        assert self.calc_knn(metric, self.queries[2]) == ["e", "f"]
