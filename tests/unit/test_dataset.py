import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vsb.workloads.dataset import Dataset
import pytest


class TestDataset:

    def test_limit(self):
        limit = 123
        name = "mnist"
        dataset = Dataset(name, limit=limit)
        # Sanity check that the complete dataset size is greater than what
        # we are going to limit to.
        dataset_info = ([d for d in dataset.list() if d["name"] == name][0])
        assert dataset_info["documents"] > limit, \
            "Too few documents in dataset to be able to limit"

        dataset.load_documents()
        assert len(dataset.documents) == limit

    def test_recall_equal(self):
        # Test recall() for equal length actual and expected lists.
        assert Dataset.recall(["1"], ["1"]) == 1.0
        assert Dataset.recall(["0"], ["1"]) == 0
        assert Dataset.recall(["1", "3"], ["1", "2"]) == 0.5
        assert Dataset.recall(["3", "1"], ["1", "2"]) == 0.5
        assert Dataset.recall(["1", "2"], ["2", "1"]) == 1
        assert Dataset.recall(["2", "3", "4", "5"], ["1", "2", "3", "4"]) == 0.75

    def test_recall_actual_fewer_expected(self):
        # Test recall() when actual matches is fewer than expected - i.e.
        # query ran with lower top_k. In this situation recall() should
        # only consider the k nearest expected_matches.
        assert Dataset.recall(["1"], ["1", "2"]) == 1.0
        assert Dataset.recall(["2"], ["1", "2"]) == 0
        assert Dataset.recall(["1"], ["1", "2", "3"]) == 1.0
        assert Dataset.recall(["1", "2"], ["1", "2", "3"]) == 1.0

    def test_recall_actual_more_expected(self):
        # Test recall() when actual matches are more than expected - i.e.
        # query ran with a higher top_k. In this situation we should still
        # compare against the full expected_matches.
        assert Dataset.recall(["1", "2"], ["1"]) == 1.0
        assert Dataset.recall(["1", "2"], ["2"]) == 1.0
        assert Dataset.recall(["1", "3"], ["2"]) == 0
        assert Dataset.recall(["1", "2", "3"], ["3"]) == 1
