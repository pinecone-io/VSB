import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from vsb.workloads.dataset import Dataset
from vsb import default_cache_dir
import pytest


class TestDataset:
    def test_get_batch_iter_all(self):
        # Test a batch iter for a single chunk yields the entire dataset.
        dataset = Dataset("mnist", cache_dir=default_cache_dir)
        iter = dataset.get_batch_iterator(1, 0, 10)
        assert sum([len(batch) for batch in iter]) == 60000

    def test_get_batch_iter_chunks(self):
        # Test a batch iter for multiple chunks yields the entire dataset.
        dataset = Dataset("mnist", cache_dir=default_cache_dir)
        # Choosing num_chunks which is not a factor of dataset size, so
        # chunk sizes are uneven.
        num_chunks = 7
        total = 0
        min_chunk_length = None
        max_chunk_length = None
        for chunk_id in range(num_chunks):
            iter = dataset.get_batch_iterator(num_chunks, chunk_id, 100)
            chunk_total = sum([len(batch) for batch in iter])
            total += chunk_total
            # Splitting a single parquet file into multiple chunks should yield
            # similar sized chunks.
            min_chunk_length = (
                min(min_chunk_length, chunk_total) if min_chunk_length else chunk_total
            )
            max_chunk_length = (
                max(max_chunk_length, chunk_total) if max_chunk_length else chunk_total
            )
        assert max_chunk_length - min_chunk_length <= 1
        assert total == 60000

    def test_get_batch_iter_multifile(self):
        # Test a batch iter for multiple chunks yields the entire dataset,
        # when the dataset is split across multiple files.
        dataset = Dataset("nq768", cache_dir=default_cache_dir)
        # Choosing num_chunks which is not a factor of dataset size, so
        # chunk sizes are uneven.
        num_chunks = 7
        total = 0
        for chunk_id in range(num_chunks):
            iter = dataset.get_batch_iterator(num_chunks, chunk_id, 100)
            chunk_total = sum([len(batch) for batch in iter])
            total += chunk_total
        assert total == 2_680_893

    def test_get_batch_iter_limit(self):
        # Test a batch iter for multiple chunks yields the entire dataset when
        # a limit is applied.
        dataset_limit = 1000
        dataset = Dataset("mnist", cache_dir=default_cache_dir, limit=dataset_limit)
        # Choosing num_chunks which is not a factor of dataset size, so
        # chunk sizes are uneven.
        num_chunks = 7
        total = 0
        for chunk_id in range(num_chunks):
            iter = dataset.get_batch_iterator(num_chunks, chunk_id, 10)
            chunk_total = sum([len(batch) for batch in iter])
            # If we have applied a limit, then we should have converted the
            # requested dataset into a Table which is then sliced, so every chunk
            # should be non-zero (and similar size).
            assert chunk_total > 0
            assert chunk_total >= (dataset_limit // num_chunks)
            assert chunk_total <= (dataset_limit // num_chunks) + 1
            total += chunk_total
        assert total == dataset_limit

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
