import datetime
from conftest import (
    check_request_counts,
    read_env_var,
    spawn_vsb_inner,
    check_recall_stats,
    check_recall_correctness,
)


def spawn_vsb(workload, timeout=60, extra_args=None):
    extra_env = {
        "VSB__PGVECTOR_USERNAME": "postgres",
        "VSB__PGVECTOR_PASSWORD": "postgres",
    }
    return spawn_vsb_inner("pgvector", workload, timeout, extra_args, extra_env)


class TestPgvector:

    def test_ivfflat(self):
        # Test IVFFLAT index type.
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", extra_args=["--pgvector_index_type=ivfflat"]
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 1, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_no_index(self):
        # Test without an index.
        # Test "-test" variant of mnist loads and runs successfully, and gives
        # perfect recall (as Postgres will perform a full kNN scan).
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", extra_args=["--pgvector_index_type=none"]
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "Populate": {"num_requests": 1, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_correctness(1.0),
                },
            },
        )

    def test_gin(self):
        # Test GIN only index type on "yfcc-test".
        (proc, stdout, stderr) = spawn_vsb(
            workload="yfcc-test", extra_args=["--pgvector_index_type=gin"]
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 10, "num_failures": 0},
                "Search": {
                    "num_requests": 500,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_ivfflatgin(self):
        # Test IVFFlat + GIN index type on "yfcc-test".
        (proc, stdout, stderr) = spawn_vsb(
            workload="yfcc-test", extra_args=["--pgvector_index_type=ivfflat+gin"]
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 10, "num_failures": 0},
                "Search": {
                    "num_requests": 500,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_hnswgin(self):
        # Test HNSW + GIN index type on "yfcc-test".
        (proc, stdout, stderr) = spawn_vsb(
            workload="yfcc-test", extra_args=["--pgvector_index_type=hnsw+gin"]
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 10, "num_failures": 0},
                "Search": {
                    "num_requests": 500,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_search_candidates(self):
        # Test pgvector_search_candidates parameter.
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            extra_args=["--pgvector_search_candidates=100"],
        )
        assert proc.returncode == 0
        # Test default value of 0 (unset) -> 2*top_k for hnsw.
        (proc, stdout, stderr) = spawn_vsb(workload="mnist-test")
        assert proc.returncode == 0
        # Defaults should produce "good" (>0.9 p95) recall.
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 1, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_correctness(0.9),
                },
            },
        )
