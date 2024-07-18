import datetime
from conftest import (
    check_request_counts,
    read_env_var,
    random_string,
    spawn_vsb_inner,
    check_recall_stats,
    check_recall_correctness,
)


def _get_index_name() -> str:
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    now = now.replace(":", "-")
    index_name = read_env_var("NAME_PREFIX") + "--" + now + "--" + random_string(10)
    index_name = index_name.lower()


def spawn_vsb(workload, timeout=60, extra_args=None):
    extra_env = {
        "VSB__PGVECTOR_USERNAME": "postgres",
        "VSB__PGVECTOR_PASSWORD": "postgres",
    }
    return spawn_vsb_inner("pgvector", workload, timeout, extra_args, extra_env)


class TestPgvector:
    def test_mnist_single(self):
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(workload="mnist-test")
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

    def test_mnist_concurrent(self):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent users
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            extra_args=["--users=4"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Given the size of each
                # chunk will be less than the batch size (600 / 4 < 1000), then the
                # number of requests will be equal to the number of users - i.e. 4
                "Populate": {"num_requests": 4, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_multiprocess(self):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            extra_args=["--processes=2", "--users=4"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Given the size of each
                # chunk will be less than the batch size (600 / 4 < 1000), then the
                # number of requests will be equal to the number of users - i.e. 4
                "Populate": {"num_requests": 4, "num_failures": 0},
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_double(self, api_key, pinecone_index_mnist):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(workload="mnist-double-test")
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "test1": {
                    # For multiple users the populate phase will chunk the records to be
                    # loaded into num_users chunks - i.e. 4 here. Given the size of each
                    # chunk will be less than the batch size (600 / 4 < 200), then the
                    # number of requests will be equal to the number of users - i.e. 4
                    "Populate": {"num_requests": 2, "num_failures": 0},
                    # The number of Search requests should equal the number in the dataset
                    # (20 for mnist-test).
                    "Search": {
                        "num_requests": 20,
                        "num_failures": 0,
                        "recall": check_recall_stats,
                    },
                },
                "test2": {
                    "Populate": {"num_requests": 2, "num_failures": 0},
                    "Search": {
                        "num_requests": 20,
                        "num_failures": 0,
                        "recall": check_recall_stats,
                    },
                },
            },
        )

    def test_mnist_double_concurrent(self, api_key, pinecone_index_mnist):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent users, and with a request rate limit set.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-double-test",
            extra_args=["--users=4", "--requests_per_sec=40"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "test1": {
                    # For multiple users the populate phase will chunk the records to be
                    # loaded into num_users chunks - i.e. 4 here. Given the size of each
                    # chunk will be less than the batch size (600 / 4 < 200), then the
                    # number of requests will be equal to the number of users - i.e. 4
                    "Populate": {"num_requests": 4, "num_failures": 0},
                    # The number of Search requests should equal the number in the dataset
                    # (20 for mnist-test).
                    "Search": {
                        "num_requests": 20,
                        "num_failures": 0,
                        "recall": check_recall_stats,
                    },
                },
                "test2": {
                    "Populate": {"num_requests": 4, "num_failures": 0},
                    "Search": {
                        "num_requests": 20,
                        "num_failures": 0,
                        "recall": check_recall_stats,
                    },
                },
            },
        )

    def test_mnist_double_multiprocess(self, api_key, pinecone_index_mnist):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-double-test",
            extra_args=["--processes=4", "--users=4"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "test1": {
                    # For multiple users the populate phase will chunk the records to be
                    # loaded into num_users chunks - i.e. 4 here. Given the size of each
                    # chunk will be less than the batch size (600 / 4 < 200), then the
                    # number of requests will be equal to the number of users - i.e. 4
                    "Populate": {"num_requests": 4, "num_failures": 0},
                    # The number of Search requests should equal the number in the dataset
                    # (20 for mnist-test).
                    "Search": {
                        "num_requests": 20,
                        "num_failures": 0,
                        "recall": check_recall_stats,
                    },
                },
                "test2": {
                    "Populate": {"num_requests": 4, "num_failures": 0},
                    "Search": {
                        "num_requests": 20,
                        "num_failures": 0,
                        "recall": check_recall_stats,
                    },
                },
            },
        )

    def test_mnist_skip_populate(self):
        # Test that skip_populate doesn't re-populate data.

        # Run once to initially populate. Using ivfflat to increase coverage
        # of that index type across tests.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", extra_args=["--pgvector_index_type=ivfflat"]
        )
        assert proc.returncode == 0
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 1, "num_failures": 0},
                "Search": {"num_requests": 20, "num_failures": 0},
            },
        )

        # Run again without population
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            extra_args=["--pgvector_index_type=ivfflat", "--skip_populate"],
        )
        assert proc.returncode == 0
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_filtered(self):
        # Tests a workload with metadata and filtering (such as YFCC-test).
        (proc, stdout, stderr) = spawn_vsb(
            workload="yfcc-test",
            extra_args=["--users=10"],
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
