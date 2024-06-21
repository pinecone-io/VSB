import datetime
from conftest import (
    check_request_counts,
    read_env_var,
    random_string,
    spawn_vsb_inner,
    check_recall_stats,
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
                # TODO: We should only issue each search query once, but currently
                # we perform the query once per process (2)
                "Search": {
                    "num_requests": 20 * 2,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_skip_populate(self):
        # Test that skip_populate doesn't re-populate data.

        # Run once to initially populate.
        (proc, stdout, stderr) = spawn_vsb(workload="mnist-test")
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
            extra_args=["--skip_populate"],
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
        # Test default value of 0 (unset).
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test"
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
