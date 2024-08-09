import os
import pytest
from conftest import (
    check_request_counts,
    check_recall_stats,
    spawn_vsb_inner,
)
from test_pinecone import (
    pinecone_api_key,
    pinecone_index_mnist,
    pinecone_index_yfcc,
)


# We need to make spawn_vsb_pinecone a fixture so it can capture pinecone-specific fixtures
def spawn_vsb_pinecone(
    workload,
    pinecone_api_key,
    pinecone_index_mnist,
    pinecone_index_yfcc,
    timeout=60,
    extra_args=None,
    **kwargs,
):
    """Spawn an instance of pinecone vsb with the given arguments, returning the proc object,
    its stdout and stderr.
    """
    args = []
    match workload:
        case "mnist-test" | "mnist-double-test":
            args += ["--pinecone_index_name", pinecone_index_mnist]
        case "yfcc-test":
            args += ["--pinecone_index_name", pinecone_index_yfcc]
        case _:
            raise ValueError(f"Specify an index name fixture for: {workload}")
    if extra_args:
        args += extra_args
    extra_env = {}
    extra_env.update({"VSB__PINECONE_API_KEY": pinecone_api_key})
    return spawn_vsb_inner("pinecone", workload, timeout, args, extra_env)


def spawn_vsb_pgvector(workload, timeout=60, extra_args=None, **kwargs):
    """Spawn an instance of pgvector vsb with the given arguments, returning the proc object,
    its stdout and stderr.
    """
    extra_env = {
        "VSB__PGVECTOR_USERNAME": "postgres",
        "VSB__PGVECTOR_PASSWORD": "postgres",
    }
    return spawn_vsb_inner("pgvector", workload, timeout, extra_args, extra_env)


@pytest.mark.parametrize("spawn_vsb", [spawn_vsb_pgvector, spawn_vsb_pinecone])
class TestCommon:

    def test_mnist_single(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
            workload="mnist-test",
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

    def test_mnist_concurrent(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent users
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
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

    def test_mnist_multiprocess(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
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

    def test_mnist_double(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
            workload="mnist-double-test",
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "test1.Populate": {"num_requests": 1, "num_failures": 0},
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "test1.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
                "test2.Populate": {"num_requests": 1, "num_failures": 0},
                "test2.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_double_concurrent(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent users, and with a request rate limit set.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
            workload="mnist-double-test",
            extra_args=["--users=4", "--requests_per_sec=40"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Given the size of each
                # chunk will be less than the batch size (600 / 4 < 200), then the
                # number of requests will be equal to the number of users - i.e. 4
                "test1.Populate": {"num_requests": 4, "num_failures": 0},
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "test1.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
                "test2.Populate": {"num_requests": 4, "num_failures": 0},
                "test2.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_double_multiprocess(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
            workload="mnist-double-test",
            extra_args=["--processes=4", "--users=4"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Given the size of each
                # chunk will be less than the batch size (600 / 4 < 200), then the
                # number of requests will be equal to the number of users - i.e. 4
                "test1.Populate": {"num_requests": 4, "num_failures": 0},
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "test1.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
                "test2.Populate": {"num_requests": 4, "num_failures": 0},
                "test2.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_skip_populate(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Test that skip_populate doesn't re-populate data.

        # Run once to initially populate. Using ivfflat to increase coverage
        # of that index type across tests.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
            workload="mnist-test",
            extra_args=["--pgvector_index_type=ivfflat"],
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
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
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

    def test_filtered(
        self, spawn_vsb, pinecone_api_key, pinecone_index_mnist, pinecone_index_yfcc
    ):
        # Tests a workload with metadata and filtering (such as YFCC-test).
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index_mnist=pinecone_index_mnist,
            pinecone_index_yfcc=pinecone_index_yfcc,
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
