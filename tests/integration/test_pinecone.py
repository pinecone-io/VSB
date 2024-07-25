import os

import pytest
from pinecone import Pinecone

from conftest import (
    check_request_counts,
    read_env_var,
    uuid,
    spawn_vsb_inner,
    check_recall_stats,
)


@pytest.fixture
def api_key():
    host = os.environ.get("PINECONE_API_KEY", None)
    if host is None or host == "":
        raise Exception(
            "PINECONE_API_KEY environment variable is not set. Set to a Pinecone API "
            "key suitable for testing against."
        )
    return host


def _create_pinecone_index(dims: int, metric: str) -> str:
    pc = Pinecone(api_key=read_env_var("PINECONE_API_KEY"))
    index_name = uuid()
    environment = os.environ.get("ENVIRONMENT")
    if environment:
        spec = {"pod": {"environment": environment, "pod_type": "p1.x1"}}
    else:
        spec = {
            "serverless": {
                "cloud": read_env_var("SERVERLESS_CLOUD"),
                "region": read_env_var("SERVERLESS_REGION"),
            }
        }
    pc.create_index(index_name, dims, spec, metric)
    return index_name


def _delete_pinecone_index(index_name: str):
    pc = Pinecone(api_key=read_env_var("PINECONE_API_KEY"))
    pc.delete_index(name=index_name, timeout=-1)


@pytest.fixture(scope="module")
def pinecone_index_mnist():
    index_name = _create_pinecone_index(dims=784, metric="euclidean")
    yield index_name
    _delete_pinecone_index(index_name)


@pytest.fixture(scope="module")
def pinecone_index_yfcc():
    index_name = _create_pinecone_index(dims=192, metric="euclidean")
    yield index_name
    _delete_pinecone_index(index_name)


def spawn_vsb(
    workload,
    api_key=None,
    index_name=None,
    index_spec=None,
    timeout=60,
    extra_args=None,
):
    """Spawn an instance of vsb with the given arguments, returning the proc object,
    its stdout and stderr.
    """
    args = []
    if index_name:
        args += ["--pinecone_index_name", index_name]
    if index_spec:
        args += ["--pinecone_index_spec", index_spec]
    if extra_args:
        args += extra_args
    extra_env = {}
    if api_key:
        extra_env.update({"VSB__PINECONE_API_KEY": api_key})
    return spawn_vsb_inner("pinecone", workload, timeout, args, extra_env)


class TestPinecone:
    def test_mnist_single(self, api_key, pinecone_index_mnist):
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", api_key=api_key, index_name=pinecone_index_mnist
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 2, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_concurrent(self, api_key, pinecone_index_mnist):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent users, and with a request limit set.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
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
                "Populate": {"num_requests": 4, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_multiprocess(self, api_key, pinecone_index_mnist):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
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
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-double-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "test1.Populate": {"num_requests": 2, "num_failures": 0},
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "test1.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
                "test2.Populate": {"num_requests": 2, "num_failures": 0},
                "test2.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_mnist_double_concurrent(self, api_key, pinecone_index_mnist):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent users, and with a request rate limit set.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-double-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
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

    def test_mnist_double_multiprocess(self, api_key, pinecone_index_mnist):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-double-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
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

    def test_mnist_skip_populate(self, api_key, pinecone_index_mnist):
        # Test that skip_populate doesn't re-populate data.

        # Run once to initially populate.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", api_key=api_key, index_name=pinecone_index_mnist
        )
        assert proc.returncode == 0
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 2, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

        # Run again without population
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
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

    def test_filtered(self, api_key, pinecone_index_yfcc):
        # Tests a workload with metadata and filtering (such as YFCC-test).
        (proc, stdout, stderr) = spawn_vsb(
            workload="yfcc-test",
            api_key=api_key,
            index_name=pinecone_index_yfcc,
            extra_args=["--users=10"],
        )
        assert proc.returncode == 0
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 210, "num_failures": 0},
                "Search": {
                    "num_requests": 500,
                    "num_failures": 0,
                    "recall": check_recall_stats,
                },
            },
        )

    def test_required_args(self, api_key, pinecone_index_mnist):
        # Tests that all required args are correctly checked for at vsb startup.
        (proc, stdout, stderr) = spawn_vsb(
            index_name=pinecone_index_mnist, workload="mnist-test"
        )
        assert proc.returncode == 2
        assert (
            "The following arguments must be specified when --database is "
            "'pinecone':"
        ) in stderr
        assert "--pinecone_api_key" in stderr

    def test_invalid_index(self, api_key, pinecone_index_mnist):
        # Tests that specifying an improperly configured index is reported gracefully,
        # without printing additional metrics / stats (which could suggest the experiment ran correctly.
        (proc, stdout, stderr) = spawn_vsb(
            workload="yfcc-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
        )
        assert proc.returncode == 2
        assert "Response time percentiles" not in stdout
        assert "Saved stats to 'reports/" not in stdout

    def test_nonexistent_index(self, api_key):
        # Tests that specifying an index which doesn't exist informs the user that a new one is created.
        index_name = uuid()
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=index_name,
        )
        _delete_pinecone_index(index_name)
        assert proc.returncode == 0
        assert "Creating new index" in stdout
        assert "Saved stats to 'reports/" in stdout
