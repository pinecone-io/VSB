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


@pytest.fixture(scope="module")
def pinecone_api_key():
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


@pytest.fixture(scope="module")
def pinecone_index_synthetic():
    index_name = _create_pinecone_index(dims=192, metric="cosine")
    yield index_name
    _delete_pinecone_index(index_name)


def spawn_vsb(
    workload,
    api_key=None,
    index_name=None,
    index_spec=None,
    timeout=60,
    extra_args: list = None,
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


# used in test_common
def spawn_vsb_pinecone(
    workload,
    pinecone_api_key,
    pinecone_index,
    timeout=60,
    extra_args=None,
    **kwargs,
):
    """Spawn an instance of pinecone vsb with the given arguments, returning the proc object,
    its stdout and stderr.
    """
    args = ["--pinecone_index_name", pinecone_index]
    if extra_args:
        args += extra_args
    extra_env = {}
    extra_env.update({"VSB__PINECONE_API_KEY": pinecone_api_key})
    return spawn_vsb_inner("pinecone", workload, timeout, args, extra_env)


class TestPinecone:

    def test_required_args(self, pinecone_api_key, pinecone_index_mnist):
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

    def test_invalid_index(self, pinecone_api_key, pinecone_index_mnist):
        # Tests that specifying an improperly configured index is reported gracefully,
        # without printing additional metrics / stats (which could suggest the experiment ran correctly.
        (proc, stdout, stderr) = spawn_vsb(
            workload="yfcc-test",
            api_key=pinecone_api_key,
            index_name=pinecone_index_mnist,
        )
        assert proc.returncode == 2
        assert "Response time percentiles" not in stdout
        assert "Saved stats to 'reports/" not in stdout

    def test_nonexistent_index(self, pinecone_api_key):
        # Tests that specifying an index which doesn't exist informs the user that a new one is created.
        index_name = uuid()
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=pinecone_api_key,
            index_name=index_name,
        )
        _delete_pinecone_index(index_name)
        assert proc.returncode == 0
        assert "Creating new index" in stdout
        assert "Saved stats to 'reports/" in stdout

    def test_overwrite(self, pinecone_api_key, pinecone_index_mnist):
        # Tests that attempting to populate an existing index fails if the
        # user doesn't specify --overwrite (recall that we create a new index
        # outside of VSB in the harness via pinecone_index_mnist).
        # Note that spawn_vsb() always specifies --overwrite, so we need to
        # additionally add --no-overwrite to the args here.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=pinecone_api_key,
            index_name=pinecone_index_mnist,
            extra_args=["--no-overwrite"],
        )
        assert proc.returncode == 2
        assert "cowardly refusing to overwrite existing data. " in stdout
