import pytest
import subprocess
import os
import sys
from subprocess import PIPE


@pytest.fixture
def index_name():
    host = os.environ.get("INDEX_NAME", None)
    if host is None or host == "":
        raise Exception(
            "INDEX_NAME environment variable is not set. Set to the host of a Pinecone index suitable for testing "
            "against."
        )
    return host


@pytest.fixture
def api_key():
    host = os.environ.get("PINECONE_API_KEY", None)
    if host is None or host == "":
        raise Exception(
            "PINECONE_API_KEY environment variable is not set. Set to a Pinecone API key suitable for testing against."
        )
    return host


def spawn_vsb(workload, api_key, index_name, timeout=60, extra_args=[]):
    env = os.environ
    env.update({"VSB__PINECONE_API_KEY": api_key})
    proc = subprocess.Popen(
        [
            "./vsb.py",
            "--database",
            "pinecone",
            "--workload",
            workload,
            "--pinecone_index_name",
            index_name,
        ]
        + extra_args,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # Echo whatever stdout / stderr we got so far, to aid in debugging
        if e.stdout:
            for line in e.stdout.decode(errors="replace").splitlines():
                print(line)
        if e.stderr:
            for line in e.stderr.decode(errors="replace").splitlines():
                print(line, file=sys.stderr)
        raise
    # Echo subprocesses stdout & stderr to our own, so pytest can capture and
    # report them on error.
    print(stdout)
    print(stderr, file=sys.stderr)
    return proc, stdout, stderr


class TestPinecone:
    def test_mnist(self, api_key, index_name):
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", api_key=api_key, index_name=index_name
        )
        # TODO: Check more here when vsb output is more structured.
        assert proc.returncode == 0

    def test_mnist_concurrent(self, api_key, index_name):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent users
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=index_name,
            extra_args=["--users=4"],
        )
        # TODO: Check more here when vsb output is more structured.
        assert proc.returncode == 0

    def test_mnist_multiprocess(self, api_key, index_name):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=index_name,
            extra_args=["--processes=2", "--users=4"],
        )
        # TODO: Check more here when vsb output is more structured.
        assert proc.returncode == 0
