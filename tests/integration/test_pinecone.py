import datetime
import json
import os
import random
import string
import sys
import subprocess
from subprocess import PIPE

import pytest
from pinecone import Pinecone


def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception(f"Environment variable {name} is not set")
    return value


def random_string(length):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


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
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    now = now.replace(":", "-")
    index_name = read_env_var("NAME_PREFIX") + "--" + now + "--" + random_string(10)
    index_name = index_name.lower()
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
    pc.delete_index(name=index_name)


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


def spawn_vsb(workload, api_key, index_name, timeout=60, extra_args=None):
    """Spawn an instance of vsb with the given arguments, returning the proc object,
    its stdout and stderr.
    """
    return spawn_vsb_inner(workload, api_key, index_name, timeout, extra_args)


def spawn_vsb_inner(
    workload, api_key=None, index_name=None, timeout=60, extra_args=None
):
    """Inner function to spawn an instance of vsb with the given arguments,
    returning the proc object, its stdout and stderr.
    Allows api_key and index_name to be omitted, to test how vsb handles that.
    """
    if extra_args is None:
        extra_args = []
    env = os.environ.copy()
    if api_key is not None:
        env.update({"VSB__PINECONE_API_KEY": api_key})
    args = [
        "./vsb.py",
        "--database",
        "pinecone",
        "--workload",
        workload,
        "--json",
        "--loglevel=DEBUG",
    ]
    if index_name:
        args += ["--pinecone_index_name", index_name]
    proc = subprocess.Popen(
        args + extra_args,
        stdout=PIPE,
        stderr=PIPE,
        env=env,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # kill process and capture as much stdout / stderr as we can.
        proc.kill()
        stdout, stderr = proc.communicate()
        print(stdout)
        print(stderr, file=sys.stderr)
        raise
    # Echo subprocesses stdout & stderr to our own, so pytest can capture and
    # report them on error.
    print(stdout)
    print(stderr, file=sys.stderr)
    return proc, stdout, stderr


def parse_stats_to_json(stdout: str) -> list(dict()):
    """
    Parse stdout from VSB into a list of JSON dictionaries, one for each
    worker which reported stats.
    """
    # For each task type (Populate, Search, ...) we see a JSON object,
    # so must handle multiple JSON objects in stdout.
    stats = []
    while stdout:
        try:
            stats += json.loads(stdout)
            break
        except json.JSONDecodeError as e:
            stdout = stdout[e.pos :]
    return stats


def check_request_counts(stdout, expected: dict()):
    stats = parse_stats_to_json(stdout)
    by_method = {s["method"]: s for s in stats}
    for phase, expected_stats in expected.items():
        assert phase in by_method, f"Missing stats for expected phase '{phase}'"
        for ex_name, ex_value in expected_stats.items():
            actual_value = by_method[phase][ex_name]
            if callable(ex_value):
                assert ex_value(actual_value), (
                    f"For phase {phase} and " f"stat {ex_name}"
                )
            else:
                assert actual_value == ex_value, (
                    f"For phase {phase} and " f"stat {ex_name}"
                )


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
                    "recall": lambda x: len(x) == 20,
                },
            },
        )

    def test_mnist_concurrent(self, api_key, pinecone_index_mnist):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent users
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=pinecone_index_mnist,
            extra_args=["--users=4"],
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
                "Search": {"num_requests": 20, "num_failures": 0},
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
                # TODO: We should only issue each search query once, but currently
                # we perform the query once per process (4)
                "Search": {"num_requests": 20 * 4, "num_failures": 0},
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
                "Search": {"num_requests": 20, "num_failures": 0},
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
                "Search": {"num_requests": 20, "num_failures": 0},
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
                "Search": {"num_requests": 500, "num_failures": 0},
            },
        )

    def test_required_args(self, api_key, pinecone_index_mnist):
        # Tests that all required args are correctly checked for at vsb startup.
        (proc, stdout, stderr) = spawn_vsb_inner(api_key=api_key, workload="mnist-test")
        assert proc.returncode == 2
        assert (
            "The following arguments must be specified when --database is "
            "'pinecone':"
        ) in stderr
        assert "--pinecone_index_name" in stderr

        (proc, stdout, stderr) = spawn_vsb_inner(
            index_name=pinecone_index_mnist, workload="mnist-test"
        )
        assert proc.returncode == 2
        assert (
            "The following arguments must be specified when --database is "
            "'pinecone':"
        ) in stderr
        assert "--pinecone_api_key" in stderr
