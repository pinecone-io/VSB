import json

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
            "--json",
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
    for phase, stats in expected.items():
        assert phase in by_method, f"Missing stats for expected phase '{phase}'"
        for stat in stats:
            assert by_method[phase][stat] == stats[stat], (
                f"For phase {phase} and " f"stat {stat}"
            )


class TestPinecone:
    def test_mnist_single(self, api_key, index_name):
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", api_key=api_key, index_name=index_name
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 600 / 200, "num_failures": 0},
                "Search": {"num_requests": 20, "num_failures": 0},
            },
        )

    def test_mnist_concurrent(self, api_key, index_name):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent users
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=index_name,
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

    def test_mnist_multiprocess(self, api_key, index_name):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=index_name,
            extra_args=["--processes=2", "--users=4"],
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
                # we perform the query once per process (2)
                "Search": {"num_requests": 20 * 2, "num_failures": 0},
            },
        )

    def test_mnist_skip_populate(self, api_key, index_name):
        # Test that skip_populate doesn't re-populate data.

        # Run once to initially populate.
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test", api_key=api_key, index_name=index_name
        )
        assert proc.returncode == 0
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {"num_requests": 600 / 200, "num_failures": 0},
                "Search": {"num_requests": 20, "num_failures": 0},
            },
        )

        # Run again without population
        (proc, stdout, stderr) = spawn_vsb(
            workload="mnist-test",
            api_key=api_key,
            index_name=index_name,
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
