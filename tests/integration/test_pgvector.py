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


def _get_index_name() -> str:
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    now = now.replace(":", "-")
    index_name = read_env_var("NAME_PREFIX") + "--" + now + "--" + random_string(10)
    index_name = index_name.lower()


def spawn_vsb(workload, timeout=60, extra_args=None):
    if extra_args is None:
        extra_args = []
    env = os.environ
    env.update(
        {"VSB__PGVECTOR_USERNAME": "postgres", "VSB__PGVECTOR_PASSWORD": "postgres"}
    )
    proc = subprocess.Popen(
        [
            "./vsb.py",
            "--database",
            "pgvector",
            "--workload",
            workload,
            "--json",
            "--loglevel=DEBUG",
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
                "Search": {"num_requests": 20, "num_failures": 0},
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
                "Search": {"num_requests": 20, "num_failures": 0},
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
                "Search": {"num_requests": 20 * 2, "num_failures": 0},
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
                "Search": {"num_requests": 20, "num_failures": 0},
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
                "Search": {"num_requests": 500, "num_failures": 0},
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
                "Search": {"num_requests": 20, "num_failures": 0},
            },
        )
