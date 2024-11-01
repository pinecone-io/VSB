import json
import os
import random
import datetime
import re
import string
import subprocess
import sys
from subprocess import PIPE


def read_env_var(name):
    value = os.environ.get(name)
    if value is None:
        raise Exception(f"Environment variable {name} is not set")
    return value


def uuid() -> str:
    """Returns a lowercase unique ID."""
    now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    now = now.replace(":", "-")
    randstr = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    id = read_env_var("NAME_PREFIX") + "--" + now + "--" + randstr
    return id.lower()


def parse_stats_to_json(stdout: str) -> list[dict]:
    """
    Parse stdout from VSB to locate the ststs.json path, then parse that file
    into a list of JSON dictionaries, one for each worker which reported stats.
    """
    # For each task type (Populate, Search, ...) we see a JSON object,
    # so must handle multiple JSON objects in stdout.
    pattern = r"Saved stats to '([^']+)'"
    if m := re.search(pattern, stdout):
        with open(m.group(1)) as f:
            return json.load(f)
    raise Exception("Failed to find stats.json path in stdout")


def check_recall_stats(actual: dict) -> bool:
    """Check that the recall stats are present and have the expected structure."""
    return all(key in actual for key in ["min", "max", "mean", "percentiles"]) and all(
        str(pct) in actual["percentiles"].keys()
        for pct in [0.1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99]
    )


def check_recall_correctness(p95_threshold: int):
    """
    Check that the recall stats are present, have the expected structure, and
    that the values are within the expected range.

    Use is intended for tests in which there is an expected approximate recall
    score (e.g. for a known dataset + database configuration), and the test
    should fail if the recall is below a certain threshold.
    """
    # Because of the metric structure, "5th percentile" is actually the 95th for recall (higher is better).
    # Evil currying hack because we want a callable to pass to check_request_counts.
    return lambda actual: (
        check_recall_stats(actual) and actual["percentiles"]["5"] >= p95_threshold
    )


def check_request_counts(stdout, expected: dict) -> None:
    """Given stdout in JSON format from vsb and a dict of expected elements
    to find in stdout, check all expected fields are present, asserting on
    any mismatches.

    Elements of expected should be nested dicts where their top-level keys
    are request_type names (Populate, Search, ...) and the values are dicts
    of expected elements to find for each request type - e.g.

        {
            "Search": {
                 "num_requests": 20,
                 "num_failures": 0,
            },
        }

    The values of the leaf elements can either be literals (e.g. '20' for
    "num_requests" above) or callables, if more complex checks are needed -
    for example to check that "recall" is a list of 20 elements (but not
    checking what the individual values are):

        {
            "Search": { "Recall": lambda x: len(x) == 20, }
        }
    """
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


def spawn_vsb_inner(
    database,
    workload,
    timeout=60,
    extra_args: list = None,
    extra_env: dict = None,
):
    """Inner function to spawn an instance of vsb with the given arguments,
    returning the proc object, its stdout and stderr.
    Allows api_key and index_name to be omitted, to test how vsb handles that.
    """
    if extra_args is None:
        extra_args = []
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    args = [
        "vsb",
        f"--database={database}",
        f"--workload={workload}",
        "--loglevel=debug",
        "--overwrite",
    ]
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
