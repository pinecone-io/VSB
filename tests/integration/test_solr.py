import pytest
from conftest import (
    check_request_counts,
    spawn_vsb_inner,
    check_recall_stats,
)


def spawn_vsb_solr(workload, timeout=60, extra_args=None, **kwargs):
    """Spawn an instance of Solr vsb with the given arguments, returning the proc object,
    its stdout and stderr.
    """
    extra_env = {
        "VSB__SOLR_USERNAME": "solr",
        "VSB__SOLR_PASSWORD": "solr",
    }
    return spawn_vsb_inner("solr", workload, timeout, extra_args, extra_env)


class TestSolr:

    def test_required_args(self):
        # Tests that all required args are correctly checked for at vsb startup.
        (proc, stdout, stderr) = spawn_vsb_solr(workload="mnist-test")
        assert proc.returncode == 2
        assert (
            "The following arguments must be specified when --database is "
            "'solr':"
        ) in stderr

    def test_invalid_index(self):
        # Tests that specifying an improperly configured index is reported gracefully.
        (proc, stdout, stderr) = spawn_vsb_solr(workload="yfcc-test")
        assert proc.returncode == 2
        assert "Response time percentiles" not in stdout

    def test_nonexistent_index(self):
        # Tests that specifying an index which doesn't exist informs the user that a new one is created.
        (proc, stdout, stderr) = spawn_vsb_solr(workload="mnist-test")
        assert proc.returncode == 0
        assert "Creating new index" in stdout

    def test_overwrite(self):
        # Tests that attempting to populate an existing index fails if the
        # user doesn't specify --overwrite.
        (proc, stdout, stderr) = spawn_vsb_solr(
            workload="mnist-test",
            extra_args=["--no-overwrite"],
        )
        assert proc.returncode == 2
        assert "cowardly refusing to overwrite existing data." in stdout
