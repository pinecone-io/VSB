from conftest import (
    spawn_vsb_inner,
)


# used in test_common
def spawn_vsb_opensearch(workload, timeout=60, extra_args=None, **kwargs):
    """Spawn an instance of pgvector vsb with the given arguments, returning the proc object,
    its stdout and stderr.
    """
    extra_env = {
        "VSB__PGVECTOR_USERNAME": "postgres",
        "VSB__PGVECTOR_PASSWORD": "postgres",
    }
    return spawn_vsb_inner("opensearch", workload, timeout, extra_args, extra_env)
