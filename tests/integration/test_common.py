"""
Common tests that should succeed on all databases.

To add a new database to the test suite, you should implement the following:

- Test[Database] class with database-specific tests in spawn_vsb_[database].py
- spawn_vsb_[database] function that takes a superset of the arguments of spawn_vsb_*
    - spawn_vsb_[database](workload, index_name, database_specific_arg, **kwargs)
    - should return a (proc, stdout, stderr) tuple with spawn_vsb_inner
- Add the spawn_vsb_[database] function to the parametrize list of TestCommon
- If you added database-specific arguments, add them to each spawn_vsb call in
    the test cases below.
"""

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
    pinecone_index_synthetic,
    spawn_vsb_pinecone,
)
from test_pgvector import spawn_vsb_pgvector
from test_opensearch import spawn_vsb_opensearch


@pytest.mark.parametrize(
    "spawn_vsb", [spawn_vsb_pgvector, spawn_vsb_pinecone, spawn_vsb_opensearch]
)
class TestCommon:

    # Unfortunately pytest won't let us selectively parametrize with fixtures, so
    # we have to pass in all database-specific fixtures to our parametrized
    # spawn_vsb functions.

    def test_mnist_single(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_mnist,
    ):
        # Test "-test" variant of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
            workload="mnist-test",
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records (600).
                "Populate": {"num_requests": lambda x: x < 600, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_mnist_concurrent(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_mnist,
    ):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent users
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
            workload="mnist-test",
            extra_args=["--users=4"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Different DBs
                # use different batch sizes, so just check we have fewer than
                # number of records (600) / number of users (4).
                "Populate": {"num_requests": lambda x: x < 600 / 4, "num_failures": 0},
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_mnist_multiprocess(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_mnist,
    ):
        # Test "-test" variant of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
            workload="mnist-test",
            extra_args=["--processes=2", "--users=4"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Different DBs
                # use different batch sizes, so just check we have fewer than
                # number of records (600) / number of users (4).
                "Populate": {"num_requests": lambda x: x < 600 / 4, "num_failures": 0},
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_mnist_double(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_mnist,
    ):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
            workload="mnist-double-test",
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records (600).
                "test1.Populate": {
                    "num_requests": lambda x: x < 600,
                    "num_failures": 0,
                },
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "test1.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
                "test2.Populate": {
                    "num_requests": lambda x: x < 600,
                    "num_failures": 0,
                },
                "test2.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_mnist_double_concurrent(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_mnist,
    ):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent users, and with a request rate limit set.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
            workload="mnist-double-test",
            extra_args=["--users=4", "--requests_per_sec=40"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Different DBs
                # use different batch sizes, so just check we have fewer than
                # number of records (600) / number of users (4).
                "test1.Populate": {
                    "num_requests": lambda x: x < 600 / 4,
                    "num_failures": 0,
                },
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "test1.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
                "test2.Populate": {
                    "num_requests": lambda x: x < 600 / 4,
                    "num_failures": 0,
                },
                "test2.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_mnist_double_multiprocess(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_mnist,
    ):
        # Test "-double-test" variant (WorkloadSequence) of mnist loads and runs successfully with
        # concurrent processes and users.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
            workload="mnist-double-test",
            extra_args=["--processes=4", "--users=4"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Different DBs
                # use different batch sizes, so just check we have fewer than
                # number of records (600) / number of users (4).
                "test1.Populate": {
                    "num_requests": lambda x: x < 600 / 4,
                    "num_failures": 0,
                },
                # The number of Search requests should equal the number in the dataset
                # (20 for mnist-test).
                "test1.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
                # For multiple users the populate phase will chunk the records to be
                # loaded into num_users chunks - i.e. 4 here. Different DBs
                # use different batch sizes, so just check we have fewer than
                # number of records (600) / number of users (4).
                "test2.Populate": {
                    "num_requests": lambda x: x < 600 / 4,
                    "num_failures": 0,
                },
                "test2.Search": {
                    "num_requests": 20,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_mnist_skip_populate(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_mnist,
    ):
        # Test that skip_populate doesn't re-populate data.

        # Run once to initially populate. Using ivfflat to increase coverage
        # of that index type across tests.
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
            workload="mnist-test",
            extra_args=["--pgvector_index_type=ivfflat"],
        )
        assert proc.returncode == 0
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records (600).
                "Populate": {"num_requests": lambda x: x < 600, "num_failures": 0},
                "Search": {"num_requests": 20, "num_failures": 0},
            },
        )

        # Run again without population
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_mnist,
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
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_filtered(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_yfcc,
    ):
        # Tests a workload with metadata and filtering (such as YFCC-test).
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_yfcc,
            workload="yfcc-test",
            extra_args=["--users=10"],
        )
        assert proc.returncode == 0
        check_request_counts(
            stdout,
            {
                # Populate num_requests counts batches, not individual records.
                "Populate": {
                    "num_requests": lambda x: x > 1 and x < 10000,
                    "num_failures": 0,
                },
                "Search": {
                    "num_requests": 500,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_synthetic(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_synthetic,
    ):
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_synthetic,
            workload="synthetic",
            extra_args=["--users=10", "--processes=2"],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "Populate": {"num_failures": 0},
                "Search": {
                    "num_requests": 100,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_synthetic_runbook(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_synthetic,
    ):
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_synthetic,
            workload="synthetic-runbook",
            extra_args=[
                "--users=2",
                "--processes=2",
                "--synthetic_steps=2",
                "--synthetic_records=1000",
                "--synthetic_requests=500",
            ],
        )
        assert proc.returncode == 0

        check_request_counts(
            stdout,
            {
                "Populate": {"num_failures": 0},
                "Search": {
                    "num_requests": 500,
                    "num_failures": 0,
                    "Recall": check_recall_stats,
                },
            },
        )

    def test_synthetic_proportional(
        self,
        spawn_vsb,
        pinecone_api_key,
        pinecone_index_synthetic,
    ):
        (proc, stdout, stderr) = spawn_vsb(
            pinecone_api_key=pinecone_api_key,
            pinecone_index=pinecone_index_synthetic,
            workload="synthetic-proportional",
            extra_args=[
                "--users=4",
                "--processes=2",
                "--synthetic_records=1000",
                "--synthetic_requests=1000",
                "--synthetic_query_ratio=0.2",
                "--synthetic_fetch_ratio=0.2",
                "--synthetic_delete_ratio=0.2",
                "--synthetic_insert_ratio=0.2",
                "--synthetic_update_ratio=0.2",
            ],
        )
        assert proc.returncode == 0

        # Each of the 5 request types should have around 200 requests each.
        check_request_counts(
            stdout,
            {
                "Populate": {"num_failures": 0},
                "Search": {
                    "num_requests": lambda x: (x >= 150 and x <= 250),
                    "num_failures": 0,
                },
                "Fetch": {
                    "num_requests": lambda x: (x >= 150 and x <= 250),
                    "num_failures": 0,
                },
                "Delete": {
                    "num_requests": lambda x: (x >= 150 and x <= 250),
                    "num_failures": 0,
                },
                "Insert": {
                    "num_requests": lambda x: (x >= 150 and x <= 250),
                    "num_failures": 0,
                },
                "Update": {
                    "num_requests": lambda x: (x >= 150 and x <= 250),
                    "num_failures": 0,
                },
            },
        )
