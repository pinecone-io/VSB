"""locustfile.py is the entrypoint for VSB as loaded by locust.
Its purpose is to:
- Hook into locust events to add VSB-specific behaviour (our own command-line
  arguments, initialise the Workload and Database for each Runner).
- Import the locust.User subclasses (user.*User) which locust will auto-register and
use to drive all our benchmarks.
"""

from vsb.cmdline_args import add_vsb_cmdline_args
from vsb.databases import Database
from vsb.workloads import Workload

import grpc.experimental.gevent as grpc_gevent
from locust import events
from locust.runners import WorkerRunner
from locust_plugins.distributor import Distributor
import locust.stats

# Note: These are _not_ unused, they are required to register our User
# and custom LoadShape classes with locust.
import users
from users import PopulateUser, RunUser, LoadShape

# patch grpc so that it uses gevent instead of asyncio. This is required to
# allow the multiple coroutines used by locust to run concurrently. Without it
# (using default asyncio) will block the whole Locust/Python process,
# in practice limiting to running a single User per worker process.
grpc_gevent.init_gevent()

# Display stats of benchmark so far to console very 5 seconds.
locust.stats.CONSOLE_STATS_INTERVAL_SEC = 5


@events.init_command_line_parser.add_listener
def on_locust_init_cmd_line_parser(parser):
    """Add the VSB-specific cmdline arguments to locust's parser, so it
    can correctly parse them and make available to VSB code.
    """
    add_vsb_cmdline_args(parser, include_locust_args=False)


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    # Override spwan rate - we want all clients to start at ~the same time.
    env = environment
    options = env.parsed_options
    num_users = options.num_users or 1
    options.spawn_rate = num_users

    # Create a Distributor which assigns monotonically user_ids to each
    # VectorSearchUser, so we can split the workload between them.
    users.distributors["user_id.populate"] = Distributor(
        environment,
        iter(range(num_users)),
        "user_id.populate",
    )
    users.distributors["user_id.run"] = Distributor(
        environment,
        iter(range(num_users)),
        "user_id.run",
    )
    if not isinstance(environment.runner, WorkerRunner):
        # For Worker runners workload setup is deferred until the test_start
        # event, to avoid multiple processes trying to download at the same time.
        env.workload = Workload(options.workload).get_class()(options.cache_dir)
        env.database = Database(options.database).get_class()(
            dimensions=env.workload.dimensions,
            metric=env.workload.metric,
            config=vars(options),
        )


@events.test_start.add_listener
def setup_worker_dataset(environment, **_kwargs):
    # happens only once in headless runs, but can happen multiple times in web ui-runs
    # in a distributed run, the master does not typically need any test data
    if isinstance(environment.runner, WorkerRunner):
        # Make the Workload available for WorkerRunners (non-Worker will have
        # already setup the dataset via on_locust_init).
        #
        # We need to perform this work in a background thread (not in
        # the current gevent greenlet) as otherwise we block the
        # current greenlet (pandas data loading is not
        # gevent-friendly) and locust's master / worker heartbeating
        # thinks the worker has gone missing and can terminate it.
        #        pool = gevent.get_hub().threadpool
        #        environment.setup_dataset_greenlet = pool.apply_async(setup_dataset,
        #                                                              kwds={
        #                                                              'environment':environment,
        #
        #                                                                    'skip_download_and_populate':True})
        env = environment
        options = env.parsed_options
        env.workload = Workload(options.workload).get_class()(options.cache_dir)
        env.database = Database(options.database).get_class()(
            dimensions=env.workload.dimensions,
            metric=env.workload.metric,
            config=vars(options),
        )
