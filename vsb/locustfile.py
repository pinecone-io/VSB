"""locustfile.py is the entrypoint for VSB as loaded by locust.
Its purpose is to:
- Hook into locust events to add VSB-specific behaviour (our own command-line
  arguments, initialise the Workload and Database for each Runner).
- Import the locust.User subclasses (user.*User) which locust will auto-register and
use to drive all our benchmarks.
"""

import logging
import traceback

from locust.exception import StopUser

from vsb.cmdline_args import add_vsb_cmdline_args
from vsb.databases import Database
from vsb.workloads import Workload

from locust import events, log
from locust.runners import WorkerRunner, MasterRunner
from locust_plugins.distributor import Distributor
import gevent
import locust.stats

# Note: These are _not_ unused, they are required to register our User
# and custom LoadShape classes with locust.
import users
from users import SetupUser, PopulateUser, RunUser, LoadShape

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

    # Create Distributors which assigns monotonically user_ids to each
    # of the different USer phasesUser, so we can split the workload between
    # them.
    phases = ["setup", "populate", "run"]
    for phase in ["user_id." + p for p in phases]:
        users.distributors[phase] = Distributor(
            environment, iter(range(num_users)), phase
        )


def setup_runner(env):
    options = env.parsed_options
    env.workload = Workload(options.workload).build(cache_dir=options.cache_dir)


@events.test_start.add_listener
def setup_worker_dataset(environment, **_kwargs):
    # happens only once in headless runs, but can happen multiple times in web ui-runs
    # in a distributed run, the master does not typically need any test data
    if not isinstance(environment.runner, MasterRunner):
        # Make the Workload available for non-MasterRunners (MasterRunners
        # only orchestrate the run when --processes is used, they don't
        # perform any actual operations and hence don't need to load a copy
        # of the workload data).
        # We need to perform this work in a background thread (not in the current
        # gevent greenlet) as otherwise we block the current greenlet (pandas data
        # loading is not gevent-friendly) and locust's master / worker heartbeat
        # thinks the worker has gone missing and can terminate it.
        pool = gevent.get_hub().threadpool
        pool.apply(setup_runner, kwds={"env": environment})

        try:
            options = environment.parsed_options
            environment.database = Database(options.database).get_class()(
                dimensions=environment.workload.dimensions,
                metric=environment.workload.metric,
                name=environment.workload.name,
                config=vars(options),
            )
        except:
            logging.error(
                "Uncaught exception in during setup - quitting: \n%s",
                traceback.format_exc(),
            )
            log.unhandled_greenlet_exception = True
            environment.runner.quit()
