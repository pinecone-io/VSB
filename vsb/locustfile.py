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

import vsb
from vsb.cmdline_args import add_vsb_cmdline_args
from vsb.databases import Database
from vsb.workloads import (
    Workload,
    WorkloadSequence,
    build_workload_sequence,
    VectorWorkloadSequence,
)
from vsb import console, logger
from locust import events, log
from locust.runners import WorkerRunner, MasterRunner
from locust_plugins.distributor import Distributor
from vsb.subscriber import Subscriber
from gevent.event import AsyncResult
import gevent
import locust.stats

# Note: These are _not_ unused, they are required to register our User
# and custom LoadShape classes with locust.
import users
from users import SetupUser, PopulateUser, RunUser, LoadShape

# Display stats of benchmark so far to console very 5 seconds.
locust.stats.CONSOLE_STATS_INTERVAL_SEC = 5


# class IterationHelper:
#     """
#     Helper class that keeps track of workload iteration in WorkloadSequences.
#     In distributed runs, each WorkerRunner will have its own instance of this class.
#     In local runs, there is one instance managed by the LocalRunner.
#     """

#     def __init__(self, workload_sequence: VectorWorkloadSequence):
#         self.iteration = 0
#         self.workload_sequence = workload_sequence
#         self.workload = next(workload_sequence)
#         self.record_count = self.workload.record_count()

#     def next(self):
#         self.iteration += 1
#         # This may raise StopIteration, to be caught by the caller.
#         self.workload = next(self.workload_sequence)
#         # env.record_count represents the cumulative record count so far,
#         # and is thus independent from the current workload's count.
#         self.record_count += self.workload.record_count()

@events.init_command_line_parser.add_listener
def on_locust_init_cmd_line_parser(parser):
    """Add the VSB-specific cmdline arguments to locust's parser, so it
    can correctly parse them and make available to VSB code.
    """
    add_vsb_cmdline_args(parser, include_locust_args=False)


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    """Hook into the locust init event to setup the VSB environment.
    This is called once per run for each process.
    """


def setup_environment(environment, **_kwargs):
    env = environment
    options = env.parsed_options
    num_users = options.num_users or 1

    logger.debug(f"on_locust_init(): runner={type(environment.runner)}")

    # Load the WorkloadSequence
    environment.workload_sequence = build_workload_sequence(
        options.workload, cache_dir=options.cache_dir
    )

    # Reset distributors for new Populate -> Run iteration
    phases = ["setup", "populate", "run"]
    # Distributors can only be initialized once per name, so we need
    # a unique phase name for each iteration: ex. "user_id.populate.2"
    for phase in [
        f"user_id.{p}.{i}"
        for p in phases
        for i in range(env.workload_sequence.workload_count())
    ]:
        users.distributors[phase] = Distributor(env, iter(range(num_users)), phase)

    logger.debug(
        f"Workload sequence: {environment.workload_sequence.name}, runner={type(environment.runner)}"
    )

    if isinstance(environment.runner, WorkerRunner):
        # In distributed mode, we only want to log problems to the console,
        # (a) because things get noisy if we log the same info from multiple
        # workers, and (b) because logs from non-master will corrupt the
        # progress bar display.
        vsb.logger.setLevel(logging.ERROR)


def setup_runner(env):
    options = env.parsed_options

    setup_environment(env)

    logger.info(
        f"Workload '{env.workload_sequence.name}' initialized "
        f"record_count={env.workload_sequence[0].record_count()} "
        f"dimensions={env.workload_sequence[0].dimensions()} "
        f"metric={env.workload_sequence[0].metric()} "
    )


# Note that this listener is guaranteed to finish before Users are spawned, but not
# before LoadShape is initialized and potentially goes through Init; be careful 
# of accessing env.iteration_helper or other environment attributes set up in
# setup_environment() before this event listener finishes.
@events.test_start.add_listener
def setup_worker_dataset(environment, **_kwargs):
    # happens only once in headless runs, but can happen multiple times in web ui-runs
    # in a distributed run, the master does not typically need any test data

    environment.iteration = 0
    users.subscribers["iteration"] = Subscriber(environment, environment.iteration, "iteration")

    # if not isinstance(environment.runner, MasterRunner):
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
            record_count=environment.workload_sequence[0].record_count(),
            dimensions=environment.workload_sequence[0].dimensions(),
            metric=environment.workload_sequence[0].metric(),
            name=environment.workload_sequence.name,
            config=vars(options),
        )
    except StopUser:
        # This is a special exception that is raised when we want to
        # stop the User from running, e.g. because the database
        # connection failed.
        log.unhandled_greenlet_exception = True
        environment.runner.quit()
    except:
        logger.error(
            "Uncaught exception in during setup - quitting: \n%s",
            traceback.format_exc(),
        )
        log.unhandled_greenlet_exception = True
        environment.runner.quit()
    


@events.quit.add_listener
def reset_console_on_quit(exit_code):
    """
    If VSB terminates uncleanly, then the console cursor may still be hidden
    as the progress bars hide it when rendered. Ensure the cursor is made
    visible again.
    """
    vsb.console.show_cursor()
