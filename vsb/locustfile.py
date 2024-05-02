"""locustfile.py is the entrypoint for VSB as loaded by locust.
Its purpose is to:
- Hook into locust events to add VSB-specific behaviour (our own command-line
  arguments, initialise the Workload and Database for each Runner).
- Define a locust.User subclass (VectorSearchUser) which locust will
  auto-register and use to drive all our benchmarks.
"""

import logging
import time
import traceback
from enum import Enum, auto

from vsb.cmdline_args import add_vsb_cmdline_args
from vsb.databases import Database
from vsb.workloads import Workload

import gevent
import grpc.experimental.gevent as grpc_gevent
from locust import User, task, events
from locust.exception import StopUser
from locust.runners import WorkerRunner
import locust.stats

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
    options.spawn_rate = options.num_users

    if not isinstance(environment.runner, WorkerRunner):
        # For Worker runners workload setup is deferred until the test starts,
        # to avoid multiple processes trying to downlaod at the same time.
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


class VectorSearchUser(User):
    """
    Represents a single user (aka client) performing requests against a
    particular Backend.
    """

    class Phase(Enum):
        Load = auto()
        Run = auto()

    def __init__(self, environment):
        super().__init__(environment)
        self.database = environment.database
        self.workload = environment.workload
        self.phase = VectorSearchUser.Phase.Load

    @task
    def request(self):
        match self.phase:
            case VectorSearchUser.Phase.Load:
                self.do_load()
            case VectorSearchUser.Phase.Run:
                self.do_run()

    def do_load(self):
        try:
            (tenant, vectors) = self.workload.next_record_batch()
            if vectors:
                index = self.database.get_namespace(tenant)

                start = time.perf_counter()
                index.upsert_batch(vectors)
                stop = time.perf_counter()

                elapsed_ms = (stop - start) * 1000.0
                self.environment.events.request.fire(
                    request_type="Populate",
                    name=self.workload.name,
                    response_time=elapsed_ms,
                    response_length=0,
                )
            else:
                logging.info("Completed Load phase, switching to Run phase")
                self.phase = VectorSearchUser.Phase.Run
        except Exception as e:
            traceback.print_exception(e)
            self.environment.runner.quit()
            raise StopUser

    def do_run(self):
        (tenant, request) = self.workload.next_request()
        if request:
            try:
                index = self.database.get_namespace(tenant)

                start = time.perf_counter()
                index.search(request)
                stop = time.perf_counter()

                elapsed_ms = (stop - start) * 1000.0
                self.environment.events.request.fire(
                    request_type="Search",
                    name=self.workload.name,
                    response_time=elapsed_ms,
                    response_length=0,
                )
            except Exception as e:
                traceback.print_exception(e)
                self.environment.runner.quit()
                raise StopUser
        else:
            # No more requests - stop this user.
            runner = self.environment.runner
            logging.info(f"User count: {runner.user_count}")
            if runner.user_count == 1:
                logging.info("Last user stopped, quitting runner")
                if isinstance(runner, WorkerRunner):
                    runner._send_stats()  # send a final report
                # need to trigger this in a separate greenlet, in case
                # test_stop handlers do something async
                gevent.spawn_later(0.1, runner.quit)
            raise StopUser()
