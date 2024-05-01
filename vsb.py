#!/usr/bin/env python3
import time
import traceback
from enum import Enum, auto
import grpc.experimental.gevent as grpc_gevent
from locust import User, events, task
from locust.env import Environment
from locust.exception import StopUser
from locust.log import setup_logging
from locust.runners import WorkerRunner
from locust.stats import (
    stats_history,
    stats_printer,
    print_stats,
    print_percentile_stats,
)

from vsb.databases import Database
from vsb.workloads import Workload
import configargparse
import gevent
import locust.stats
import logging


# patch grpc so that it uses gevent instead of asyncio. This is required to
# allow the multiple coroutines used by locust to run concurrently. Without it
# (using default asyncio) will block the whole Locust/Python process,
# in practice limiting to running a single User per worker process.
grpc_gevent.init_gevent()

setup_logging("INFO")
# Display stats of benchmark so far to console very 5 seconds.
locust.stats.CONSOLE_STATS_INTERVAL_SEC = 5


class VectorSearchUser(User):
    class Phase(Enum):
        Load = auto()
        Run = auto()

    """Represents a single user (aka client) performing requests against
    a particular Backend."""

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
                # need to trigger this in a separate greenlet, in case test_stop handlers do something async
                gevent.spawn_later(0.1, runner.quit)
            raise StopUser()


def main():
    parser = configargparse.ArgumentParser(
        prog="VCB", description="Vector Search Bench"
    )
    parser.add_argument(
        "--database",
        required=True,
        choices=tuple(e.value for e in Database),
        help="The vector search database to test",
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=tuple(e.value for e in Workload),
        help="The workload to run",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=1,
        help="Number of clients concurrently accessing the database",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/VSB/cache",
        help="Directory to store downloaded datasets",
    )

    pinecone_group = parser.add_argument_group("Options specific to pinecone database")
    pinecone_group.add_argument(
        "--pinecone_api_key",
        type=str,
        help="API Key to connect to Pinecone index",
        env_var="VSB__PINECONE_API_KEY",
    )
    pinecone_group.add_argument(
        "--pinecone_index_name",
        type=str,
        help="Name of Pinecone index to connect to",
        env_var="VSB__PINECONE_INDEX_NAME",
    )

    options = parser.parse_args()

    # setup Environment and Runner
    env = Environment(user_classes=[VectorSearchUser], events=events)
    env.options = options

    env.workload = Workload(env.options.workload).get_class()(options.cache_dir)
    env.database = Database(env.options.database).get_class()(
        dimensions=env.workload.dimensions,
        metric=env.workload.metric,
        config=vars(options),
    )

    runner = env.create_local_runner()

    # start a WebUI instance
    # web_ui = env.create_web_ui("127.0.0.1", 8089)

    # execute init event handlers (only really needed if you have registered any)
    env.events.init.fire(environment=env, runner=runner)  # , web_ui=web_ui)

    # start a greenlet that periodically outputs the current stats
    gevent.spawn(stats_printer(env.stats))

    # start a greenlet that save current stats to history
    gevent.spawn(stats_history, env.runner)

    # start the test
    runner.start(options.clients, spawn_rate=10)

    # wait for the greenlets
    runner.greenlet.join()

    # stop the web server for good measures
    #    web_ui.stop()

    print_stats(runner.stats, current=False)
    print_percentile_stats(runner.stats)


if __name__ == "__main__":
    main()
