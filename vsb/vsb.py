#!/usr/bin/env python3

from enum import Enum, auto
from locust import User, events, task, TaskSet
from locust.env import Environment
from locust.exception import StopUser
from locust.log import setup_logging
from locust.runners import WorkerRunner
from locust.stats import stats_history, stats_printer

from databases import Database
from workloads import Workload
import argparse
import gevent
import logging
import sys


setup_logging("INFO")


class VectorSearchUser(User):
    class Phase(Enum):
        Load = auto()
        Run = auto()

    """Represents a single user (aka client) performing requests against
    a particular Backend."""
    def __init__(self, environment):
        super().__init__(environment)
        self.count = 0
        self.database = Database(environment.options.database).build()
        self.workload = Workload(environment.options.workload).build()
        self.phase = VectorSearchUser.Phase.Load

    @task
    def request(self):
        match self.phase:
            case VectorSearchUser.Phase.Load:
                self.do_load()
            case VectorSearchUser.Phase.Run:
                self.do_run()

    def do_load(self):
        if batch := self.workload.next_record_batch():
            print(f"Batch: {batch}")
            print(f"Loading batch of size:", len(batch))
            self.database.upsert(batch)
        else:
            # No more data to load, advance to Run phase.
            self.phase = VectorSearchUser.Phase.Run

    def do_run(self):
        if self.workload.execute_next_request(self.workload):
            print(f"Issue request {self.count}: request")
        else:
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
    parser = argparse.ArgumentParser(
        prog='VCB',
        description='Vector Search Bench')
    parser.add_argument("--database", required=True,
                        choices=tuple(e.value for e in Database))
    parser.add_argument("--workload", required=True,
                        choices=tuple(e.value for e in Workload))

    options = parser.parse_args()

    # setup Environment and Runner
    env = Environment(user_classes=[VectorSearchUser], events=events)
    env.options = options

    runner = env.create_local_runner()

    # start a WebUI instance
    #web_ui = env.create_web_ui("127.0.0.1", 8089)

    # execute init event handlers (only really needed if you have registered any)
    env.events.init.fire(environment=env, runner=runner)# , web_ui=web_ui)

    # start a greenlet that periodically outputs the current stats
    gevent.spawn(stats_printer(env.stats))

    # start a greenlet that save current stats to history
    gevent.spawn(stats_history, env.runner)

    # start the test
    runner.start(1, spawn_rate=10)

    # in 30 seconds stop the runner
    gevent.spawn_later(30, runner.quit)

    # wait for the greenlets
    runner.greenlet.join()

    # stop the web server for good measures
#    web_ui.stop()


if __name__ == "__main__":
    main()
