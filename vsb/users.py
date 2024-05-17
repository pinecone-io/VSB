import logging
import time
import traceback
from enum import Enum, auto

from locust import User, task, LoadTestShape
from locust.exception import StopUser

from vsb.databases import DB
from vsb.vsb_types import RecordList, SearchRequest
from vsb.workloads import VectorWorkload

# Dict of Distributors - objects which distribute test data across all
# VSB Users, potentially across multiple processes.
distributors = {}


class PopulateUser(User):
    """
    Represents a single user (aka client) populating records from a workload
    into a particular Vector Search database.
    """

    class State(Enum):
        Active = auto()
        Finalize = auto()
        Done = auto()

    def __init__(self, environment):
        super().__init__(environment)
        # Assign a globally unique (potentially across multiple locust processes)
        # user_id, to use for selecting which subset of the workload this User
        # will operate on.
        self.user_id = next(distributors["user_id.populate"])
        logging.debug(f"Initialising PopulateUser id:{self.user_id}")
        self.users_total = environment.parsed_options.num_users
        self.database: DB = environment.database
        self.workload: VectorWorkload = environment.workload
        self.state = PopulateUser.State.Active
        self.load_iter = None

    @task
    def request(self):
        match self.state:
            case PopulateUser.State.Active:
                self.do_load()
            case PopulateUser.State.Finalize:
                self.do_finalize()
            case PopulateUser.State.Done:
                # Nothing more to do, but sleep briefly here to prevent
                # us busy-looping in this state.
                time.sleep(0.1)

    def do_load(self):
        try:
            if not self.load_iter:
                batch_size = self.database.get_batch_size(
                    self.workload.get_sample_record()
                )
                self.load_iter = self.workload.get_record_batch_iter(
                    self.users_total, self.user_id, batch_size
                )
            try:
                vectors: RecordList
                (tenant, vectors) = next(self.load_iter)
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
            except StopIteration:
                logging.debug(f"User id:{self.user_id} completed Populate phase")
                self.state = PopulateUser.State.Finalize
        except Exception as e:
            traceback.print_exception(e)
            self.environment.runner.quit()
            raise StopUser

    def do_finalize(self):
        """Perform any database-specific finalization of the populate phase
        (e.g. wait for index building to be complete) before PopulateUser
        declares complete.
        """
        if self.user_id == 0:
            # First user only performs finalization (don't want
            # to call repeatedly if >1 user).
            self.database.finalize_population(self.workload.record_count)
        self.environment.runner.send_message(
            "update_progress", {"user": self.user_id, "phase": "populate"}
        )
        self.state = PopulateUser.State.Done


class RunUser(User):
    """
    Represents a single user (aka client) performing requests from a workload
    into a particular Vector Search database.
    """

    class State(Enum):
        Active = auto()
        Done = auto()

    def __init__(self, environment):
        super().__init__(environment)
        # Assign a globally unique (potentially across multiple locust processes)
        # user_id, to use for selecting which subset of the workload this User
        # will operate on.
        self.user_id = next(distributors["user_id.run"])
        logging.debug(f"Initialising RunUser id:{self.user_id}")
        self.database = environment.database
        self.workload = environment.workload
        self.state = RunUser.State.Active

    @task
    def request(self):
        match self.state:
            case RunUser.State.Active:
                self.do_run()
            case RunUser.State.Done:
                # Nothing more to do, but sleep briefly here to prevent
                # us busy-looping in this state.
                time.sleep(0.1)

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
            # No more requests - user is done.
            logging.debug(f"User id:{self.user_id} completed Run phase")
            self.environment.runner.send_message(
                "update_progress", {"user": self.user_id, "phase": "run"}
            )
            self.state = RunUser.State.Done


class LoadShape(LoadTestShape):
    """
    Custom LoadTestShape which consists of two phases, where different
    User classes are spawned for each of the Populate and Load phases.
    """

    use_common_options = True

    class Phase(Enum):
        Initialize = auto()
        Populate = auto()
        TransitionToRun = auto()
        Run = auto()
        Done = auto()

    def __init__(self):
        super().__init__()
        self.phase = LoadShape.Phase.Initialize
        self.num_users = 0
        self.completed_users = {"populate": set(), "run": set()}

    def tick(self):
        """Called ~1sec by locust, this method defines what state the Benchmark
        run is in by specifying what tasks should currently be running.
        Note that locust doesn't appear to let you directory switch from running
        N tasks of ClassA to N tasks of ClassB - if you attempt to do that
        then it doesn't actually start any ClassB tasks. As such we need to
        first reduce task count to 0, then ramp back to N tasks of ClassB.
        """
        match self.phase:
            case LoadShape.Phase.Initialize:
                # self.runner is not initialised until after __init__(), so we must
                # lazily register our message handler and user count on the first
                # tick() call.
                self.runner.environment.runner.register_message(
                    "update_progress", self.on_update_progress
                )
                parsed_opts = self.runner.environment.parsed_options
                self.num_users = parsed_opts.num_users
                self.phase = (
                    LoadShape.Phase.Run
                    if parsed_opts.skip_populate
                    else LoadShape.Phase.Populate
                )
                return self.tick()
            case LoadShape.Phase.Populate:
                return self.num_users, self.num_users, [PopulateUser]
            case LoadShape.Phase.TransitionToRun:
                if self.get_current_user_count() == 0:
                    # stopped all previous Populate Users, can switch to Run
                    # phase now
                    self.phase = LoadShape.Phase.Run
                    return self.tick()
                return 0, self.num_users, []
            case LoadShape.Phase.Run:
                return self.num_users, self.num_users, [RunUser]
            case LoadShape.Phase.Done:
                return None
            case _:
                raise ValueError(f"Invalid phase:{self.phase}")

    def on_update_progress(self, msg, **kwargs):
        # Fired when VSBLoadShape (running on the master) receives an
        # "update_progress" message.
        logging.debug(
            f"VSBLoadShape.update_progress() - user:{msg.data['user']}, phase:{msg.data['phase']}"
        )
        match self.phase:
            case LoadShape.Phase.Populate:
                assert msg.data["phase"] == "populate"
                self.completed_users["populate"].add(msg.data["user"])
                num_completed = len(self.completed_users["populate"])
                if num_completed == self.runner.environment.parsed_options.num_users:
                    logging.info(
                        f"VSBLoadShape.update_progress() - all "
                        f"{num_completed} Populate users completed - "
                        f"moving to Run phase"
                    )
                    self.phase = LoadShape.Phase.TransitionToRun
                else:
                    logging.debug(
                        f"VSBLoadShape.update_progress() - users have now "
                        f"completed: {self.completed_users['populate']}"
                    )
            case LoadShape.Phase.TransitionToRun:
                logging.error(
                    f"VSBLoadShape.update_progress() - Unexpected progress update in "
                    f"TransitionToRun phase!"
                )
            case LoadShape.Phase.Run:
                assert msg.data["phase"] == "run"
                self.completed_users["run"].add(msg.data["user"])
                num_completed = len(self.completed_users["run"])
                if num_completed == self.runner.environment.parsed_options.num_users:
                    logging.info(
                        f"VSBLoadShape.update_progress() - all "
                        f"{num_completed} Run users completed Run phase - "
                        f"finishing benchmark"
                    )
                    self.phase = LoadShape.Phase.Done
            case LoadShape.Phase.Done:
                logging.error(
                    f"VSBLoadShape.update_progress() - Unexpected progress update in Done phase!"
                )
