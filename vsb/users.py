import time
import traceback
from enum import Enum, auto

import rich.progress
from locust import User, task, LoadTestShape
from locust.exception import StopUser
import locust.stats

import vsb
import vsb.logging
from vsb import metrics, metrics_tracker
from vsb.databases import DB
from vsb.vsb_types import RecordList, SearchRequest
from vsb.workloads import VectorWorkload
from vsb import logger

# Dict of Distributors - objects which distribute test data across all
# VSB Users, potentially across multiple processes.
distributors = {}


class SetupUser(User):
    """
    Represents a single user (aka client) setting up the connection / workload
    for a particular Vector Search database.
    """

    class State(Enum):
        Active = auto()
        Done = auto()

    def __init__(self, environment):
        super().__init__(environment)
        logger.debug(f"Initialising SetupUser")
        self.database: DB = environment.database
        self.state = self.State.Active

    @task
    def setup(self):
        """Perform any database-specific initialization of the populate phase"""
        match self.state:
            case self.State.Active:
                self.database.initialize_population()
                self.environment.runner.send_message(
                    "update_progress",
                    {
                        "user": 0,
                        "phase": "setup",
                        "record_count": self.environment.workload.record_count,
                        "request_count": self.environment.workload.request_count,
                    },
                )
                self.state = self.State.Done
            case self.State.Done:
                # Nothing more to do, but sleep briefly here to prevent
                # us busy-looping in this state.
                time.sleep(0.1)


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
        logger.debug(f"Initialising PopulateUser id:{self.user_id}")
        self.users_total = environment.parsed_options.num_users or 1
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
                    counters={"records": len(vectors)},
                )
            except StopIteration:
                logger.debug(f"User id:{self.user_id} completed Populate phase")
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
            logger.debug("PopulateUser finalizing population...")
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
        logger.debug(f"Initialising RunUser id:{self.user_id}")
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
        request: SearchRequest
        (tenant, request) = self.workload.next_request()
        if request:
            try:
                index = self.database.get_namespace(tenant)

                start = time.perf_counter()
                results = index.search(request)
                stop = time.perf_counter()

                calc_metrics = metrics.calculate_metrics(request, results)

                elapsed_ms = (stop - start) * 1000.0
                self.environment.events.request.fire(
                    request_type="Search",
                    name=self.workload.name,
                    response_time=elapsed_ms,
                    response_length=0,
                    metrics=calc_metrics,
                )
            except Exception as e:
                traceback.print_exception(e)
                self.environment.runner.quit()
                raise StopUser
        else:
            # No more requests - user is done.
            logger.debug(f"User id:{self.user_id} completed Run phase")
            self.environment.runner.send_message(
                "update_progress", {"user": self.user_id, "phase": "run"}
            )
            self.state = RunUser.State.Done


class LoadShape(LoadTestShape):
    """
    Custom LoadTestShape which consists of three phases, where different
    User classes are spawned for each of the Setup, Populate and Run phases.

    In each phase there will be the same number of User tasks created (--users=N).
    """

    use_common_options = True

    class Phase(Enum):
        Init = auto()
        """Perform LoadShape initialization """
        Setup = auto()
        """Setup database, performing any necessary tasks before records are loaded
         (e.g. create tables / indexes, configure server)."""
        TransitionFromSetup = auto()
        """Wait for all Setup users to complete before advancing to either Populate
        or Run phase (depending on if --skip_populate was specified)."""
        Populate = auto()
        """Upsert records and build indexes (either during data load or when all
         records have been upserted)."""
        TransitionToRun = auto()
        """Wait for all Populate Users to complete before advancing to Run phase"""
        Run = auto()
        """Issue requests (queries) to the database and recording the results."""
        Done = auto()
        """Final phase when all Run Users have completed"""

    def __init__(self):
        super().__init__()
        self.phase = LoadShape.Phase.Init
        self.record_count: int = None
        self.progress_task_id: rich.progress.TaskID = None
        self.num_users = 0
        self.skip_populate = False
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
            case LoadShape.Phase.Init:
                # self.runner is not initialised until after __init__(), so we must
                # lazily register our message handler and other information from
                # self.runner on the first tick() call.
                self.runner.register_message("update_progress", self.on_update_progress)
                parsed_opts = self.runner.environment.parsed_options
                self.num_users = parsed_opts.num_users
                self.skip_populate = parsed_opts.skip_populate
                self._transition_phase(LoadShape.Phase.Setup)
                return self.tick()
            case LoadShape.Phase.Setup:
                vsb.progress.update(self.progress_task_id, total=1)
                return 1, 1, [SetupUser]
            case LoadShape.Phase.TransitionFromSetup:
                if self.get_current_user_count() == 0:
                    # Finished all previous SetupUser tasks, can switch to next
                    # phase now
                    self._transition_phase(
                        LoadShape.Phase.Run
                        if self.skip_populate
                        else LoadShape.Phase.Populate
                    )
                    return self.tick()
                return 0, self.num_users, []
            case LoadShape.Phase.Populate:
                self._update_progress_bar()
                return self.num_users, self.num_users, [PopulateUser]
            case LoadShape.Phase.TransitionToRun:
                if self.get_current_user_count() == 0:
                    # stopped all previous Populate Users, can switch to Run
                    # phase now
                    self._transition_phase(LoadShape.Phase.Run)
                    return self.tick()
                return 0, self.num_users, []
            case LoadShape.Phase.Run:
                self._update_progress_bar()
                return self.num_users, self.num_users, [RunUser]
            case LoadShape.Phase.Done:
                return None
            case _:
                raise ValueError(f"Invalid phase:{self.phase}")

    @property
    def finished(self):
        return self.phase == LoadShape.Phase.Done

    def _transition_phase(self, new: Phase):
        # Record and log the start of the publicly visible phases.
        tracked_phases = [
            LoadShape.Phase.Setup,
            LoadShape.Phase.Populate,
            LoadShape.Phase.Run,
        ]
        if vsb.progress is not None:
            self._update_progress_bar(mark_completed=True)
            vsb.progress.update(
                self.progress_task_id, description=f"✔ {self.phase.name} complete"
            )
            vsb.progress.stop()
            vsb.progress = None
        if self.phase in tracked_phases:
            metrics_tracker.record_phase_end(self.phase.name)
        self.phase = new
        if self.phase in tracked_phases:
            metrics_tracker.record_phase_start(self.phase.name)
            # Started a new phase - create a progress object for it (which will
            # display progress bars for each task on screen)
            vsb.progress = vsb.logging.make_progressbar()
            self.progress_task_id = vsb.progress.add_task(
                f"Performing {self.phase.name} phase", total=None
            )

    def on_update_progress(self, msg, **kwargs):
        # Fired when VSBLoadShape (running on the master) receives an
        # "update_progress" message.
        logger.debug(
            f"VSBLoadShape.update_progress() - user:{msg.data['user']}, phase:{msg.data['phase']}"
        )
        match self.phase:
            case LoadShape.Phase.Setup:
                assert msg.data["phase"] == "setup"
                self.record_count = msg.data["record_count"]
                self.request_count = msg.data["request_count"]
                logger.debug(
                    f"VSBLoadShape.update_progress() - SetupUser completed with "
                    f"record_count={self.record_count}, request_count="
                    f"{self.request_count} - "
                    f"moving to TransitionFromSetup phase"
                )
                self._transition_phase(LoadShape.Phase.TransitionFromSetup)
            case LoadShape.Phase.Populate:
                assert msg.data["phase"] == "populate"
                self.completed_users["populate"].add(msg.data["user"])
                num_completed = len(self.completed_users["populate"])
                if num_completed == self.runner.environment.parsed_options.num_users:
                    logger.debug(
                        f"VSBLoadShape.update_progress() - all "
                        f"{num_completed} Populate users completed - "
                        f"moving to Run phase"
                    )
                    self._transition_phase(LoadShape.Phase.TransitionToRun)
                else:
                    logger.debug(
                        f"VSBLoadShape.update_progress() - users have now "
                        f"completed: {self.completed_users['populate']}"
                    )
            case LoadShape.Phase.TransitionToRun:
                logger.error(
                    f"VSBLoadShape.update_progress() - Unexpected progress update in "
                    f"TransitionToRun phase!"
                )
            case LoadShape.Phase.Run:
                assert msg.data["phase"] == "run"
                self.completed_users["run"].add(msg.data["user"])
                num_completed = len(self.completed_users["run"])
                if num_completed == self.runner.environment.parsed_options.num_users:
                    logger.debug(
                        f"VSBLoadShape.update_progress() - all "
                        f"{num_completed} Run users completed Run phase - "
                        f"finishing benchmark"
                    )
                    self._transition_phase(LoadShape.Phase.Done)
            case LoadShape.Phase.Done:

                logger.error(
                    f"VSBLoadShape.update_progress() - Unexpected progress update in Done phase!"
                )

    def _update_progress_bar(self, mark_completed: bool = False):
        """Update the phase progress bar for the current phase."""
        match self.phase:
            case LoadShape.Phase.Setup:
                vsb.progress.update(
                    self.progress_task_id, total=1, completed=1 if mark_completed else 0
                )
            case LoadShape.Phase.Populate:
                completed = vsb.metrics_tracker.calculated_metrics.get(
                    "Populate", {}
                ).get("records", 0)

                env = self.runner.environment
                stats: locust.stats.StatsEntry = env.stats.get(
                    env.parsed_options.workload, "Populate"
                )
                duration = time.time() - stats.start_time
                rps_str = "  Records/sec: [magenta]{:.1f}".format(completed / duration)
                vsb.progress.update(
                    self.progress_task_id,
                    completed=completed,
                    total=self.record_count,
                    extra_info=rps_str,
                )
            case LoadShape.Phase.Run:
                # TODO: When we add additional request types other than Search,
                # we need to expand this to include them.
                env = self.runner.environment
                stats: locust.StatsEntry = env.stats.get(
                    env.parsed_options.workload, "Search"
                )

                # Display current (last 10s) values for some significant metrics
                # in the progress_details row.
                ops_str = f"{stats.current_rps:.1f} op/s"
                latency_str = ", ".join(
                    [
                        f"p{p}={stats.get_current_response_time_percentile(p/100.0) or '...'}ms"
                        for p in [50, 95]
                    ]
                )

                def get_recall_pct(p):
                    recall = vsb.metrics_tracker.get_metric_percentile(
                        "Search", "recall", p
                    )
                    return f"{recall:.2f}" if recall else "..."

                recall_str = ", ".join([f"p{p}={get_recall_pct(p)}" for p in [50, 5]])

                last_n = locust.stats.CURRENT_RESPONSE_TIME_PERCENTILE_WINDOW
                metrics_str = (
                    f"  Current metrics (last {last_n}s): [magenta]{ops_str}[/magenta]"
                    + " | "
                    + f"[magenta]latency: {latency_str}[/magenta]"
                    + " | "
                    + f"[magenta]recall: {recall_str}"
                )

                vsb.progress.update(
                    self.progress_task_id,
                    completed=stats.num_requests,
                    total=self.request_count,
                    extra_info=metrics_str,
                )
