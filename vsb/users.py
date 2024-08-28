import time
import traceback
from enum import Enum, auto

import rich.progress
from locust import User, task, LoadTestShape, constant_throughput, runners
from locust.exception import ResponseError, StopUser
import locust.stats

import gevent

import vsb
import vsb.logging
from vsb import metrics, metrics_tracker
from vsb.databases import DB
import vsb.metrics_tracker
from vsb.vsb_types import (
    RecordList,
    SearchRequest,
    InsertRequest,
    UpdateRequest,
    DeleteRequest,
    FetchRequest,
    QueryRequest,
)
from vsb.workloads import VectorWorkload
from vsb import logger
import vsb.workloads
import vsb.workloads.synthetic_workload
import vsb.workloads.synthetic_workload.synthetic_workload

# Dict of Distributors - objects which distribute test data across all
# VSB Users, potentially across multiple processes.
distributors = {}

# Dict of Subscribers - objects which subscribe workers to shared
# data controlled by the master, across multiple processes.
subscribers = {}


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
                    },
                )
                self.state = self.State.Done
            case self.State.Done:
                # Nothing more to do, but sleep briefly here to prevent
                # us busy-looping in this state.
                gevent.sleep(0.1)


class PopulateUser(User):
    """
    Represents a single user (aka client) populating records from a workload
    into a particular Vector Search database.
    """

    class State(Enum):
        Active = auto()
        Done = auto()

    def __init__(self, environment):
        super().__init__(environment)
        iteration = subscribers["iteration"]()
        # Assign a globally unique (potentially across multiple locust processes)
        # user_id, to use for selecting which subset of the workload this User
        # will operate on.
        self.user_id = next(distributors[f"user_id.populate.{iteration}"])
        self.users_total = environment.parsed_options.num_users or 1
        self.database: DB = environment.database
        self.workload: VectorWorkload = environment.workload_sequence[iteration]
        self.state = PopulateUser.State.Active
        self.load_iter = None
        logger.debug(
            f"PopulateUser.__init__() id:{self.user_id} workload:{self.workload.name} iteration: {iteration} thread: {gevent.getcurrent()}"
        )

    @task
    def request(self):
        match self.state:
            case PopulateUser.State.Active:
                self.do_load()
            case PopulateUser.State.Done:
                # Nothing more to do, but sleep briefly here to prevent
                # us busy-looping in this state.
                gevent.sleep(0.1)

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
                index.insert_batch(vectors)
                stop = time.perf_counter()

                elapsed_ms = (stop - start) * 1000.0
                req_type = (
                    f"{self.workload.get_stats_prefix()}.Populate"
                    if self.environment.workload_sequence.workload_count() > 1
                    else "Populate"
                )
                self.environment.events.request.fire(
                    request_type=req_type,
                    name=self.workload.name,
                    response_time=elapsed_ms,
                    response_length=0,
                    counters={"records": len(vectors)},
                )
            except StopIteration:
                logger.debug(f"User id:{self.user_id} completed Populate phase")
                self.environment.runner.send_message(
                    "update_progress", {"user": self.user_id, "phase": "populate"}
                )
                self.state = PopulateUser.State.Done
        except Exception as e:
            traceback.print_exception(e)
            self.environment.runner.quit()
            raise StopUser


class FinalizeUser(User):
    """
    Represents a single user (aka client) finalizing the population phase of a workload
    into a particular Vector Search database.
    """

    class State(Enum):
        Active = auto()
        Done = auto()

    def __init__(self, environment):
        super().__init__(environment)
        iteration = subscribers["iteration"]()
        logger.debug(f"FinalizeUser.__init__() iteration:{iteration}")
        # Assign a globally unique (potentially across multiple locust processes)
        # user_id, to use for selecting which subset of the workload this User
        # will operate on.
        self.user_id = 0
        self.database = environment.database
        self.record_count: int = environment.workload_sequence.record_count_upto(
            iteration
        )
        self.state = FinalizeUser.State.Active

    @task
    def request(self):
        match self.state:
            case FinalizeUser.State.Active:
                self.do_finalize()
            case FinalizeUser.State.Done:
                # Nothing more to do, but sleep briefly here to prevent
                # us busy-looping in this state.
                gevent.sleep(0.1)

    def do_finalize(self):
        """Perform any database-specific finalization of the populate phase
        (e.g. wait for index building to be complete) before PopulateUser
        declares complete.
        """
        logger.debug("FinalizeUser finalizing population...")
        self.database.finalize_population(self.record_count)
        self.environment.runner.send_message(
            "update_progress", {"user": self.user_id, "phase": "finalize"}
        )
        self.state = FinalizeUser.State.Done


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
        iteration = subscribers["iteration"]()
        # Assign a globally unique (potentially across multiple locust processes)
        # user_id, to use for selecting which subset of the workload this User
        # will operate on.
        self.user_id = next(distributors[f"user_id.run.{iteration}"])
        self.users_total = environment.parsed_options.num_users
        self.database = environment.database
        self.workload = environment.workload_sequence[iteration]
        self.state = RunUser.State.Active
        opts = environment.parsed_options
        self.target_throughput = opts.requests_per_sec / float(opts.num_users)
        logger.debug(
            f"Initialising RunUser id:{self.user_id}, target request/sec:{self.target_throughput}"
        )
        self.query_iter = None

    @task
    def request(self):
        match self.state:
            case RunUser.State.Active:
                self.do_run()
            case RunUser.State.Done:
                # Nothing more to do, but sleep briefly here to prevent
                # us busy-looping in this state.
                gevent.sleep(0.1)

    def wait_time(self):
        """Method called by locust to control how long this task should wait between
        executions.
        """
        if self.target_throughput > 0:
            return constant_throughput(self.target_throughput)(self)
        return 0

    def do_run(self):
        if not self.query_iter:
            batch_size = self.database.get_batch_size(self.workload.get_sample_record())
            self.query_iter = self.workload.get_query_iter(
                self.users_total, self.user_id, batch_size
            )

        tenant: str = None
        request: QueryRequest = None
        try:
            (tenant, request) = next(self.query_iter)
        except StopIteration:
            # No more requests - user is done.
            logger.debug(f"User id:{self.user_id} completed Run phase")
            self.environment.runner.send_message(
                "update_progress", {"user": self.user_id, "phase": "run"}
            )
            self.state = RunUser.State.Done
            return
        try:
            index = self.database.get_namespace(tenant)
            start = time.perf_counter()
            match request:
                case SearchRequest():
                    results = index.search(request)
                case InsertRequest():
                    results = index.insert_batch(request.records)
                case UpdateRequest():
                    results = index.update_batch(request.records)
                case FetchRequest():
                    results = index.fetch_batch(request.ids)
                case DeleteRequest():
                    results = index.delete_batch(request.ids)
            stop = time.perf_counter()
            elapsed_ms = (stop - start) * 1000.0
            match request:
                case SearchRequest():
                    if self.workload.recall_available():
                        calc_metrics = metrics.calculate_metrics(request, results)
                    else:
                        # TODO: change when recall calculation is implemented
                        # We can't calculate recall for synthetic proportional workloads right now,
                        # so don't collect data to pollute the output table.
                        calc_metrics = {}
                    type_label = "Search"
                    reqs = None
                case InsertRequest():
                    calc_metrics = {}
                    type_label = "Insert"
                    reqs = len(request.records)
                case UpdateRequest():
                    calc_metrics = {}
                    type_label = "Update"
                    reqs = len(request.records)
                case FetchRequest():
                    calc_metrics = {}
                    type_label = "Fetch"
                    reqs = len(request.ids)
                case DeleteRequest():
                    calc_metrics = {}
                    type_label = "Delete"
                    reqs = len(request.ids)
                case _:
                    raise ValueError(f"Unknown request type:{request}")

            req_type = (
                f"{self.workload.get_stats_prefix()}.{type_label}"
                if self.environment.workload_sequence.workload_count() > 1
                else type_label
            )
            self.environment.events.request.fire(
                request_type=req_type,
                name=self.workload.name,
                response_time=elapsed_ms,
                response_length=0,
                metrics=calc_metrics,
                counters={"requests": reqs} if reqs else {},
            )
        except Exception as e:
            traceback.print_exception(e)
            self.environment.runner.quit()
            raise StopUser


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
        WaitingForWorkers = auto()
        """Wait for all workers to complete process-wide setup before starting the benchmark"""
        Setup = auto()
        """Setup database, performing any necessary tasks before records are loaded
         (e.g. create tables / indexes, configure server)."""
        TransitionFromSetup = auto()
        """Wait for all Setup users to complete before advancing to either Populate
        or Run phase (depending on if --skip_populate was specified)."""
        Populate = auto()
        """Upsert records and build indexes (either during data load or when all
         records have been upserted)."""
        TransitionToFinalize = auto()
        """Wait for all Populate Users to complete before advancing to Finalize phase"""
        Finalize = auto()
        """Perform any finalization tasks (building index, etc.) after all records 
        have been loaded."""
        TransitionToRun = auto()
        """Wait for all Finalize Users to complete before advancing to Run phase"""
        Run = auto()
        """Issue requests (queries) to the database and recording the results."""
        Done = auto()
        """Final phase when all Run Users have completed"""

    def __init__(self):
        super().__init__()
        logger.debug(f"Initialising LoadShape")
        self.phase = LoadShape.Phase.Init
        self.record_count: int = None
        self.request_count: int = None
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
        logger.debug(f"LoadShape.tick() - phase:{self.phase}")
        match self.phase:
            case LoadShape.Phase.Init:
                # self.runner is not initialised until after __init__(), so we must
                # lazily register our message handler and other information from
                # self.runner on the first tick() call.
                self.runner.register_message("update_progress", self.on_update_progress)
                parsed_opts = self.runner.environment.parsed_options
                self.num_users = parsed_opts.num_users
                self.skip_populate = parsed_opts.skip_populate
                self.no_aggregate_stats = parsed_opts.synthetic_no_aggregate_stats
                # manually change phase because _transition_phase depends on environment
                # attributes like workload_sequence that might not be set up yet
                logger.debug(f"switching to WaitingForWorkers phase")
                self.phase = LoadShape.Phase.WaitingForWorkers
                return self.tick()
            case LoadShape.Phase.WaitingForWorkers:
                if (
                    len(self.runner.environment.setup_completed_workers)
                    == self.runner.environment.parsed_options.expect_workers
                ):
                    # All workers have completed process-wide setup, so we can start
                    logger.debug(
                        f"All workers have completed setup - starting benchmark"
                    )
                    self._transition_phase(LoadShape.Phase.Setup)
                    return self.tick()
                return 0, self.num_users, []
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
            case LoadShape.Phase.TransitionToFinalize:
                if self.get_current_user_count() == 0:
                    # stopped all previous Populate Users, can switch to Finalize
                    # phase now
                    if (
                        self.runner.environment.database.skip_refinalize()
                        and self.runner.environment.iteration > 0
                    ):
                        self._transition_phase(LoadShape.Phase.Run)
                    else:
                        self._transition_phase(LoadShape.Phase.Finalize)
                    return self.tick()
                return 0, self.num_users, []
            case LoadShape.Phase.Finalize:
                self._update_progress_bar()
                return 1, 1, [FinalizeUser]
            case LoadShape.Phase.TransitionToRun:
                if self.get_current_user_count() == 0:
                    # stopped all previous Finalize Users, can switch to Run
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
            LoadShape.Phase.Finalize,
            LoadShape.Phase.Run,
        ]
        if vsb.progress is not None:
            self._update_progress_bar(mark_completed=True)
            vsb.progress.update(
                self.progress_task_id, description=f"✔ {self.phase.name} complete"
            )
            vsb.progress.stop()
            vsb.progress = None
        if hasattr(self.runner.environment, "workload_sequence"):
            # We have to use the previous iteration for repeated Populate phases.
            # This is because the current iteration has already been updated, from
            # iter1.Run -> iter2.TransitionFromSetup. This would try to end
            # iter2.Run instead, which is not what we want.
            if (
                self.runner.environment.iteration > 0
                and self.phase == LoadShape.Phase.Run
            ):
                iteration = self.runner.environment.iteration - 1
            else:
                iteration = self.runner.environment.iteration
            workload = self.runner.environment.workload_sequence[iteration]
            # Setup phase is a special case, as it doesn't have a workload
            phase_display_name = (
                f"{workload.name}-{self.phase.name}"
                if self.phase != LoadShape.Phase.Setup
                and self.runner.environment.workload_sequence.workload_count() > 1
                else self.phase.name
            )
        else:
            phase_display_name = self.phase.name
        if self.phase in tracked_phases:
            metrics_tracker.record_phase_end(phase_display_name)
        self.phase = new
        if hasattr(self.runner.environment, "workload_sequence"):
            workload = self.runner.environment.workload_sequence[
                self.runner.environment.iteration
            ]
            phase_display_name = (
                f"{workload.name}-{self.phase.name}"
                if self.runner.environment.workload_sequence.workload_count() > 1
                else self.phase.name
            )
        else:
            phase_display_name = self.phase.name
        if self.phase in tracked_phases:
            metrics_tracker.record_phase_start(phase_display_name)
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
                self.record_count = (
                    self.runner.environment.workload_sequence.record_count_upto(0)
                )
                self.request_count = self.runner.environment.workload_sequence[
                    0
                ].request_count()
                logger.debug(
                    f"VSBLoadShape.update_progress() - SetupUser completed with "
                    f"record_count={self.record_count}, request_count="
                    f"{self.request_count} - "
                    f"moving to TransitionFromSetup phase"
                )
                self._transition_phase(LoadShape.Phase.TransitionFromSetup)
            case LoadShape.Phase.TransitionFromSetup:
                logger.error(
                    f"VSBLoadShape.update_progress() - Unexpected progress update in "
                    f"TransitionFromSetup phase!"
                )
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
                    # Reset completed_users for next Populate -> Run iteration
                    self.completed_users["populate"] = set()
                    self._transition_phase(LoadShape.Phase.TransitionToFinalize)
                else:
                    logger.debug(
                        f"VSBLoadShape.update_progress() - users have now "
                        f"completed: {self.completed_users['populate']}"
                    )
            case LoadShape.Phase.TransitionToFinalize:
                logger.error(
                    f"VSBLoadShape.update_progress() - Unexpected progress update in "
                    f"TransitionToFinalize phase!"
                )
            case LoadShape.Phase.Finalize:
                assert msg.data["phase"] == "finalize"
                logger.debug(
                    f"VSBLoadShape.update_progress() - completed Finalize phase"
                )
                self._transition_phase(LoadShape.Phase.TransitionToRun)
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
                        f"{num_completed} Run users completed Run phase"
                    )
                    if (
                        self.runner.environment.iteration
                        < self.runner.environment.workload_sequence.workload_count() - 1
                    ):
                        logger.debug(
                            f"VSBLoadShape.update_progress() - "
                            f"switching to next workload in sequence"
                        )
                        self.runner.environment.iteration += 1
                        subscribers["iteration"].update(
                            self.runner.environment.iteration
                        )
                        self.record_count = (
                            self.runner.environment.workload_sequence.record_count_upto(
                                self.runner.environment.iteration
                            )
                        )
                        self.request_count = self.runner.environment.workload_sequence[
                            self.runner.environment.iteration
                        ].request_count()
                        # Reset completed_users for next iteration
                        self.completed_users["run"] = set()
                        self._transition_phase(
                            LoadShape.Phase.TransitionFromSetup
                        )  # Use TransitionFromSetup to scale down and switch back to Populate
                    else:
                        # No more workloads in WorkloadSequence.
                        logger.debug(
                            f"VSBLoadShape.update_progress() - "
                            f"no more workloads to run - finishing benchmark"
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

                env = self.runner.environment
                workload = env.workload_sequence[env.iteration]
                req_type = (
                    f"{workload.get_stats_prefix()}.Populate"
                    if env.workload_sequence.workload_count() > 1
                    else "Populate"
                )
                completed = vsb.metrics_tracker.calculated_metrics.get(
                    req_type, {}
                ).get("records", 0)

                stats: locust.stats.StatsEntry = env.stats.get(workload.name, req_type)
                duration = time.time() - stats.start_time
                rps_str = "  Records/sec: [magenta]{:.1f}".format(completed / duration)
                # If --synthetic-no-aggregate-stats is set, then we need special
                # handling to update the progress bar, since each workload in the
                # sequence uses the same name.
                if (
                    isinstance(
                        env.workload_sequence,
                        vsb.workloads.synthetic_workload.synthetic_workload.SyntheticRunbook,
                    )
                    and not self.no_aggregate_stats
                ):
                    # All cumulative records are stored under stats[workload.name],
                    # we don't need to sum up previous workloads.
                    previous_record_count = 0
                    total = env.workload_sequence.record_count()
                else:
                    previous_record_count = (
                        (env.workload_sequence.record_count_upto(env.iteration - 1))
                        if env.iteration > 0
                        else 0
                    )
                    total = self.record_count
                vsb.progress.update(
                    self.progress_task_id,
                    completed=completed
                    + previous_record_count,  # Add records from previous workloads
                    total=total,
                    extra_info=rps_str,
                )
            case LoadShape.Phase.Finalize:
                vsb.progress.update(
                    self.progress_task_id, completed=1 if mark_completed else 0
                )
            case LoadShape.Phase.Run:
                env = self.runner.environment
                workload = env.workload_sequence[env.iteration]
                cumulative_current_rps = 0
                cumulative_num_requests = 0
                for req_name in ["Search", "Insert", "Update", "Fetch", "Delete"]:
                    req_type = (
                        f"{workload.get_stats_prefix()}.{req_name}"
                        if env.workload_sequence.workload_count() > 1
                        else req_name
                    )
                    stats: locust.stats.StatsEntry = env.stats.get(
                        workload.name, req_type
                    )
                    cumulative_current_rps += stats.current_rps
                    completed = (
                        vsb.metrics_tracker.calculated_metrics.get(req_type, {}).get(
                            "requests", 0
                        )
                        if req_name != "Search"
                        else stats.num_requests
                    )
                    cumulative_num_requests += completed

                # Display current (last 10s) values for some significant metrics
                # in the progress_details row.
                ops_str = f"{cumulative_current_rps:.1f} op/s"
                latency_str = ", ".join(
                    [
                        f"p{p}={stats.get_current_response_time_percentile(p/100.0) or '...'}ms"
                        for p in [50, 95]
                    ]
                )

                def get_recall_pct(p):
                    req_type = (
                        f"{workload.get_stats_prefix()}.Search"
                        if env.workload_sequence.workload_count() > 1
                        else "Search"
                    )
                    recall = vsb.metrics_tracker.get_metric_percentile(
                        req_type, "recall", p
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

                # If --synthetic-no-aggregate-stats is set, the cumulative request
                # count is stored in the stats[workload.name] object. We just use
                # the total request count for the entire runbook as the total,
                # although it breaks convention with non-runbook workloads.
                if (
                    isinstance(
                        env.workload_sequence,
                        vsb.workloads.synthetic_workload.synthetic_workload.SyntheticRunbook,
                    )
                    and not self.no_aggregate_stats
                ):
                    total = env.workload_sequence.request_count()
                else:
                    total = self.request_count
                vsb.progress.update(
                    self.progress_task_id,
                    completed=cumulative_num_requests,
                    total=total,
                    extra_info=metrics_str,
                )
