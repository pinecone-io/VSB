"""
Metrics gathering and reporting.

When running single-process, this module gathers metrics for all requests in the
`calculated_metrics` global variable, which is reported at the end of the run.

When running multiprocess, each process accumulates the metrics for its
requests in `calculated_metrics`, and uses the locust worker_report and
report_to_master events o periodically communicate the metrics back to the master
process.
"""

import json
import time
from collections import defaultdict

import locust.env
from hdrh.histogram import HdrHistogram
from locust import events
from locust.runners import WorkerRunner
from locust.stats import (
    RequestStats,
    get_readable_percentiles,
)

import vsb
from vsb import logger
import rich.table
import rich.console
import rich.box

# Calculated custom metrics for each performed operation.
# Nested dict, where top-level is the dataset name (for distinguishing multiple-
# dataset workloads), then the request_type (Populate, Search, ...), under
# that there is an instance of HdrHistogram or a plain int for each metric name (
# recall, ...)
calculated_metrics: dict[str, dict[str, HdrHistogram | int]] = {}

# Start and end times for each phase of the workload.
phases: dict[str, str] = {}

# Scale factor for the histograms - we multiply all values by this factor before
# storing them in the histogram. This is because metrics are typically floats
# in domain [0.0, 1.0], yet HdrHistogram only deals with integer values so
# we need to map to the supported range.
HDR_SCALE_FACTOR = 1000

# Percentiles to report for each metric, expressed in the range [0.0, 1.0].
REPORT_PERCENTILES = [
    0.001,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999,
    0.9999,
]


def get_histogram(request_type: str, metric: str) -> HdrHistogram:
    """
    Get the histogram for the given metric for the given request type,
    creating an empty histogram is the request type and/or metric is not already
    recorded.
    """
    req_type_metrics = calculated_metrics.setdefault(request_type, dict())
    return req_type_metrics.setdefault(metric, HdrHistogram(1, 100_000, 3))


def update_counter(request_type: str, metric: str, value: int) -> None:
    """
    Update the counter for the given metric for the given request type,
    creating a counter from zero histogram is the request type and/or metric is
    not already recorded.
    """
    req_type_metrics = calculated_metrics.setdefault(request_type, defaultdict(int))
    req_type_metrics[metric] += value


def get_metric_percentile(request_type: str, metric: str, percentile: float) -> float:
    """
    Get the value of the given percentile for the given metric for the given
    request type, or None if the metric is not recorded.
    """
    if req_type_metrics := calculated_metrics.get(request_type, None):
        if isinstance(value := req_type_metrics.get(metric, None), HdrHistogram):
            return value.get_value_at_percentile(percentile) / HDR_SCALE_FACTOR
    return None


def get_stats_json(stats: RequestStats) -> str:
    """
    Serialise locust's standard stats, then merge in our custom metrics.
    """
    serialized = stats.serialize_stats()
    for s in serialized:
        if custom := calculated_metrics.get(s["method"], None):
            for metric, value in custom.items():
                if isinstance(value, HdrHistogram):
                    info = {
                        "min": value.get_min_value() / HDR_SCALE_FACTOR,
                        "max": value.get_max_value() / HDR_SCALE_FACTOR,
                        "mean": value.get_mean_value() / HDR_SCALE_FACTOR,
                        "percentiles": {},
                    }
                    for p in REPORT_PERCENTILES:
                        p_key = f"{p * 100:g}"
                        info["percentiles"][p_key] = (
                            value.get_value_at_percentile(p * 100) / HDR_SCALE_FACTOR
                        )
                else:
                    info = value
                s[metric] = info
    return json.dumps(serialized, indent=4)


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """
    Event handler that get triggered on every request and records the
    requests' custom metrics here (from "metrics" kwarg).
    """
    if req_metrics := kwargs.get("metrics", None):
        for k, v in req_metrics.items():
            get_histogram(request_type, k).record_value(v * HDR_SCALE_FACTOR)
    if req_counters := kwargs.get("counters", None):
        for k, v in req_counters.items():
            update_counter(request_type, k, v)


@events.report_to_master.add_listener
def on_report_to_master(client_id, data: dict()):
    """
    This event is triggered on the worker instances every time a stats report is
    to be sent to the locust master. It will allow us to add the extra metrics
    from the local worker to the dict that is being sent, and then we clear
    the local metrics in the worker.
    """
    serialized = {}
    for req_type, metrics in calculated_metrics.items():
        # Serialise each HdrHistogram to base64 string, then add to data to be sent to
        # the master instance.
        serialized[req_type] = {}
        hist: HdrHistogram
        for name, value in metrics.items():
            if isinstance(value, HdrHistogram):
                serialized[req_type][name] = value.encode()
            else:
                serialized[req_type][name] = value
    data["metrics"] = serialized
    calculated_metrics.clear()


@events.worker_report.add_listener
def on_worker_report(client_id, data: dict()):
    """
    This event is triggered on the master instance when a new stats report arrives
    from a worker. Here we add our metrics to the master's aggregated
    stats dict.
    """
    for req_type, metrics in data["metrics"].items():
        for metric_name, value in metrics.items():
            if isinstance(value, bytes):
                # Decode the base64 string back to an HdrHistogram, then add to
                # the master's stats.
                get_histogram(req_type, metric_name).decode_and_add(value)
            else:
                update_counter(req_type, metric_name, value)


def format_error_count(value: int) -> str:
    style_if_above = ("red",)
    default_style = "blue"
    style = style_if_above if value > 0 else default_style
    return f"[{style}]{value}[/]"


def get_stats_summary(stats: RequestStats, current=True) -> str:
    """
    stats summary will be returned as a string containing a formatted table
    """
    table = rich.table.Table(
        title="Operation Summary",
        box=rich.box.SIMPLE_HEAD,
        collapse_padding=True,
    )

    table.add_column("Operation", justify="left", style="cyan", no_wrap=True)
    table.add_column("Requests", justify="right", style="blue")
    table.add_column("Failures", justify="right", style="blue")
    table.add_column("Requests/sec", justify="right", style="blue")
    table.add_column("Failures/sec", justify="right", style="blue")

    for key in sorted(stats.entries.keys()):
        r = stats.entries[key]
        table.add_row(
            r.method,
            str(r.num_requests),
            format_error_count(r.num_failures) + f"({r.fail_ratio * 100:.0f}%)",
            f"{r.current_rps:.0f}" if current else f"{r.total_rps:.0f}",
            (
                format_error_count(r.current_fail_per_sec)
                if current
                else format_error_count(r.total_fail_per_sec)
            ),
        )

    return table


def get_metrics_stats_summary(stats: RequestStats) -> rich.table.Table:
    """
    Format the latency and any custom metrics into a table and return it.
    """
    table = rich.table.Table(
        title="Metrics Summary",
        box=rich.box.SIMPLE_HEAD,
        collapse_padding=True,
    )

    # Define columns
    table.add_column("Operation", justify="left", style="cyan", no_wrap=True)
    table.add_column("Metric", justify="left", style="blue", no_wrap=True)
    table.add_column("Min", justify="right", style="magenta", min_width=4)
    for percentile in get_readable_percentiles(REPORT_PERCENTILES):
        table.add_column(percentile, justify="right", style="yellow", min_width=4)
    table.add_column("Max", justify="right", style="magenta", min_width=4)
    table.add_column("Mean", justify="right", style="magenta", min_width=4)

    # Populate the table with stats entries, sorted by Operation
    for index, key in enumerate(sorted(stats.entries.keys())):
        # First add the locust-tracked response times (latency)
        r = stats.entries[key]
        if r.response_times:
            row = [
                r.method,
                "Latency (ms)",
                f"{(r.min_response_time or 0):.0f}",
                *[
                    f"{r.get_response_time_percentile(p):.0f}"
                    for p in REPORT_PERCENTILES
                ],
                f"{r.max_response_time:.0f}",
                f"{r.avg_response_time:.0f}",
            ]
            table.add_row(*row)
        request = key[1]
        # Also include any custom metrics for this request type.
        if custom := calculated_metrics.get(r.name, dict()).get(request, None):
            for metric, value in custom.items():
                if isinstance(value, HdrHistogram):
                    row = [
                        request,
                        metric.capitalize(),
                        *[
                            f"{value.get_value_at_percentile(p * 100) / HDR_SCALE_FACTOR:.2f}"
                            for p in REPORT_PERCENTILES
                        ],
                        f"{value.get_min_value() / HDR_SCALE_FACTOR:.2f}",
                        f"{value.get_max_value() / HDR_SCALE_FACTOR:.2f}",
                        f"{value.get_mean_value() / HDR_SCALE_FACTOR:.2f}",
                    ]
                    table.add_row(*row)
        # Separate each request type with a blank row
        if index < len(stats.entries) - 1:
            table.add_row()

    return table


@events.quitting.add_listener
def print_metrics_on_quitting(environment: locust.env.Environment):
    # Emit stats once on the master (if running in distributed mode) or
    # once on the LocalRunner (if running in single-process mode).
    if (
        not isinstance(environment.runner, WorkerRunner)
        and environment.shape_class.finished
    ):
        vsb.console.print("")
        vsb.console.print(get_stats_summary(environment.stats, False))
        vsb.console.print(get_metrics_stats_summary(environment.stats))

        stats_file = vsb.log_dir / "stats.json"
        stats_file.write_text(get_stats_json(environment.stats))
        logger.info(f"Saved stats to '{stats_file}'")


def record_phase_start(phase: str):
    """Record the start of a new phase in the workload."""
    phases.setdefault(phase, {})["start"] = time.time()
    logger.info(f"Starting {phase} phase")


def record_phase_end(phase: str):
    """Record the end of a new phase in the workload."""
    phases.setdefault(phase, {})["end"] = time.time()
    if "start" not in phases[phase]:
        logger.warning(f"Ending phase {phase} without starting it")
    else:
        logger.info(
            f"Completed {phase} phase, took "
            f"{phases[phase]['end'] - phases[phase]['start']:.2f}s"
        )
