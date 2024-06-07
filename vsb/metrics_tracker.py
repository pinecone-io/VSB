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
import logging

from hdrh.histogram import HdrHistogram
from locust import events
from locust.stats import RequestStats

# Calculated custom metrics for each performed operation.
# Nested dict, where top-level is the request_type (Populate, Search, ...), under
# that there is an instance of HdrHistogram for each metric name (
# recall, ...)
calculated_metrics: dict[str, dict[str, HdrHistogram]] = {}

# Scale factor for the histograms - we multiply all values by this factor before
# storing them in the histogram. This is because metrics are typically floats
# in domain [0.0, 1.0], yet HdrHistogram only deals with integer values so
# we need to map to the supported range.
HDR_SCALE_FACTOR = 1000


def get_histogram(request_type: str, metric: str) -> HdrHistogram:
    """
    Get the histogram for the given metric for the given request type,
    creating an empty histogram is the request type and/or metric is not already
    recorded.
    """
    req_type_metrics = calculated_metrics.setdefault(request_type, dict())
    return req_type_metrics.setdefault(metric, HdrHistogram(1, 100_000, 3))


def print_stats_json(stats: RequestStats) -> None:
    """
    Serialise locust's standard stats, then merge in our custom metrics.
    # Note we replace locust.stats.print_stats_json with this function via
    # monkey-patching - see vsb/main.py.
    """
    serialized = stats.serialize_stats()
    for s in serialized:
        if custom := calculated_metrics.get(s["method"], None):
            for metric, hist in custom.items():
                info = {
                    "min": hist.get_min_value() / HDR_SCALE_FACTOR,
                    "max": hist.get_max_value() / HDR_SCALE_FACTOR,
                    "mean": hist.get_mean_value() / HDR_SCALE_FACTOR,
                    "percentiles": {},
                }
                for p in [1, 5, 25, 50, 90, 99, 99.9, 99.99]:
                    info["percentiles"][p] = (
                        hist.get_value_at_percentile(p) / HDR_SCALE_FACTOR
                    )
                s[metric] = info
    print(json.dumps(serialized, indent=4))


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """
    Event handler that get triggered on every request and records the
    requests' custom metrics here (from "metrics" kwarg).
    """
    if req_metrics := kwargs.get("metrics", None):
        for k, v in req_metrics.items():
            get_histogram(request_type, k).record_value(v * HDR_SCALE_FACTOR)


@events.report_to_master.add_listener
def on_report_to_master(client_id, data: dict()):
    """
    This event is triggered on the worker instances every time a stats report is
    to be sent to the locust master. It will allow us to add the extra metrics
    from the local worker to the dict that is being sent, and then we clear
    the local metrics in the worker.
    """
    logging.debug(
        f"metrics.on_report_to_master(): calculated_metrics:{calculated_metrics}"
    )
    serialized = {}
    for req_type, metrics in calculated_metrics.items():
        # Serialise each HdrHistogram to base64 string, then add to data to be sent to
        # the master instance.
        serialized[req_type] = {}
        hist: HdrHistogram
        for metric, hist in metrics.items():
            serialized[req_type][metric] = hist.encode()
    data["metrics"] = serialized
    calculated_metrics.clear()


@events.worker_report.add_listener
def on_worker_report(client_id, data: dict()):
    """
    This event is triggered on the master instance when a new stats report arrives
    from a worker. Here we add our metrics to the master's aggregated
    stats dict.
    """
    logging.debug(f"metrics.on_worker_report(): data:{data}")
    for req_type, metrics in data["metrics"].items():
        # Decode the base64 string back to an HdrHistogram, then add to the master's
        # stats.
        for metric_name, base64_histo in metrics.items():
            get_histogram(req_type, metric_name).decode_and_add(base64_histo)
