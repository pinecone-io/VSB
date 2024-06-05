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
from abc import abstractmethod, ABC

from locust import events
import locust.stats
from vsb.vsb_types import SearchRequest

# Calculated custom metrics for each performed operation.
# Nested dict, where top-level is the request_type (Populate, Search, ...), under
# that there is a dict keyed on metric name (recall, ...)
calculated_metrics: dict[str, dict[str, list[float]]] = {}


class Metric(ABC):
    """Generic abstraction for a metric against a Vector Search request"""

    @staticmethod
    @abstractmethod
    def name() -> str:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def measure(request: SearchRequest, results: list[str]) -> float:
        raise NotImplementedError


class Recall(Metric):
    """Measure the recall of the given request / result."""

    @staticmethod
    def name():
        return "recall"

    @staticmethod
    def measure(request: SearchRequest, results: list[str]) -> float:
        if not request.neighbors:
            return None
        expected: set[str] = set(request.neighbors)
        actual: set[str] = set(results)
        matches = len(expected & actual)
        return matches / len(expected)


METRICS = (Recall,)
"""Metrics to be calculated for each request."""


def calculate_metrics(request: SearchRequest, results: list[str]) -> dict[str, float]:
    metrics = dict()
    for metric in METRICS:
        value = metric.measure(request, results)
        metrics[metric.name()] = value
    return metrics


def print_stats_json(stats: locust.stats.RequestStats) -> None:
    """
    Serialise locust's standard stats, then merge in our custom metrics.
    # Note we replace locust.stats.print_stats_json with this function via
    # monkey-patching - see vsb.py.
    """
    serialized = stats.serialize_stats()
    for s in serialized:
        if custom := calculated_metrics.get(s["method"], None):
            s.update(custom)
    print(json.dumps(serialized, indent=4))


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """
    Event handler that get triggered on every request and records the
    requests' custom metrics here (from "metrics" kwarg).
    """
    if req_metrics := kwargs.get("metrics", None):
        req_type_metrics = calculated_metrics.setdefault(request_type, dict())
        for k, v in req_metrics.items():
            req_type_metrics.setdefault(k, []).append(v)


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
    data["metrics"] = calculated_metrics.copy()
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
        req_type_metrics = calculated_metrics.setdefault(req_type, dict())
        for k, v in metrics.items():
            req_type_metrics.setdefault(k, []).extend(v)
