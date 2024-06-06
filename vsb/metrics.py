from abc import ABC, abstractmethod

from vsb.vsb_types import SearchRequest


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
