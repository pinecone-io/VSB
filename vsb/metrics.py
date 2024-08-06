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
        return Recall._calculate(results, request.neighbors[: request.top_k])

    @staticmethod
    def _calculate(actual: list[str], expected: list[str]) -> float:
        if not expected:
            if not actual:
                # If we expect [] and receive [], the result is fully correct.
                return 1.0
            # If we expect [] and receive vectors, the result is (fully) incorrect.
            return 0.0
        matches = len(set(expected) & set(actual))
        return matches / len(expected)


class AveragePrecision(Metric):
    """Measure the average precision of the given request / result."""

    @staticmethod
    def name():
        return "Average Precision"

    @staticmethod
    def measure(request: SearchRequest, results: list[str]) -> float:
        return AveragePrecision._calculate(results, request.neighbors[: request.top_k])

    @staticmethod
    def _calculate(actual: list[str], expected: list[str]) -> float:
        if not expected:
            if not actual:
                # If we expect [] and receive [], the result is fully correct.
                return 1.0
            # If we expect [] and receive vectors, the result is (fully) incorrect.
            return 0.0
        relevant = set(expected)
        num_true_positives = 0
        precision = 0
        for i, result in enumerate(actual):
            if result in relevant:
                num_true_positives += 1
            precision += num_true_positives / (i + 1)
        return precision / len(expected)


class ReciprocalRank(Metric):
    """Measure the reciprocal rank of the given request / result."""

    @staticmethod
    def name():
        return "Reciprocal Rank"

    @staticmethod
    def measure(request: SearchRequest, results: list[str]) -> float:
        return ReciprocalRank._calculate(results, request.neighbors[: request.top_k])

    @staticmethod
    def _calculate(actual: list[str], expected: list[str]) -> float:
        if not expected:
            if not actual:
                # If we expect [] and receive [], the result is fully correct.
                return 1.0
            # If we expect [] and receive vectors, the result is (fully) incorrect.
            return 0.0
        relevant = set(expected)
        for i, result in enumerate(actual):
            if result in relevant:
                return 1 / (i + 1)
        return 0.0


METRICS = (Recall, AveragePrecision, ReciprocalRank)
"""Metrics to be calculated for each request."""


def calculate_metrics(request: SearchRequest, results: list[str]) -> dict[str, float]:
    metrics = dict()
    for metric in METRICS:
        value = metric.measure(request, results)
        metrics[metric.name()] = value
    return metrics
