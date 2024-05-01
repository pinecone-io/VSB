from abc import ABC, abstractmethod

from vsb.vsb_types import Record, SearchRequest, DistanceMetric


class VectorWorkload(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of this workload.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        The dimensions of (dense) vectors for this workload.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def metric(self) -> DistanceMetric:
        """
        The distance metric of this workload.
        """
        raise NotImplementedError

    @abstractmethod
    def next_record_batch(self) -> (str, list[Record]):
        """
        For initial dataset ingest, loads the next batch of records into
        the given database.
        Returns true if loading should continue (i.e. more data to load), or
        false if loading is complete.
        """
        raise NotImplementedError

    @abstractmethod
    def next_request(self) -> (str, SearchRequest | None):
        """Obtain the next request for this workload. Returns a tuple of
        (tenant, Request), where Request is None if there are no more Requests
        to issue.
        """
        raise NotImplementedError
