from abc import ABC, abstractmethod

from vsb.vsb_types import Record, SearchRequest, DistanceMetric


class RecordBatchIterator:
    """
    An iterator over a sequence of Records from a Vector Workload. Yields
    batches of records of a requested count.
    Used for initial data ingest; a RecordBatchIterator supports iterating over
    a subset of Records if multiple users are ingesting data concurrently -
    e.g. for a complete Workload of 1,000 Records and 2 users, each instance of
    RecordBatchIterator (suitably initialised) will yield 500 records each.
    """


class VectorWorkload(ABC):
    @abstractmethod
    def __init__(self, cache_dir: str):
        raise NotImplementedError

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
    def get_record_batch_iter(
        self, num_users: int, user_id: int
    ) -> RecordBatchIterator:
        """
        For initial dataset ingest, returns a RecordBatchIterator over the
        records for the specified `user_id`.
        Returns a tuple of (namespace, batch of records), or (None, None)
        if there are no more records to load.
        :param num_users: The number of clients the dataset ingest is
            distributed across.
        :param user_id: The ID of the user requesting the iterator.
        """
        raise NotImplementedError

    @abstractmethod
    def next_request(self) -> (str, SearchRequest | None):
        """Obtain the next request for this workload. Returns a tuple of
        (tenant, Request), where Request is None if there are no more Requests
        to issue.
        """
        raise NotImplementedError
