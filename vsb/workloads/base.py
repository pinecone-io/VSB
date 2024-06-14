from abc import ABC, abstractmethod
from collections.abc import Iterator

from vsb.vsb_types import SearchRequest, DistanceMetric, RecordList, Record


class VectorWorkload(ABC):
    @abstractmethod
    def __init__(self, name: str, **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

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

    @property
    @abstractmethod
    def record_count(self) -> int:
        """
        The number of records in the initial workload after population, but
        before issuing any additional requests.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def request_count(self) -> int:
        """
        The number of requests in the Run phase of the test.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sample_record(self) -> Record:
        """
        Return a sample record from the workload, to aid in databases sizing
        the batch size to use, etc.
        """
        raise NotImplementedError

    @abstractmethod
    def get_record_batch_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, RecordList]]:
        """
        For initial record ingest, returns a RecordBatchIterator over the
        records for the specified `user_id`, assuming there is a total of
        `num_users` which will be ingesting data - i.e. for the entire workload
        to be loaded there should be `num_users` calls to this method.
        Returns an Iterator which yields a tuple of
        (namespace, batch of records), or (None, None) if there are no more
        records to load.
        :param num_users: The number of clients the dataset ingest is
            distributed across.
        :param user_id: The ID of the user requesting the iterator.
        :param batch_size: The size of the batches to create.
        """
        raise NotImplementedError

    @abstractmethod
    def next_request(self) -> (str, SearchRequest | None):
        """Obtain the next request for this workload. Returns a tuple of
        (tenant, Request), where Request is None if there are no more Requests
        to issue.
        """
        raise NotImplementedError
