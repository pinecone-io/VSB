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

    @staticmethod
    @abstractmethod
    def dimensions() -> int:
        """
        The dimensions of (dense) vectors for this workload.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def metric() -> DistanceMetric:
        """
        The distance metric of this workload.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def record_count() -> int:
        """
        The number of records in the initial workload after population, but
        before issuing any additional requests.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def request_count() -> int:
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
    def get_query_iter(
        self, num_users: int, user_id: int
    ) -> Iterator[tuple[str, SearchRequest]]:
        """
        Returns an iterator over the sequence of queries for the given user_id,
        assuming a total of `num_users` which will be issuing queries - i.e.
        for the entire query set to be requested there should be `num_users` calls
        to this method.
        Returns an Iterator which yields a tuple of (tenant, Request).
        :param num_users: The number of clients the queries are distributed across.
        :param user_id: The ID of the user requesting the iterator.
        """
        raise NotImplementedError


class VectorWorkloadSequence(ABC):
    @abstractmethod
    def __init__(self, name: str, **kwargs):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    @abstractmethod
    def workload_count() -> int:
        """
        The number of workloads in the sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> VectorWorkload:
        """
        Return the workload at the specified index.
        """
        raise NotImplementedError

    def record_count_upto(self, index: int) -> int:
        """
        Return the total number of records in the sequencedf
        up to and including the specified index's workload.
        """
        if index >= self.workload_count():
            raise IndexError
        return sum(self[index].record_count() for index in range(index + 1))


class SingleVectorWorkloadSequence(VectorWorkloadSequence):
    def __init__(self, name: str, workload: VectorWorkload):
        super().__init__(name)
        self.workload = workload

    @staticmethod
    def workload_count() -> int:
        return 1

    def __getitem__(self, index: int) -> VectorWorkload:
        if index != 0:
            raise IndexError
        return self.workload
