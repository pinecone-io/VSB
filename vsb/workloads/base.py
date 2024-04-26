from abc import ABC, abstractmethod


class VectorWorkload(ABC):
    @abstractmethod
    def next_record_batch(self):
        """
        For initial dataset ingest, returns the next
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def execute_next_request(self, db: 'DB') -> bool:
        """Obtain the next request for this workload and execute against the given
        database. Returns true if execution should continue after this request,
        or false if the workload is complete.
        """
        raise NotImplementedError
