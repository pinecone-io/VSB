from abc import ABC, abstractmethod
from enum import Enum, auto

from vsb.vsb_types import Record, SearchRequest, DistanceMetric


class Namespace(ABC):
    """Abstract class with represents a set of one or more vector records
    grouped together by some logical association (e.g. a single tenant / user).
    A Database consists of one or more Namespaces, and each record exists
    in exactly one namespace.
    Specific implementations should subclass this and implement all abstract
    methods.
    Instance of this (derived) class are typically created via the corresponding
    (concrete) DB get_namespace() method.
    """

    @abstractmethod
    def insert_batch(self, batch: list[Record]):
        raise NotImplementedError

    @abstractmethod
    def update_batch(self, batch: list[Record]):
        raise NotImplementedError

    @abstractmethod
    def search(self, request: SearchRequest) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def fetch_batch(self, request: list[str]) -> list[Record]:
        raise NotImplementedError

    @abstractmethod
    def delete_batch(self, request: list[str]):
        raise NotImplementedError


class DB(ABC):
    """Abstract class which represents a vector database made up of one or more
    Namespaces, where each Namespace contains one or more records.

    Specific Vector DB implementations should subclass this and implement all abstract methods.
    """

    @abstractmethod
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: DistanceMetric,
        name: str,
        config: dict,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_batch_size(self, sample_record: Record) -> int:
        """Return the preferred batch size for a workload consisting of
        records similar to the sample record."""
        raise NotImplementedError

    @abstractmethod
    def get_namespace(self, namespace_name: str) -> Namespace:
        """
        Returns the Namespace to use for the specified namespace name.
        """
        raise NotImplementedError

    def initialize_population(self):
        """
        Performs any initialization of the database at the beginning of the
        Populate phase - i.e. before Namespace.upsert_batch() is called.
        Note this is only called once per test run, irrespective of how many
        users have been configured (not per user).
        This could include creating any initial datastructures on the database,
        or one-time configuration. For implementations which don't need
        any initialization before data is populated, this method left as empty.
        """
        pass

    def finalize_population(self, record_count: int):
        """
        Performs any finalization of the database at the end of the Populate
        phase. Call should block until the database is ready to perform the
        next phase.
        For databases which index records asynchronously, this should wait for all
        records to be indexed.
        For databases which perform indexing as a separate step to data ingest,
        this should create the index(es).
        """
        pass

    def skip_refinalize(self):
        """
        Returns True if the database does not need to be re-finalized after
        population.

        For databases with separate index building steps that would run during
        finalize, we may want to skip trying to build an existing index.
        """
        pass

    def get_record_count(self) -> int:
        """
        Returns the total number of records in the database.
        """
        pass
