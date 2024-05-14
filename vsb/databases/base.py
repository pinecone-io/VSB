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
    def upsert(self, key, vector, metadata):
        raise NotImplementedError

    @abstractmethod
    def upsert_batch(self, batch: list[Record]):
        raise NotImplementedError

    @abstractmethod
    def search(self, request: SearchRequest) -> list[str]:
        raise NotImplementedError


class DB(ABC):
    """Abstract class which represents a vector database made up of one or more
    Namespaces, where each Namespace contains one or more records.

    Specific Vector DB implementations should subclass this and implement all abstract methods.
    """

    @abstractmethod
    def __init__(self, dimensions: int, metric: DistanceMetric, config: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_namespace(self, namespace_name: str) -> Namespace:
        """
        Returns the Namespace to use for the specified namespace name.
        """
        raise NotImplementedError

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
        raise NotImplementedError
