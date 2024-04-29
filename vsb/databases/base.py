from abc import ABC, abstractmethod
from enum import Enum, auto

from vsb.vsb_types import Record, SearchRequest


class Index(ABC):
    """Abstract class with represents an index or one or more vector records.
    Specific implementations should subclass this and implement all abstract
    methods.
    Instance of this (derived) class are typically created via the corresponding
    (concrete) DB create_index() method.
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
    """Abstract class which represents a database which can store vector
    records in one or more Indexes. Specific Vector DB implementations should
    subclass this and implement all abstract methods.
    """

    @abstractmethod
    def get_index(self, tenant: str) -> Index:
        raise NotImplementedError
