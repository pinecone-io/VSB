from abc import ABC, abstractmethod
from enum import Enum, auto


class Request(Enum):
    Upsert = auto()
    Search = auto()


class Index(ABC):
    """Abstract class with represents an index or one or more vector records.
    Specific implementations should subclass this and implement all abstract
    methods.
    Instance of this (derived) class are typically created via the corresponding
    (concrete) DB create_index() method.
    """

    @abstractmethod
    def upsert(self, ident, vector, metadata):
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector):
        raise NotImplementedError

    def do_request(self, request):
        print(f"Got request: {request}")
        match request.operation:
            case Request.Upsert:
                self.upsert(request.id, request.vector, request.metadata)
                return
            case Request.Search:
                response = self.search(request.q_vector)
                # Record timing, calculate Recall etc.


class DB(ABC):
    """Abstract class which represents a database which can store vector
    records in one or more Indexes. Specific Vector DB implementations should
    subclass this and implement all abstract methods.
    """

    @abstractmethod
    def create_index(self, tenant: str) -> Index:
        raise NotImplementedError
