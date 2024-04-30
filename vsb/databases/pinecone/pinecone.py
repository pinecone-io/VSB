from pinecone.grpc import PineconeGRPC, GRPCIndex

from ..base import DB, Namespace
from ...vsb_types import Vector, Record, SearchRequest


class PineconeNamespace(Namespace):
    def __init__(self, index: GRPCIndex, namespace: str):
        # TODO: Support multiple namespaces
        self.index = index

    def upsert(self, ident, vector, metadata):
        raise NotImplementedError

    def upsert_batch(self, batch: list[Record]):
        self.index.upsert(batch)

    def search(self, request: SearchRequest) -> list[str]:
        result = self.index.query(vector=request.values, top_k=request.top_k)
        matches = [m["id"] for m in result["matches"]]
        return matches


class PineconeDB(DB):
    def __init__(self, config):
        self.pc = PineconeGRPC(config.pinecone_api_key)
        self.index = self.pc.Index(name=config.pinecone_index_name)

    def get_namespace(self, namespace: str) -> Namespace:
        return PineconeNamespace(self.index, namespace)
