from pinecone.grpc import PineconeGRPC, GRPCIndex

from ..base import DB, Index
from ...vsb_types import Vector, Record, SearchRequest


class PineconeIndex(Index):
    def __init__(self, index: GRPCIndex, tenent: str):
        # TODO: Support multiple indexes (namesspaces)
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
    def __init__(self, config: dict):
        self.pc = PineconeGRPC(config["api_key"])
        self.index = self.pc.Index(name=config["index_name"])

    def get_index(self, tenant: str) -> Index:
        return PineconeIndex(self.index, tenant)
