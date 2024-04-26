from ..base import DB, Index


class PineconeIndex(Index):
    def __init__(self, tenent: str):
        pass

    def upsert(self, ident, vector, metadata):
        pass

    def search(self, query_vector):
        pass


class PineconeDB(DB):
    def __init__(self):
        print("PineconeDB::__init__")

    def create_index(self, tenant: str) -> Index:
        return PineconeIndex()

