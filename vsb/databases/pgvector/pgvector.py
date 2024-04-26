from ..vectordb import VectorDB


class PGVectorDB(VectorDB):
    def __init__(self):
        print("PGVectorDB::__init__")

    def upsert(self, ident, vector, metadata):
        pass

    def search(self, query_vector):
        pass
