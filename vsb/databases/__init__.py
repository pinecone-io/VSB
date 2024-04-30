from enum import Enum
from .base import DB


class Database(Enum):
    """Set of supported database backends, the value is the string used to
    specify via --database="""

    Pinecone = "pinecone"
    PGVector = "pgvector"

    def get_class(self) -> type[DB]:
        """Return the DB class to use, based on the value of the enum"""
        match self:
            case Database.Pinecone:
                from .pinecone.pinecone import PineconeDB

                return PineconeDB
            case Database.PGVector:
                from .pgvector.pgvector import PGVectorDB

                return PGVectorDB
