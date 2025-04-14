from enum import Enum
from .base import DB


class Database(Enum):
    """Set of supported database backends, the value is the string used to
    specify via --database="""

    Pinecone = "pinecone"
    OpenSearch = "opensearch"
    PGVector = "pgvector"

    def get_class(self) -> type[DB]:
        """Return the DB class to use, based on the value of the enum"""
        match self:
            case Database.Pinecone:
                from .pinecone.pinecone import PineconeDB

                return PineconeDB
            case Database.OpenSearch:
                from .opensearch.opensearch import OpenSearchDB

                return OpenSearchDB
            case Database.PGVector:
                from .pgvector.pgvector import PgvectorDB

                return PgvectorDB
