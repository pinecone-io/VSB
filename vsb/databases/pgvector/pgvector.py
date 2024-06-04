import logging

import numpy as np
import pgvector.psycopg
import psycopg
from psycopg.types.json import Jsonb

from .filter_util import FilterUtil
from ..base import DB, Namespace
from ...vsb_types import Record, DistanceMetric, RecordList, SearchRequest


class PgvectorNamespace(Namespace):
    """For pgvector, the VSB namespace abstraction maps to a postgres table
    plus associated pgvector index.
    """

    def __init__(self, connection, table: str, metric: DistanceMetric, namespace: str):
        # TODO: Support multiple namespaces
        self.conn = connection
        self.table = table
        self.metric = metric

    def upsert(self, ident, vector, metadata):
        raise NotImplementedError

    def upsert_batch(self, batch: RecordList):
        # pgvector / psycopg expects a list of tuples.
        data = [(rec.id, np.array(rec.values), Jsonb(rec.metadata)) for rec in batch]
        upsert_query = (
            "INSERT INTO " + self.table + " (id, embedding, metadata) "
            "VALUES (%s, %s, %s)"
        )
        with self.conn.cursor() as cur:
            cur.executemany(upsert_query, data)

    def search(self, request: SearchRequest) -> list[str]:
        match self.metric:
            case DistanceMetric.Cosine:
                operator = "<=>"
            case DistanceMetric.Euclidean:
                operator = "<->"
            case DistanceMetric.DotProduct:
                operator = "<#>"
        where = FilterUtil.to_sql(request.filter)
        select_query = (
            f"SELECT id FROM {self.table} {where} ORDER BY embedding "
            f"{operator} %s "
            f"LIMIT %s"
        )
        result = self.conn.execute(
            select_query, (np.array(request.values), request.top_k)
        ).fetchall()
        matches = [r[0] for r in result]
        return matches


class PgvectorDB(DB):
    def __init__(
        self, dimensions: int, metric: DistanceMetric, name: str, config: dict
    ):
        self.index_type = config["pgvector_index_type"]
        match self.index_type:
            case "hnsw":
                pass
            case "ivfflat":
                self.ivfflat_lists = config["pgvector_ivfflat_lists"]
            case _:
                raise ValueError(
                    "Unsupported pgvector index type {}".format(self.index_type)
                )
        self.metric = metric
        # Postgres doesn't like "-" in identifier names so sanitize when
        # forming table name from workload name.
        self.table = name.replace("-", "_")
        self.skip_populate = config["skip_populate"]
        self.dimensions = dimensions

        # Connect to postgres and setup pgvector support.
        self.conn = psycopg.connect(
            host=config["pgvector_host"],
            user=config["pgvector_username"],
            password=config["pgvector_password"],
            dbname=config["pgvector_database"],
            autocommit=True,
        )
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        pgvector.psycopg.register_vector(self.conn)

    def get_batch_size(self, sample_record: Record) -> int:
        # Initially use a fixed batch size of 1000; this seems to be
        # a reasonable trade-off between network / protocol overhead
        # and not too large a transaction for a range of vector dimensions.
        return 1000

    def get_namespace(self, namespace_name: str) -> Namespace:
        return PgvectorNamespace(self.conn, self.table, self.metric, namespace_name)

    def initialize_population(self):
        # Start with an empty table if we are going to populate it.
        if not self.skip_populate:
            self.conn.execute("DROP TABLE IF EXISTS " + self.table)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS "
            + self.table
            + " (id VARCHAR PRIMARY KEY, embedding vector("
            + str(self.dimensions)
            + "), metadata JSONB)",
        )

    def finalize_population(self, record_count: int):
        # Create index.
        sql = (
            f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx ON "
            f"{self.table} USING {self.index_type} (embedding "
            f"{PgvectorDB._get_distance_func(self.metric)})"
        )
        match self.index_type:
            case "ivfflat":
                sql += f" WITH (lists = {self.ivfflat_lists})"
        self.conn.execute(sql)

    @staticmethod
    def _get_distance_func(metric: DistanceMetric) -> str:
        match metric:
            case DistanceMetric.Cosine:
                return "vector_cosine_ops"
            case DistanceMetric.Euclidean:
                return "vector_l2_ops"
            case DistanceMetric.DotProduct:
                return "vector_ip_ops"
        raise ValueError("Invalid metric:{}".format(metric))
