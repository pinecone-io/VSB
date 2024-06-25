import numpy as np
import math
import pgvector.psycopg
import psycopg
from psycopg.types.json import Jsonb

import vsb
from .filter_util import FilterUtil
from ..base import DB, Namespace
from ...vsb_types import Record, DistanceMetric, RecordList, SearchRequest
from vsb import logger


class PgvectorNamespace(Namespace):
    """For pgvector, the VSB namespace abstraction maps to a postgres table
    plus associated pgvector index.
    """

    def __init__(
        self,
        connection,
        table: str,
        metric: DistanceMetric,
        index_type: str,
        search_candidates: int,
        ivfflat_lists: int,
        namespace: str,
    ):
        # TODO: Support multiple namespaces
        self.conn = connection
        self.table = table
        self.metric = metric
        self.index_type = index_type
        self.search_candidates = search_candidates
        self.ivfflat_lists = ivfflat_lists

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
        match self.index_type:
            case "hnsw":
                # For HNSW, we use a default of 2 * top_k for ef_search. See https://github.com/pgvector/pgvector.
                setup_search_statement = f"SET hnsw.ef_search = {(2 * request.top_k) if self.search_candidates == 0 else self.search_candidates}"
            case "ivfflat":
                # For IVFFLAT, we use a default of sqrt(lists) for probes. See https://github.com/pgvector/pgvector.
                setup_search_statement = f"SET ivfflat.probes = {math.isqrt(self.ivfflat_lists) if self.search_candidates == 0 else self.search_candidates}"
        self.conn.execute(setup_search_statement)
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
                self.ivfflat_lists = None
            case "ivfflat":
                self.ivfflat_lists = config["pgvector_ivfflat_lists"]
            case _:
                raise ValueError(
                    "Unsupported pgvector index type {}".format(self.index_type)
                )
        self.metric = metric
        self.search_candidates = config["pgvector_search_candidates"]
        # Postgres doesn't like "-" in identifier names so sanitize when
        # forming table name from workload name.
        self.table = name.replace("-", "_")
        self.skip_populate = config["skip_populate"]
        self.dimensions = dimensions

        # Connect to postgres and setup pgvector support.
        self.conn = psycopg.connect(
            host=config["pgvector_host"],
            port=config["pgvector_port"],
            user=config["pgvector_username"],
            password=config["pgvector_password"],
            dbname=config["pgvector_database"],
            autocommit=True,
        )
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        # Register handler to log notices. Note we do this after loading
        # the vector extension as that reports a notice if already loaded (even with
        # IF NOT EXISTS) and we don't want to log that.
        def log_notice(diag: psycopg.errors.Diagnostic):
            msg = f"pgvector: {diag.severity} - {diag.message_primary}"
            if diag.message_detail:
                msg += f". Details: {diag.message_detail}"
            if diag.message_hint:
                msg += f" Hint: {diag.message_hint}"
            logger.warning(msg)

        self.conn.add_notice_handler(log_notice)

        pgvector.psycopg.register_vector(self.conn)

    def get_batch_size(self, sample_record: Record) -> int:
        # Initially use a fixed batch size of 1000; this seems to be
        # a reasonable trade-off between network / protocol overhead
        # and not too large a transaction for a range of vector dimensions.
        return 1000

    def get_namespace(self, namespace_name: str) -> Namespace:
        return PgvectorNamespace(
            self.conn,
            self.table,
            self.metric,
            self.index_type,
            self.search_candidates,
            self.ivfflat_lists,
            namespace_name,
        )

    def initialize_population(self):
        with vsb.logging.progress_task(
            "  Create pgvector table", "  ✔ pgvector table created"
        ):
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
        with vsb.logging.progress_task(
            f"  Create pgvector index ({self.index_type})",
            f"  ✔ pgvector index ({self.index_type}) created",
        ):
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
