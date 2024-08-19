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
        self.warned_no_metadata = False

    def upsert_batch(self, batch: RecordList):
        # pgvector / psycopg expects a list of tuples.
        data = [(rec.id, np.array(rec.values), Jsonb(rec.metadata)) for rec in batch]

        # Warn the user once if they're using a GIN index on a
        # dataset that doesn't have metadata.
        if not self.warned_no_metadata:
            if all([rec.metadata is None for rec in batch]):
                self.warned_no_metadata = True
                logger.warning(
                    f"You're using a {self.index_type} index type, "
                    f"but this workload doesn't seem to have metadata. "
                    f"Are you sure this is correct?"
                )

        upsert_query = (
            "INSERT INTO " + self.table + " (id, embedding, metadata) "
            "VALUES (%s, %s, %s)"
        )
        with self.conn.cursor() as cur:
            cur.executemany(upsert_query, data)

    def search(self, request: SearchRequest) -> list[str]:
        match self.index_type:
            case "hnsw" | "hnsw+gin":
                # For HNSW, we use a default of 2 * top_k for ef_search. See https://github.com/pgvector/pgvector.
                setup_search_statement = f"SET hnsw.ef_search = {(2 * request.top_k) if self.search_candidates == 0 else self.search_candidates}"
            case "ivfflat" | "ivfflat+gin":
                # For IVFFLAT, we use a default of sqrt(lists) for probes. See https://github.com/pgvector/pgvector.
                setup_search_statement = f"SET ivfflat.probes = {math.isqrt(self.ivfflat_lists) if self.search_candidates == 0 else self.search_candidates}"
            case "gin" | "none":
                setup_search_statement = None
            case _:
                raise ValueError(
                    "Unsupported pgvector index type {}".format(self.index_type)
                )
        if setup_search_statement:
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

    def delete_batch(self, request: list[str]):
        delete_query = f"DELETE FROM {self.table} WHERE id = ANY(%s)"
        self.conn.execute(delete_query, (request,))

    def fetch_batch(self, request: list[str]) -> list[Record]:
        select_query = (
            f"SELECT id, embedding, metadata FROM {self.table} WHERE id = ANY(%s)"
        )
        result = self.conn.execute(select_query, (request,)).fetchall()
        records = [Record(id=r[0], values=r[1], metadata=r[2]) for r in result]
        return records


class PgvectorDB(DB):
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: DistanceMetric,
        name: str,
        config: dict,
    ):
        self.index_type = config["pgvector_index_type"]
        match self.index_type:
            case "hnsw" | "hnsw+gin" | "gin" | "none":
                self.ivfflat_lists = None
            case "ivfflat" | "ivfflat+gin":
                self.ivfflat_lists = config["pgvector_ivfflat_lists"]
                if self.ivfflat_lists == 0:
                    # Automatically calculate number of lists as per
                    # pgvector docs recommendation.
                    if record_count <= 1_000_000:
                        self.ivfflat_lists = max(1, record_count // 1_000)
                    else:
                        self.ivfflat_lists = math.isqrt(record_count)
                    logger.debug(
                        "PgvectorDB: automatically calculated IVFFlat lists="
                        f"{self.ivfflat_lists}"
                    ),
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
        maintenance_work_mem = config.get("pgvector_maintenance_work_mem")
        self.conn.execute(f"SET maintenance_work_mem = '{maintenance_work_mem}'")

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
            "  Create pgvector table",
            "  ✔ pgvector table created",
            total=None,
        ):
            with self.conn.cursor() as cur:
                # Disable notices around DROP TABLE / CREATE TABLE - Postgres reports a
                # notice if the table already exists - even when using
                # "IF NOT EXISTS"
                cur.execute("SET client_min_messages TO ERROR")

                # Start with an empty table if we are going to populate it.
                if not self.skip_populate:
                    cur.execute("DROP TABLE IF EXISTS " + self.table)
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS "
                    + self.table
                    + " (id VARCHAR PRIMARY KEY, embedding vector("
                    + str(self.dimensions)
                    + "), metadata JSONB);",
                )
                cur.execute("RESET client_min_messages")

    def finalize_population(self, record_count: int):
        # Create index.
        if self.index_type == "none":
            return

        with vsb.logging.progress_task(
            f"  Create pgvector index ({self.index_type})",
            f"  ✔ pgvector index ({self.index_type}) created",
            total=None,
        ):
            if "hnsw" in self.index_type:
                sql = (
                    f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx ON "
                    f"{self.table} USING hnsw (embedding "
                    f"{PgvectorDB._get_distance_func(self.metric)})"
                )
                self.conn.execute(sql)
            if "ivfflat" in self.index_type:
                sql = (
                    f"CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx ON "
                    f"{self.table} USING ivfflat (embedding "
                    f"{PgvectorDB._get_distance_func(self.metric)}) WITH (lists = {self.ivfflat_lists})"
                )
                self.conn.execute(sql)
            if "gin" in self.index_type:
                sql = (
                    f"CREATE INDEX IF NOT EXISTS {self.table}_metadata_idx ON "
                    f"{self.table} USING gin (metadata)"
                )
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
