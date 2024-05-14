import logging

from pinecone import PineconeException
from pinecone.grpc import PineconeGRPC, GRPCIndex
import grpc.experimental.gevent as grpc_gevent
import time

from ..base import DB, Namespace
from ...vsb_types import Record, SearchRequest, DistanceMetric, RecordList

# patch grpc so that it uses gevent instead of asyncio. This is required to
# allow the multiple coroutines used by locust to run concurrently. Without it
# (using default asyncio) will block the whole Locust/Python process,
# in practice limiting to running a single User per worker process.
grpc_gevent.init_gevent()


class PineconeNamespace(Namespace):
    def __init__(self, index: GRPCIndex, namespace: str):
        # TODO: Support multiple namespaces
        self.index = index

    def upsert(self, ident, vector, metadata):
        raise NotImplementedError

    def upsert_batch(self, batch: RecordList):
        # Pinecone expects a list of dicts (or tuples).
        dicts = [dict(rec) for rec in batch]
        self.index.upsert(dicts)

    def search(self, request: SearchRequest) -> list[str]:
        result = self.index.query(
            vector=request.values, top_k=request.top_k, filter=request.filter
        )
        matches = [m["id"] for m in result["matches"]]
        return matches


class PineconeDB(DB):
    def __init__(self, dimensions: int, metric: DistanceMetric, config: dict):
        self.pc = PineconeGRPC(config["pinecone_api_key"])
        index_name = config["pinecone_index_name"]
        self.index = self.pc.Index(name=index_name)
        info = self.pc.describe_index(index_name)
        index_dims = info["dimension"]
        if dimensions != index_dims:
            raise ValueError(
                f"PineconeDB index '{index_name}' has incorrect dimensions - expected:{dimensions}, found:{index_dims}"
            )
        index_metric = info["metric"]
        if metric.value != index_metric:
            raise ValueError(
                f"PineconeDB index '{index_name}' has incorrect metric - expected:{metric.value}, found:{index_metric}"
            )
        # Start with an empty index if we are going to populate it.
        if not config["skip_populate"]:
            try:
                self.index.delete(delete_all=True)
            except PineconeException as e:
                # Serverless indexes can throw a "Namespace not found" exception for
                # delete_all if there are no documents in the index. Simply ignore,
                # as the post-condition is the same.
                pass

    def get_namespace(self, namespace: str) -> Namespace:
        return PineconeNamespace(self.index, namespace)

    def finalize_population(self, record_count: int):
        logging.debug(f"PineconeDB: Waiting for record count to reach {record_count}")
        """Wait until all records are visible in the index"""
        while True:
            index_count = self.index.describe_index_stats()["total_vector_count"]
            if index_count >= record_count:
                logging.debug(
                    f"PineconeDB: Index vector count reached {index_count}, "
                    f"finalize is complete"
                )
                break
            time.sleep(1)
