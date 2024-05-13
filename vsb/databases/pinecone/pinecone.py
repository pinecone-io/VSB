from pinecone import PineconeException
from pinecone.grpc import PineconeGRPC, GRPCIndex, GRPCClientConfig
import grpc.experimental.gevent as grpc_gevent

from ..base import DB, Namespace
from ...vsb_types import Vector, Record, SearchRequest, DistanceMetric

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

    def upsert_batch(self, batch: list[Record]):
        self.index.upsert(batch)

    def search(self, request: SearchRequest) -> list[str]:
        result = self.index.query(vector=request.values, top_k=request.top_k)
        matches = [m["id"] for m in result["matches"]]
        return matches


class PineconeDB(DB):
    def __init__(self, dimensions: int, metric: DistanceMetric, config: dict):
        self.pc = PineconeGRPC(config["pinecone_api_key"],
                               proxy_url="http://localhost:8088",
                               ssl_ca_certs="/home/daver/.mitmproxy/mitmproxy-ca-cert.pem")
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
        try:
            self.index.delete(delete_all=True)
        except PineconeException as e:
            # Serverless indexes can throw a "Namespace not found" exception for
            # delete_all if there are no documents in the index. Simply ignore,
            # as the post-condition is the same.
            pass

    def get_namespace(self, namespace: str) -> Namespace:
        return PineconeNamespace(self.index, namespace)
