import logging
from typing import Callable, Any

from pinecone import PineconeException
from pinecone.grpc import PineconeGRPC, GRPCIndex
import grpc.experimental.gevent as grpc_gevent
import grpc
import time

from ..base import DB, Namespace
from ...vsb_types import Vector, Record, SearchRequest, DistanceMetric

# patch grpc so that it uses gevent instead of asyncio. This is required to
# allow the multiple coroutines used by locust to run concurrently. Without it
# (using default asyncio) will block the whole Locust/Python process,
# in practice limiting to running a single User per worker process.
grpc_gevent.init_gevent()

from grpc_interceptor import ClientCallDetails, ClientInterceptor


class MetadataClientInterceptor(ClientInterceptor):

    def intercept(
        self,
        method: Callable,
        request_or_iterator: Any,
        call_details: grpc.ClientCallDetails,
    ):
        """Override this method to implement a custom interceptor.

        This method is called for all unary and streaming RPCs. The interceptor
        implementation should call `method` using a `grpc.ClientCallDetails` and the
        `request_or_iterator` object as parameters. The `request_or_iterator`
        parameter may be type checked to determine if this is a singluar request
        for unary RPCs or an iterator for client-streaming or client-server streaming
        RPCs.

        Args:
            method: A function that proceeds with the invocation by executing the next
                interceptor in the chain or invoking the actual RPC on the underlying
                channel.
            request_or_iterator: RPC request message or iterator of request messages
                for streaming requests.
            call_details: Describes an RPC to be invoked.

        Returns:
            The type of the return should match the type of the return value received
            by calling `method`. This is an object that is both a
            `Call <https://grpc.github.io/grpc/python/grpc.html#grpc.Call>`_ for the
            RPC and a `Future <https://grpc.github.io/grpc/python/grpc.html#grpc.Future>`_.

            The actual result from the RPC can be got by calling `.result()` on the
            value returned from `method`.
        """
        new_details = ClientCallDetails(
            call_details.method,
            call_details.timeout,
            call_details.metadata,
            call_details.credentials,
            call_details.wait_for_ready,
            call_details.compression,
        )

        print(f"Intercept method: {call_details.method}")
        future = method(request_or_iterator, new_details)
        future.result()
        print(f"< Code: {future.code()}")
        if call_details.method == "/VectorService/Upsert":
            future._call._state.code = grpc.StatusCode.UNAVAILABLE
        print(f"! Modified Code: {future.code()}")
        return future


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
        self.pc = PineconeGRPC(config["pinecone_api_key"])
        index_name = config["pinecone_index_name"]
        self.index = self.pc.Index(
            name=index_name, interceptors=[MetadataClientInterceptor()]
        )
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
