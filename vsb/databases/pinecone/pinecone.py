import logging

from locust.exception import StopUser

import vsb
from vsb import logger
from pinecone import PineconeException, NotFoundException
from pinecone.grpc import PineconeGRPC, GRPCIndex
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, after_log
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

    def upsert_batch(self, batch: RecordList):
        # Pinecone expects a list of dicts (or tuples).
        dicts = [dict(rec) for rec in batch]
        self.index.upsert(dicts)

    def search(self, request: SearchRequest) -> list[str]:
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_query_with_retry():
            return self.index.query(
                vector=request.values, top_k=request.top_k, filter=request.filter
            )

        result = do_query_with_retry()
        matches = [m["id"] for m in result["matches"]]
        return matches


class PineconeDB(DB):
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: DistanceMetric,
        name: str,
        config: dict,
    ):
        self.pc = PineconeGRPC(config["pinecone_api_key"])
        self.skip_populate = config["skip_populate"]
        index_name = config["pinecone_index_name"]
        try:
            self.index = self.pc.Index(name=index_name)
        except NotFoundException as e:
            logger.error(
                f"PineconeDB: Specified index '{index_name}' was not found. Check the "
                f"index exists and the specified API key can access it."
            )
            raise StopUser() from e
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

    def get_batch_size(self, sample_record: Record) -> int:
        # Return the largest batch size possible, based on the following
        # constraints:
        # - Max id length is 512 bytes
        # - Max namespace length is 500 bytes.
        # - Max metadata size is 40KiB.
        # - Maximum sparse value count is 1000
        #   - Sparse values are made up sequence of pairs of int and float.
        # - Maximum dense vector count is 1000.
        # Given the above, calculate the maximum possible sized record, based
        # on which fields are present in the sample record.
        max_id = 512
        max_values = len(sample_record.values) * 4
        max_metadata = 40 * 1024 if sample_record.metadata else 0
        # determine how many we could fit in the max message size of 2MB.
        max_sparse_values = 0  # TODO: Add sparse values
        max_record_size = max_id + max_metadata + max_values + max_sparse_values
        max_namespace = 500  # Only one namespace per VectorUpsert request.
        size_based_batch_size = ((2 * 1024 * 1024) - max_namespace) // max_record_size
        max_batch_size = 1000
        batch_size = min(size_based_batch_size, max_batch_size)
        logger.debug(f"PineconeDB.get_batch_size() - Using batch size of {batch_size}")
        return batch_size

    def get_namespace(self, namespace: str) -> Namespace:
        return PineconeNamespace(self.index, namespace)

    def initialize_population(self):
        # Start with an empty index if we are going to populate it.
        if not self.skip_populate:
            try:
                self.index.delete(delete_all=True)
            except PineconeException as e:
                # Serverless indexes can throw a "Namespace not found" exception for
                # delete_all if there are no documents in the index. Simply ignore,
                # as the post-condition is the same.
                pass

    def finalize_population(self, record_count: int):
        """Wait until all records are visible in the index"""
        logger.debug(f"PineconeDB: Waiting for record count to reach {record_count}")
        with vsb.logging.progress_task(
            "  Finalize population", "  ✔ Finalize population", total=record_count
        ) as finalize_id:
            while True:
                index_count = self.index.describe_index_stats()["total_vector_count"]
                if vsb.progress:
                    vsb.progress.update(finalize_id, completed=index_count)
                if index_count >= record_count:
                    logger.debug(
                        f"PineconeDB: Index vector count reached {index_count}, "
                        f"finalize is complete"
                    )
                    break
                time.sleep(1)
