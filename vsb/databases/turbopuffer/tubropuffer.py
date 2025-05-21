import logging

from locust.exception import StopUser

import vsb
from vsb import logger
import turbopuffer as tpuf
from pinecone import PineconeException, NotFoundException, UnauthorizedException
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
#grpc_gevent.init_gevent()


class TurbopufferNamespace(Namespace):
    def __init__(self, index, namespace, metric):
        # TODO: Support multiple namespaces
        self.index = index
        self.metric = metric

    def insert_batch(self, batch: RecordList):
        # Turbopuffer expects a list of dicts (or tuples).
        #dicts = self.data_upload_body(batch)
        dicts = [{"id": rec.id, "vector": rec.values} for rec in batch]
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_insert_with_retry():
            return self.index.write(upsert_rows=dicts, distance_metric=self.metric)
        
        upload_response = do_insert_with_retry()
        logger.debug(f"TurbopufferDB: response from data upload API: {upload_response}")

    def update_batch(self, batch: list[Record]):
        # Turbopuffer treats insert and update as the same operation.
        self.insert_batch(batch)

    def search(self, request: SearchRequest) -> list[str]:
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_query_with_retry():
            return self.index.query(
                rank_by=["vector", "ANN", request.values], top_k=request.top_k, filters=request.filter
            )

        result = do_query_with_retry()
        matches = [m.id for m in result.rows]
        return matches

    def fetch_batch(self, request: list[str]) -> list[Record]:
        return self.index.fetch(request).vectors.values

    def delete_batch(self, request: list[str]):
        self.index.delete(request)
    '''
    def data_upload_body(self, batch: RecordList) -> list[dict]:
        data = []
        for record in batch:
            data.append({
                "id": record.id,
                "vector": record.values
            }) # TODO: Add metadata to the data upload body with Keys unnested
        return data
    '''

class TurbopufferDB(DB):
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: DistanceMetric,
        name: str,
        config: dict,
    ):
        #self.pc = PineconeGRPC(config["pinecone_api_key"])
        self.metric = metric
        self.tpuf.api_key = config["turbopuffer_api_key"]
        self.skip_populate = config["skip_populate"]
        self.overwrite = config["overwrite"]
        self.index_name = config["turbopuffer_index_name"]
        self.tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"

        if self.index_name is None:
            # None specified, default to "vsb-<workload>"
            self.index_name = f"vsb-{name}"
        self.index = tpuf.Namespace(self.index_name)

        # Check if index exists
        namespaces = tpuf.namespaces()
        if self.index_name in namespaces:
            self.schema = self.index.schema()
            logger.info(f"TurbopufferDB: Index '{self.index_name}' already exists, and Schema for this index is '{self.schema}'")
            self.index_exists = True
        else:
            logger.info(f"TurbopufferDB: Index '{self.index_name}' does not exist, will be created during data population")
            self.index_exists = False

        '''
        try:
            self.index = self.pc.Index(name=self.index_name)
            self.created_index = False
        except UnauthorizedException:
            api_key = config["pinecone_api_key"]
            masked_api_key = api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]
            logger.critical(
                f"PineconeDB: Got UnauthorizedException when attempting to connect "
                f"to index '{self.index_name}' using API key '{masked_api_key}' - check "
                f"your API key and permissions"
            )
            raise StopUser()
        except NotFoundException:
            logger.info(
                f"PineconeDB: Specified index '{self.index_name}' was not found, or the "
                f"specified API key cannot access it. Creating new index '{self.index_name}'."
            )
            self.pc.create_index(
                name=self.index_name,
                dimension=dimensions,
                metric=metric.value,
                spec=spec,
            )
            self.index = self.pc.Index(name=self.index_name)
            self.created_index = True

        info = self.pc.describe_index(self.index_name)
        index_dims = info["dimension"]
        if dimensions != index_dims:
            raise ValueError(
                f"PineconeDB index '{self.index_name}' has incorrect dimensions - expected:{dimensions}, found:{index_dims}"
            )
        index_metric = info["metric"]
        if metric.value != index_metric:
            raise ValueError(
                f"PineconeDB index '{self.index_name}' has incorrect metric - expected:{metric.value}, found:{index_metric}"
            )
        '''

    def close(self):
        self.index.close()

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
        # determine how many we could fit in the max message size of 256MB.
        max_sparse_values = 0  # TODO: Add sparse values
        max_record_size = max_id + max_metadata + max_values + max_sparse_values
        max_namespace = 500  # Only one namespace per VectorUpsert request.
        size_based_batch_size = ((256 * 1024 * 1024) - max_namespace) // max_record_size
        max_batch_size = 10000
        batch_size = min(size_based_batch_size, max_batch_size)
        logger.debug(f"TurbopufferDB.get_batch_size() - Using batch size of {batch_size}")
        return batch_size

    def get_namespace(self, namespace: str) -> Namespace:
        return TurbopufferNamespace(self.index, namespace, self.metric)

    def initialize_population(self):
        # If the index already existed before VSB (we didn't create it) and
        # user didn't specify skip_populate; require --overwrite before
        # deleting the existing index.
        if self.skip_populate:
            return
        if self.index_exists and not self.overwrite:
            msg = (
                f"TurbopufferDB: Index '{self.index_name}' already exists - cowardly "
                f"refusing to overwrite existing data. Specify --overwrite to "
                f"delete it, or specify --skip_populate to skip population phase."
            )
            logger.critical(msg)
            raise StopUser()
        if self.index_exists and self.overwrite:
            try:
                logger.info(
                    f"TurbopufferDB: Deleting existing index '{self.index_name}' before "
                    f"population (--overwrite=True)"
                )
                self.index.delete_all()
            except Exception as e:
                logger.error(f"TurbopufferDB: Error deleting existing index '{self.index_name}' before population: {e}")
                raise StopUser()
            
        logger.info(f"Initialize Population: Index '{self.index_name}' does not exist, will be created during data population")

    def finalize_population(self, record_count: int):
        """Wait until all records are visible in the index"""
        logger.debug(f"TurbopufferDB: Waiting for record count to reach {record_count}")
        time.sleep(30) # TODO: Remove this after we get the API to get the record count from Turbopuffer index
        with vsb.logging.progress_task(
            "  Finalize population", "  ✔ Finalize population", total=record_count
        ) as finalize_id:
            while True:
                #index_count = self.index.describe_index_stats()["total_vector_count"]
                index_count = record_count # TODO: Remove this after we get the API to get the record count from Turbopuffer index
                if vsb.progress:
                    vsb.progress.update(finalize_id, completed=index_count)
                if index_count >= record_count:
                    logger.debug(
                        f"TurbopufferDB: Index vector count reached {index_count}, "
                        f"finalize is complete"
                    )
                    break
                time.sleep(1)

    #def skip_refinalize(self):
    #    return False

    #def get_record_count(self) -> int:
    #    return self.index.describe_index_stats()["total_vector_count"]
