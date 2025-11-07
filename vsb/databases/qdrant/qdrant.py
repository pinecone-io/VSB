import logging
import time
from locust.exception import StopUser
import numpy as np

import vsb
from vsb import logger

from ..base import DB, Namespace
from ...vsb_types import Record, SearchRequest, DistanceMetric, RecordList


from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, after_log


class QdrantNamespace(Namespace):
    def __init__(self, client: QdrantClient, index_name: str):
        # TODO: Support multiple namespaces
        self.client = client
        self.index_name = index_name

    def insert_batch(self, batch: RecordList):
        # Qdrant expects a list of PointStruct objects.
        data = self.data_upload_body(batch)
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_insert_with_retry():
            return self.client.upsert(collection_name=self.index_name, points=data)
        
        upload_response = do_insert_with_retry()
        logger.debug(f"QdrantDB: response from data upload API: {upload_response}") # TODO: Change this to debug after testing

    def update_batch(self, batch: list[Record]):
        # Qdrant treats insert and update as the same operation.
        self.insert_batch(batch)

    def search(self, request: SearchRequest) -> list[str]:
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_query_with_retry():
            return self.index.query_points(
                collection_name=self.index_name, query=request.values, limit=request.top_k, query_filter=request.filter
            )

        result = do_query_with_retry()
        matches = [m.id for m in result.points]
        return matches

    def fetch_batch(self, request: list[str]) -> list[Record]:
        # Fetching records not directly supported; requires implementation of metadata storage
        raise NotImplementedError("fetch_batch not supported for Qdrant")

    def delete_batch(self, request: list[str]):
        # deleting the records not directly supported; requires implementation of delete by vsb_vec_id in the metadata
        raise NotImplementedError("delete_batch not supported for Qdrant")

    def data_upload_body(self, batch: RecordList) -> list[dict]:
        points = []
        for rec in batch:
            points.append(
                PointStruct(
                    id=rec.id,
                    vector=rec.values,
                    payload=rec.metadata if hasattr(rec, "metadata") and isinstance(rec.metadata, dict) else {},
                )
            )
        return points


class QdrantDB(DB):
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: DistanceMetric,
        name: str,
        config: dict,
    ):
        self.api_key = config["qdrant_api_key"]
        self.host = config["qdrant_host"]
        self.port = config["qdrant_port"]
        self.skip_populate = config["skip_populate"]
        self.overwrite = config["overwrite"]
        self.index_name = config["qdrant_collection_name"]
        self.dimensions = dimensions
        self.metric = metric

        if self.index_name is None:
            # None specified, default to "vsb-<workload>"
            self.index_name = f"vsb-{name}"

        # Create the Qdrant client
        self.client = QdrantClient(url=f"http://{self.host}:{self.port}", api_key=self.api_key)


    def close(self):
        self.client.close()

    def create_index(self):
        # Create the Qdrant index
        try:
            if not self.client.collection_exists(self.index_name):
                self.client.create_collection(
                    collection_name=self.index_name,
                    vectors_config=VectorParams(size=self.dimensions, distance=DistanceMetric._get_distance_func(self.metric.value)),
                )
                self.created_index = True
            else:
                self.check_index_config()
                self.created_index = False
        except Exception as e:
            logger.critical(f"Error creating Qdrant index '{self.index_name}': {e}")
            raise StopUser()
    
    def check_index_config(self):
        collection_info = self.client.get_collection(self.index_name)
        index_dims = collection_info.config.params.vectors.size
        if self.dimensions != index_dims:
            raise ValueError(
                f"Qdrant index '{self.index_name}' has incorrect dimensions - expected:{self.dimensions}, found:{index_dims}"
            )
        index_metric = collection_info.config.params.vectors.distance
        if self.metric.value != index_metric:
            raise ValueError(
                f"Qdrant index '{self.index_name}' has incorrect metric - expected:{self.metric.value}, found:{index_metric}"
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
        # determine how many we could fit in the max message size of 30MB.
        max_sparse_values = 0  # TODO: Add sparse values
        max_record_size = max_id + max_metadata + max_values + max_sparse_values
        max_namespace = 500  # Only one namespace per VectorUpsert request.
        size_based_batch_size = ((30 * 1024 * 1024) - max_namespace) // max_record_size
        max_batch_size = 1000
        #batch_size = min(size_based_batch_size, max_batch_size)
        batch_size = 1500   # TODO: Troubleshooting ingestion failures, remove it later
        logger.debug(f"QdrantDB.get_batch_size() - Using batch size of {batch_size}")
        return batch_size

    def get_namespace(self, namespace: str) -> Namespace:
        return QdrantNamespace(self.client, self.index_name)

    def initialize_population(self):
        # If the index already existed before VSB (we didn't create it) and
        # user didn't specify skip_populate; require --overwrite before
        # deleting the existing index.
        if self.skip_populate:
            return
        self.create_index()
        if not self.created_index and not self.overwrite:
            msg = (
                f"QdrantDB: Collection '{self.index_name}' already exists - cowardly "
                f"refusing to overwrite existing data. Specify --overwrite to "
                f"delete it, or specify --skip_populate to skip population phase."
            )
            logger.critical(msg)
            raise StopUser()
        if not self.created_index:
            try:
                logger.info(
                    f"QdrantDB: Deleting existing collection '{self.index_name}' before "
                    f"population (--overwrite=True)"
                )
                self.client.delete_collection(self.index_name)
                logger.info(f"Collection '{self.index_name}' cleared for population")
                time.sleep(10)
                self.create_index()
            except Exception as e:
                logger.critical(f"Error deleting collection '{self.index_name}': {e}")
                raise StopUser()

    def finalize_population(self, record_count: int):
        """Wait until all records are visible in the index"""
        logger.debug(f"QdrantDB: Waiting for record count to reach {record_count}")
        with vsb.logging.progress_task(
            "  Finalize population", "  ✔ Finalize population", total=record_count
        ) as finalize_id:
            while True:
                index_count = self.get_record_count()
                if vsb.progress:
                    vsb.progress.update(finalize_id, completed=index_count)
                if index_count >= record_count:
                    logger.debug(
                        f"QdrantDB: Collection vector count reached {index_count}, "
                        f"finalize is complete"
                    )
                    break
                time.sleep(1)

    def skip_refinalize(self):
        return False

    def get_record_count(self) -> int:
        return self.client.get_collection(self.index_name).points_count

    @staticmethod
    def _get_distance_func(metric: DistanceMetric) -> str:
        match metric:
            case DistanceMetric.Cosine:
                return "COSINE"
            case DistanceMetric.Euclidean:
                return "EUCLID"
            case DistanceMetric.DotProduct:
                return "DOT"
        raise ValueError("Invalid metric:{}".format(metric))
