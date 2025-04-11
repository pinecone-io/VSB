import logging
import time
from locust.exception import StopUser

import vsb
from vsb import logger

from ..base import DB, Namespace
from ...vsb_types import Record, SearchRequest, DistanceMetric, RecordList

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, after_log
import numpy as np


class OpenSearchNamespace(Namespace):
    def __init__(
        self, client: OpenSearch, service: str, index_name: str, dimensions: int, namespace: str
    ):
        self.client = client
        self.service = service
        self.index_name = index_name
        self.dimensions = dimensions
        self.namespace = namespace

    def insert_batch(self, batch: RecordList):
        data = self.bulk_upload_body(batch)
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_insert_with_retry():
            return helpers.bulk(self.client, data)
        
        upload_response = do_insert_with_retry()
        #logger.debug(f"OpenSearchDB: response from bulk helper upload: {upload_response}")

    def update_batch(self, batch: list[Record]):
        self.insert_batch(batch)

    def search(self, request: SearchRequest) -> list[str]:
        query = self.search_query_body(request)
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_query_with_retry():
            return self.client.search(body=query, index=self.index_name)

        response = do_query_with_retry()
        # sending the VSB Id's of the top k results
        ids = self.search_response_to_record_list(response)
        #logger.debug(f"OpenSearchDB: response IDs from search: {ids}")
        return ids

    def fetch_batch(self, request: list[str]) -> list[Record]:
        # Fetching records not directly supported; requires implementation of metadata storage
        raise NotImplementedError("fetch_batch not supported for OpenSearch")

    def delete_batch(self, request: list[str]):
        # deleting the records not directly supported; requires implementation of delete by vsb_vec_id in the metadata
        raise NotImplementedError("delete_batch not supported for OpenSearch")
    
    def bulk_upload_body(self, batch: RecordList) -> list[dict]:
        match self.service:
            case "aoss":
                data = []
                for rec in batch:
                    data.append({"_index": self.index_name, "vsb_vec_id": rec.id, "v_content": np.array(rec.values)})                 
            case "es":
                data = []
                for rec in batch:
                    data.append({"_index": self.index_name, "_id": rec.id, "v_content": np.array(rec.values)})
            case _:
                raise ValueError(f"Invalid service: {self.service}")
        return data
    
    def search_query_body(self, request: SearchRequest) -> dict:
        match self.service:
            case "aoss":
                query = {
                    "size": request.top_k,
                    "fields": ["vsb_vec_id"],
                    "_source": False,
                    "query": {
                        "knn": {
                            "v_content": {"vector": request.values, "k": self.dimensions}
                        }
                    },
                }
            case "es":
                query = {
                    "size": request.top_k,
                    "_source": False,
                    "query": {
                        "knn": {
                            "v_content": {"vector": request.values, "k": self.dimensions}
                        }
                    },
                }
            case _:
                raise ValueError(f"Invalid service: {self.service}")
        return query
            
    def search_response_to_record_list(self, response: dict) -> list[Record]:
        ids = []
        match self.service:
            case "aoss":
                ids = [m["fields"]["vsb_vec_id"][0] for m in response["hits"]["hits"]]
            case "es":
                ids = [m["_id"] for m in response["hits"]["hits"]]
            case _:
                raise ValueError(f"Invalid service: {self.service}")
        return ids
    


class OpenSearchDB(DB):
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: str,
        name: str,
        config: dict,
    ):
        self.region = config["opensearch_region"]
        self.service = config["opensearch_service"]
        self.access_key = config["aws_access_key"]
        self.secret_key = config["aws_secret_key"]
        self.token = config["aws_session_token"]

        self.index_name = config["opensearch_index_name"]
        self.skip_populate = config["skip_populate"]
        self.overwrite = config["overwrite"]
        self.dimensions = dimensions
        self.metric = metric

        # Create the OpenSearch client
        if self.access_key and self.secret_key and self.region and self.service and self.token:
            logger.info(
                f"OpenSearchDB: Using AWS credentials for OpenSearch client, region: {self.region}, service: {self.service}"
            )
            auth = AWS4Auth(
                self.access_key,
                self.secret_key,
                self.region,
                self.service,
                session_token=self.token,
            )
        elif config["opensearch_username"] and config["opensearch_password"]:
            logger.info(
                "OpenSearchDB: Using username / password credentials for OpenSearch client"
            )
            auth = (config["opensearch_username"], config["opensearch_password"])
        else:
            logger.critical(
                "OpenSearchDB: No AWS credentials or username/password provided for "
                "OpenSearch client. Please specify either AWS credentials or username & password."
            )
            raise StopUser()

        use_tls = config["opensearch_use_tls"]

        self.client = OpenSearch(
            hosts=[{"host": config["opensearch_host"],
                    "port": config["opensearch_port"]}],
            http_auth=auth,
            timeout=900,
            use_ssl=use_tls,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )

        # Generate index name if not specified
        if self.index_name is None:
            # None specified, default to "vsb-<workload>"
            self.index_name = f"vsb-{name}"

    def close(self):
        self.client.close()

    def create_index(self):
        # Create the index
        index_body = self.create_index_body()

        if not self.client.indices.exists(self.index_name):
            logger.info(
                f"OpenSearchDB: Specified index '{self.index_name}' was not found, or the "
                f"specified AWS Access keys cannot access it. Creating new index '{self.index_name}'."
            )
            self.client.indices.create(index=self.index_name, body=index_body)
            self.created_index = True
            time.sleep(30)
        else:
            self.created_index = False

    def get_batch_size(self, sample_record: Record) -> int:
        # Return the largest batch size possible, based on the following
        # constraints:
        # - Max id length is 512 bytes
        # - Max index name length is 500 bytes.
        # - Max metadata size is 40KiB. 
        # - Maximum sparse value count is 1000
        #   - Sparse values are made up sequence of pairs of int and float.
        # - Maximum dense vector count is 1000.
        # Given the above, calculate the maximum possible sized record, based
        # on which fields are present in the sample record.
        max_id = 512
        max_values = len(sample_record.values) * 4
        max_metadata = 40 * 1024 if sample_record.metadata else 0
        # determine how many we could fit in the max message size of 3MB.
        max_sparse_values = 0  # TODO: Add sparse values
        max_indexname = 500  # Each record has the index name specified.
        max_record_size = max_id + max_metadata + max_values + max_sparse_values + max_indexname
        size_based_batch_size = (3 * 1024 * 1024) // max_record_size
        max_batch_size = 1000
        #batch_size = min(size_based_batch_size, max_batch_size)
        batch_size = 100  # TODO: Troubleshooting ingestion failures, remove it later
        logger.debug(f"OpenSearchDB.get_batch_size() - Using batch size of {batch_size}")
        return batch_size

    def get_namespace(self, namespace: str) -> Namespace:
        return OpenSearchNamespace(
            self.client, self.service, self.index_name, self.dimensions, namespace
        )

    def initialize_population(self):
        if self.skip_populate:
            return
        self.create_index()
        if not self.created_index and not self.overwrite:
            msg = (
                f"OpenSearchDB: Index '{self.index_name}' already exists - cowardly "
                f"refusing to overwrite existing data. Specify --overwrite to "
                f"delete it, or specify --skip_populate to skip population phase."
            )
            logger.critical(msg)
            raise StopUser()
        if not self.created_index:
            try:
                logger.info(
                    f"OpenSearchDB: Deleting existing index '{self.index_name}' before "
                    f"population (--overwrite=True)"
                )
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Index '{self.index_name}' cleared for population")
                time.sleep(10)
                self.create_index()
            except Exception as e:
                logger.critical(f"Error deleting index '{self.index_name}': {e}")
                raise StopUser()

    def finalize_population(self, record_count: int):
        """Wait until all records are visible in the index"""
        logger.debug(f"OpenSearchDB: Waiting for record count to reach {record_count}")
        with vsb.logging.progress_task(
            "  Finalize population", "  ✔ Finalize population", total=record_count
        ) as finalize_id:
            while True:
                index_count = self.get_record_count()
                if vsb.progress:
                    vsb.progress.update(finalize_id, completed=index_count)
                if index_count >= record_count:
                    logger.debug(
                        f"OpenSearchDB: Index vector count reached {index_count}, "
                        f"finalize is complete"
                    )
                    break
                time.sleep(1)

    def skip_refinalize(self):
        return False

    def get_record_count(self) -> int:
        return self.client.count(index=self.index_name)["count"]
    
    def create_index_body(self) -> dict:
        match self.service:
            case "aoss":
                index_body = {
                    "settings": {"index.knn": True},
                    "mappings": {"properties": {
                        "v_content": {"type": "knn_vector", "dimension": self.dimensions, "method": {"name": "hnsw", "space_type": OpenSearchDB._get_distance_func(self.metric), "engine": OpenSearchDB._get_engine_func(self.metric)}},
                        "vsb_vec_id": {"type": "text", "fields": {"keyword": {"type": "keyword"}},},
                    },
                },
            }
            case "es":
                index_body = {
                    "settings": {"index.knn": True},
                    "mappings": {"properties": {"v_content": {"type": "knn_vector", "dimension": self.dimensions, "method": {"name": "hnsw", "space_type": OpenSearchDB._get_distance_func(self.metric), "engine": OpenSearchDB._get_engine_func(self.metric)}},
                    }
                },
                }
            case _:
                raise ValueError(f"Invalid service: {self.service}")
        return index_body

    @staticmethod
    def _get_distance_func(metric: DistanceMetric) -> str:
        match metric:
            case DistanceMetric.Cosine:
                return "cosinesimil"
            case DistanceMetric.Euclidean:
                return "l2"
            case DistanceMetric.DotProduct:
                return "innerproduct"
        raise ValueError("Invalid metric:{}".format(metric))
    
    @staticmethod
    def _get_engine_func(metric: DistanceMetric) -> str:
        match metric:
            case DistanceMetric.Cosine:
                return "nmslib"
            case DistanceMetric.Euclidean:
                return "faiss"
            case DistanceMetric.DotProduct:
                return "faiss"
        raise ValueError("Invalid metric:{}".format(metric))
