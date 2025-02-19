import logging
from logging import config
import time
from locust.exception import StopUser

import vsb
from vsb import logger

from ..base import DB, Namespace
from ...vsb_types import Record, SearchRequest, DistanceMetric, RecordList

from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import numpy as np

# OpenSearch-specific configurations
host = config["opensearch_host"]
region = config["opensearch_region"]
service = 'aoss'
access_key = ''
secret_key = ''
token = ''
awsauth = AWS4Auth(access_key, secret_key, region, service, session_token=token)

# Initialize the OpenSearch client
client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    timeout=300,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

class OpenSearchNamespace(Namespace):
    def __init__(self, index_name: str):
        self.index_name = index_name

    def insert_batch(self, batch: RecordList):
        actions = []
        action = {"index": {"_index": self.index_name}}
        for rec in batch:
            vector_document = {
                "vsb_vec_id": rec.id,
                "v_content": np.array(rec.values)
            }
            actions.append(action)
            actions.append(vector_document)

        # Bulk ingest documents
        client.bulk(body=actions)

    def update_batch(self, batch: list[Record]):
        self.insert_batch(batch)

    def search(self, request: SearchRequest) -> list[str]:
        query = {
            "size": request.top_k,
            "fields": ["vsb_vec_id"],
            "_source": False,
            "query": {
                "knn": {
                    "v_content": {
                        "vector": request.values,
                        "k": len(request.values)
                    }
                }
            }
        }
        response = client.search(body=query, index=self.index_name)
        #sending the VSB Id's of the top k results
        vsb_id = []
        [vsb_id.append(m["fields"]["vsb_vec_id"][0]) for m in response["hits"]["hits"]]
        return vsb_id

    def fetch_batch(self, request: list[str]) -> list[Record]:
        # Fetching records not directly supported; requires implementation of metadata storage
        raise NotImplementedError("fetch_batch not supported for OpenSearch")

    def delete_batch(self, request: list[str]):
        #deleting the records not directly supported; requires implementation of delete by vsb_vec_id in the metadata
        raise NotImplementedError("delete_batch not supported for OpenSearch")


class OpenSearchDB(DB):
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: str,
        name: str,
        config: dict,
    ):
        self.index_name = config["opensearch_index_name"]
        self.skip_populate = config["skip_populate"]
        self.overwrite = config["overwrite"]
        if self.index_name is None:
            # None specified, default to "vsb-<workload>"
            self.index_name = f"vsb-{name}"

        index_body = {
            "settings": {"index.knn": True},
            "mappings": {
                "properties": {
                    "vsb_vec_id": { "type": "text", "fields": { "keyword": { "type": "keyword" } } },
                    "v_content": {"type": "knn_vector", "dimension": dimensions},
                }
            }
        }
        
        if not client.indices.exists(self.index_name):
            logger.info(
                f"OpenSearchDB: Specified index '{self.index_name}' was not found, or the "
                f"specified AWS Access keys cannot access it. Creating new index '{self.index_name}'."
            )
            client.indices.create(index=self.index_name, body=index_body)
            self.created_index = True
        else:
            logger.info(
                f"OpenSearchDB: Index '{self.index_name}' already exists. Skipping index creation."
            )
            self.created_index = False


    def get_batch_size(self, sample_record: Record) -> int:
        # Similar constraints as Pinecone, OpenSearch also has limits on batch sizes
        #max_record_size = 1024 * 40  # Estimate 40KB for each record
        #max_batch_size = 500  # OpenSearch handles bulk requests in chunks
        #For now, we'll use a batch size of 100
        batch_size = 100
        return batch_size

    def get_namespace(self, namespace: str) -> Namespace:
        return OpenSearchNamespace(namespace)

    def initialize_population(self):
        if self.skip_populate:
            return
        if not self.created_index and not self.overwrite:
            msg = (
                f"OpenSearchDB: Index '{self.index_name}' already exists - cowardly "
                f"refusing to overwrite existing data. Specify --overwrite to "
                f"delete it, or specify --skip_populate to skip population phase."
            )
            logger.critical(msg)
            raise StopUser()
        try:
            logger.info(
                f"OpenSearchDB: Deleting existing index '{self.index_name}' before "
                f"population (--overwrite=True)"
            )
            client.indices.delete(index=self.index_name)
            logger.info(f"Index '{self.index_name}' cleared for population")
        except Exception as e:
            logger.critical(f"Error deleting index '{self.index_name}': {e}")
            raise StopUser()

    #def finalize_population(self, record_count: int):
    #    logger.debug(f"Waiting for index to reach record count: {record_count}")
    #    while True:
    #        stats = client.indices.stats(index=self.index_name)
    #        count = stats["_all"]["primaries"]["docs"]["count"]
    #        if count >= record_count:
    #            break
    #        time.sleep(1)

    def skip_refinalize(self):
        return False

    #def get_record_count(self) -> int:
    #    stats = client.indices.stats(index=self.index_name)
    #    return stats["_all"]["primaries"]["docs"]["count"]

