import json
import logging
import os
import requests

from locust.exception import StopUser

import vsb
from vsb import logger
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
grpc_gevent.init_gevent()


def _create_index_with_dedicated_read_nodes(
    api_key: str,
    index_name: str,
    dimension: int,
    metric: str,
    spec: dict,
    node_type: str,
    shards: int,
    replicas: int,
    api_version: str = "2025-10",
):
    """Create a Pinecone index with dedicated read nodes using the REST API.

    Args:
        api_key: Pinecone API key
        index_name: Name of the index to create
        dimension: Vector dimension
        metric: Distance metric (cosine, euclidean, or dotproduct)
        spec: Base index spec (should contain serverless config)
        node_type: Node type for dedicated read nodes (e.g., b1, b2)
        shards: Number of shards for dedicated read nodes
        replicas: Number of replicas for dedicated read nodes
        api_version: Pinecone API version that supports dedicated read nodes
    """
    # Ensure spec contains serverless config
    if "serverless" not in spec:
        raise ValueError(
            "Dedicated read nodes are only supported for serverless indexes. "
            "Spec must contain 'serverless' configuration."
        )

    # Build the spec with dedicated read capacity
    serverless_config = spec["serverless"].copy()
    serverless_config["read_capacity"] = {
        "mode": "Dedicated",
        "dedicated": {
            "node_type": node_type,
            "scaling": "Manual",
            "manual": {"shards": shards, "replicas": replicas},
        },
    }

    body = {
        "name": index_name,
        "dimension": dimension,
        "metric": metric,
        "vector_type": "dense",
        "deletion_protection": "disabled",
        "tags": {},
        "spec": {"serverless": serverless_config},
    }

    headers = {
        "Api-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Pinecone-API-Version": api_version,
    }

    # Add additional headers from environment variable if present
    additional_headers_json = os.environ.get("PINECONE_ADDITIONAL_HEADERS")
    if additional_headers_json:
        try:
            additional_headers = json.loads(additional_headers_json)
            headers.update(additional_headers)
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse PINECONE_ADDITIONAL_HEADERS: {additional_headers_json}"
            )

    # Use controller host from environment or default to production
    controller_host = os.environ.get(
        "PINECONE_CONTROLLER_HOST", "https://api.pinecone.io"
    )
    api_url = f"{controller_host}/indexes"

    resp = requests.post(api_url, json=body, headers=headers)

    if resp.status_code not in (200, 201):
        raise RuntimeError(
            f"Error creating index with dedicated read nodes: {resp.status_code} {resp.text}"
        )

    logger.info(
        f"PineconeDB: Created index '{index_name}' with dedicated read nodes "
        f"(node_type={node_type}, shards={shards}, replicas={replicas})"
    )
    return resp.json()


class PineconeNamespace(Namespace):
    def __init__(self, index: GRPCIndex, namespace: str):
        # TODO: Support multiple namespaces
        self.index = index
        self.namespace = namespace

    def insert_batch(self, batch: RecordList):
        # Pinecone expects a list of dicts (or tuples).
        dicts = [dict(rec) for rec in batch]
        self.index.upsert(vectors=dicts, namespace=self.namespace)

    def update_batch(self, batch: list[Record]):
        # Pinecone treats insert and update as the same operation.
        self.insert_batch(batch)

    def search(self, request: SearchRequest) -> list[str]:
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_query_with_retry():
            return self.index.query(
                vector=request.values,
                top_k=request.top_k,
                filter=request.filter,
                namespace=self.namespace,
            )

        result = do_query_with_retry()
        matches = [m["id"] for m in result["matches"]]
        return matches

    def fetch_batch(self, request: list[str]) -> list[Record]:
        return self.index.fetch(ids=request, namespace=self.namespace).vectors.values

    def delete_batch(self, request: list[str]):
        self.index.delete(ids=request, namespace=self.namespace)


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
        self.api_key = config["pinecone_api_key"]
        self.skip_populate = config["skip_populate"]
        self.overwrite = config["overwrite"]
        self.index_name = config["pinecone_index_name"]
        self.namespace = config["pinecone_namespace_name"]
        self.multi_namespace = config.get("pinecone_multi_namespace", False)
        self.use_dedicated_read_nodes = config.get(
            "pinecone_dedicated_read_nodes", False
        )
        self.dedicated_node_type = config.get("pinecone_dedicated_node_type", "b1")
        self.dedicated_shards = config.get("pinecone_dedicated_shards", 1)
        self.dedicated_replicas = config.get("pinecone_dedicated_replicas", 1)

        if self.index_name is None:
            # None specified, default to "vsb-<workload>"
            self.index_name = f"vsb-{name}"
        spec = config["pinecone_index_spec"]
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
            # Check if multi-namespace mode is enabled - if so, don't create index
            if self.multi_namespace:
                logger.critical(
                    f"PineconeDB: Index '{self.index_name}' does not exist. Multi-namespace mode requires an existing populated index. Please create the index first or use single-namespace mode."
                )
                raise StopUser()

            logger.info(
                f"PineconeDB: Specified index '{self.index_name}' was not found, or the "
                f"specified API key cannot access it. Creating new index '{self.index_name}'."
            )

            # Use REST API if dedicated read nodes are enabled
            if self.use_dedicated_read_nodes:
                _create_index_with_dedicated_read_nodes(
                    api_key=self.api_key,
                    index_name=self.index_name,
                    dimension=dimensions,
                    metric=metric.value,
                    spec=spec,
                    node_type=self.dedicated_node_type,
                    shards=self.dedicated_shards,
                    replicas=self.dedicated_replicas,
                )
            else:
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

        # Initialize namespaces list (empty for both modes to avoid AttributeError)
        self.namespaces = []

        # Multi-namespace mode: discover and validate namespaces
        if self.multi_namespace:
            if self.created_index:
                logger.critical(
                    f"PineconeDB: Cannot use multi_namespace mode with a newly created index. Index '{self.index_name}' must already exist and be populated."
                )
                raise StopUser()

            self.namespaces = self._discover_all_namespaces()
            if not self.namespaces:
                logger.critical(
                    f"PineconeDB: No populated namespaces found in index '{self.index_name}'. Multi-namespace mode requires at least one namespace with records."
                )
                raise StopUser()

            logger.info(
                f"PineconeDB: Discovered {len(self.namespaces)} namespaces: {', '.join(self.namespaces)}"
            )

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
        if self.multi_namespace:
            # Validate namespace exists in discovered namespaces
            if namespace not in self.namespaces:
                raise ValueError(
                    f"PineconeDB: Namespace '{namespace}' not found in discovered namespaces. Available namespaces: {', '.join(self.namespaces)}"
                )
            return PineconeNamespace(self.index, namespace)
        else:
            # Single namespace mode: ignore parameter, use configured namespace
            return PineconeNamespace(self.index, self.namespace)

    def initialize_population(self):
        # If the namespace already existed before VSB (we didn't create it) and
        # user didn't specify skip_populate; require --overwrite before
        # deleting the existing namespace.
        if self.skip_populate:
            return

        # Check if the namespace exists
        self.namespace_exists = self.check_namespace_exists(self.namespace)

        # If the namespace exists and user didn't specify --overwrite, raise an error
        if self.namespace_exists and not self.overwrite:
            msg = (
                f"PineconeDB: Namespace '{self.namespace}' already exists - cowardly "
                f"refusing to overwrite existing data. Specify --overwrite to "
                f"delete it, or specify --skip_populate to skip population phase."
            )
            logger.critical(msg)
            raise StopUser()

        # If the namespace exists and user specified --overwrite, delete the namespace
        if self.namespace_exists and self.overwrite:
            try:
                logger.info(
                    f"PineconeDB: Deleting existing namespace '{self.namespace}' before "
                    f"population (--overwrite=True)"
                )
                self.index.delete_namespace(namespace=self.namespace)
            except PineconeException as e:
                # Serverless indexes can throw a "Namespace not found" exception for
                # delete_namespace if there are no documents in the index. Simply ignore,
                # as the post-condition is the same.
                pass

    def finalize_population(self, record_count: int):
        """Wait until all records are visible in the namespace"""
        logger.debug(f"PineconeDB: Waiting for record count to reach {record_count}")
        with vsb.logging.progress_task(
            "  Finalize population", "  ✔ Finalize population", total=record_count
        ) as finalize_id:
            while True:
                namespace_rec_count = self.get_record_count()
                if vsb.progress:
                    vsb.progress.update(finalize_id, completed=namespace_rec_count)
                if namespace_rec_count >= record_count:
                    logger.debug(
                        f"PineconeDB: Namespace vector count reached {namespace_rec_count}, "
                        f"finalize is complete"
                    )
                    break
                time.sleep(1)

    def skip_refinalize(self):
        return False

    def get_record_count(self) -> int:
        return int(
            self.index.describe_namespace(namespace=self.namespace)["record_count"]
        )

    def _discover_all_namespaces(self) -> list[str]:
        """Discover all populated namespaces in the index."""
        populated_namespaces = []
        try:
            # list_namespaces returns a generator of dicts with 'name' and 'record_count'
            for ns in self.index.list_namespaces():
                # Convert record_count to int (API may return string)
                record_count = int(ns["record_count"])
                if record_count > 0:
                    populated_namespaces.append(ns["name"])
                    logger.debug(
                        f"PineconeDB: Namespace '{ns['name']}' has {record_count} records"
                    )
        except PineconeException as e:
            logger.error(
                f"PineconeDB: Error listing namespaces in index '{self.index_name}': {e}"
            )
            raise ValueError(
                f"Failed to list namespaces in index '{self.index_name}': {e}"
            ) from e

        if not populated_namespaces:
            raise ValueError(
                f"No populated namespaces found in index '{self.index_name}'. Multi-namespace mode requires at least one namespace with records."
            )

        return sorted(populated_namespaces)

    def _get_namespaces_for_user(self, user_id: int, num_users: int) -> list[str]:
        """Distribute namespaces across users using round-robin algorithm."""
        if num_users > len(self.namespaces):
            raise ValueError(
                f"Cannot distribute {num_users} users across {len(self.namespaces)} namespaces. Number of users must be <= number of namespaces."
            )
        return [
            self.namespaces[i] for i in range(user_id, len(self.namespaces), num_users)
        ]

    def check_namespace_exists(self, namespace: str) -> bool:
        """Check if a namespace exists inside the current index using list_namespaces generator."""
        try:
            # list_namespaces returns a generator of dicts with 'name' and 'record_count'
            for ns in self.index.list_namespaces():
                if ns["name"] == namespace:
                    logger.info(
                        f"PineconeDB: Namespace '{namespace}' exists in index '{self.index_name}'."
                    )
                    return True

            logger.info(
                f"PineconeDB: Namespace '{namespace}' does not exist in index '{self.index_name}'."
            )
            return False

        except PineconeException as e:
            logger.error(
                f"PineconeDB: Error while listing namespaces in index '{self.index_name}' - {e}"
            )
            return False
