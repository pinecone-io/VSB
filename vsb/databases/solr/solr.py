import logging
from re import I

from locust.exception import StopUser

import vsb
from vsb import logger
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, after_log
import time

from ..base import DB, Namespace
from ...vsb_types import Record, SearchRequest, DistanceMetric, RecordList

import requests
import numpy as np

# Define custom exceptions for Solr
class SolrException(Exception):
    pass

class NotFoundException(SolrException):
    pass

class UnauthorizedException(SolrException):
    pass


class SolrClient:
    def __init__(self, base_url: str, core: str | None = None, index_name: str | None = None, skip_populate: bool = False):
        """
        base_url: 'http://localhost:8983/solr'
        core:     optional core name. If provided, base_url becomes '<base>/<core>'
        """
        self._root = base_url.rstrip('/')           # http://localhost:8983/solr
        self._core = core
        self.index_name = index_name
        self.base_url = f"{self._root}/{core}" if core else self._root
        self.skip_populate = skip_populate

    # --- helpers -------------------------------------------------------------

    def _root_get(self, path: str, **kw):
        return requests.get(f"{self._root}{path}", timeout=30, **kw)

    def _core_get(self, core: str, path: str, **kw):
        return requests.get(f"{self._root}/{core}{path}", timeout=30, **kw)

    def _core_post_json(self, core: str, path: str, payload: dict):
        return requests.post(
            f"{self._root}/{core}{path}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

    def _wait_for_solr(self, deadline_s: int = 60):
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            try:
                r = self._root_get("/admin/info/system?wt=json")
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise SolrException("Solr did not become ready at /admin/info/system")

    def _wait_for_core_loaded(self, core: str, deadline_s: int = 60):
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            try:
                r = self._core_get(core, "/admin/luke?wt=json")
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise SolrException(f"Core '{core}' did not load (no /admin/luke)")

    # --- public API used by your DB wrapper ---------------------------------

    def create_index(self, name: str, dimension: int, metric: str, spec: dict | None = None):
        """
        Idempotent:
        - If core doesn't exist, create it from the built-in `_default` configset.
        - Wait for core to load.
        - Ensure DenseVector fieldType + fields exist.
        - If an existing knn_vector has a different dimension, raise (use a different core).
        """
        core = name
        if self.skip_populate:
            return
        self._wait_for_solr()

        # 1) Create core (idempotent)
        # Requires `_default` to be present under $SOLR_HOME/configsets.
        # If you see 400 "Could not load configuration ... /var/solr/data/configsets/_default",
        # seed it once inside the container:
        #   docker exec -it solr bash -lc 'mkdir -p /var/solr/data/configsets &&
        #     cp -a /opt/solr/server/solr/configsets/_default /var/solr/data/configsets/_default'
        created = False
        status = self._root_get(f"/admin/cores?action=STATUS&core={core}&wt=json")
        # STATUS is 200 even if missing, so try CREATE and tolerate "already exists".
        create = self._root_get(f"/admin/cores?action=CREATE&name={core}&configSet=_default&wt=json")
        if create.status_code == 200:
            created = True
        else:
            body = create.text
            if create.status_code == 400 and "already exists" in body:
                pass  # ok
            elif create.status_code == 400 and "Could not load configuration" in body:
                raise SolrException(
                    "CREATE failed: _default configset not found under $SOLR_HOME/configsets. "
                    "Seed it once inside the container (see comment in create_index())."
                )
            elif create.status_code not in (200, 400):
                raise SolrException(f"CREATE failed {create.status_code}: {body}")

        # 2) Wait for core to be live
        self._wait_for_core_loaded(core)

        # 3) Schema: ensure fieldType + fields
        metric_map = {
            "cosine": "cosine",
            "dot": "dot_product",
            "dot_product": "dot_product",
            "ip": "dot_product",
            "euclidean": "euclidean",
            "l2": "euclidean",
        }
        sim = metric_map.get(str(metric).lower(), "cosine")

        # fieldType
        ft = self._core_get(core, "/schema/fieldtypes/knn_vector")
        if ft.status_code == 404:
            r = self._core_post_json(
                core,
                "/schema",
                {
                    "add-field-type": {
                        "name": "knn_vector",
                        "class": "solr.DenseVectorField",
                        "vectorDimension": int(dimension),
                        "similarityFunction": sim,
                    }
                },
            )
            if r.status_code != 200:
                raise SolrException(f"add-field-type failed {r.status_code}: {r.text}")
        elif ft.status_code == 200:
            # sanity: dimension must match
            try:
                curr = ft.json()["fieldType"]["attributes"]["vectorDimension"]
                if int(curr) != int(dimension):
                    raise SolrException(
                        f"Core '{core}' already has knn_vector dim={curr}, requested dim={dimension}. "
                        "Use a different core per dimension."
                    )
            except Exception:
                pass
        else:
            raise SolrException(f"GET fieldtype failed {ft.status_code}: {ft.text}")

        def ensure_field(name, ftype, extra=None):
            r = self._core_get(core, f"/schema/fields/{name}")
            if r.status_code == 404:
                payload = {"add-field": {"name": name, "type": ftype, "stored": True, "indexed": True}}
                if name == "id":
                    payload["add-field"]["required"] = True
                if extra:
                    payload["add-field"].update(extra)
                rr = self._core_post_json(core, "/schema", payload)
                if rr.status_code != 200:
                    raise SolrException(f"add-field {name} failed {rr.status_code}: {rr.text}")
            elif r.status_code != 200:
                raise SolrException(f"GET field {name} failed {r.status_code}: {r.text}")

        ensure_field("id", "string")
        ensure_field("metadata", "text_general")
        ensure_field("values", "knn_vector")

        # 4) Point this client at the core going forward
        self._core = core
        self.base_url = f"{self._root}/{core}"

    # --- existing methods (unchanged) ---------------------------------------

    def commit(self):
        r = requests.get(f"{self.base_url}/update?commit=true", timeout=30)
        if r.status_code != 200:
            raise SolrException(f"Failed to commit changes: {r.text}")

    def add(self, documents):
        documents = [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in doc.items()}
            for doc in documents
        ]
        r = requests.post(f"{self.base_url}/update", json=documents, timeout=60)
        if r.status_code != 200:
            raise SolrException(f"Failed to add documents: {r.status_code} {r.text}")
        self.commit()

    def search(self, q, rows=10, fq=None, ef=2000):
        vector_query = ",".join(map(str, q))
        params = {
            "q": f"{{!knn f=values topK={rows}}}[{vector_query}]",
            "rows": str(rows),
            "fl": "id,score,metadata",
            "ef": ef,
        }
        print(params)
        if fq:
            params["fq"] = fq
        r = requests.get(f"{self.base_url}/select", params=params, timeout=30)
        if r.status_code == 404:
            raise NotFoundException(f"Document not found: {self.base_url}/select {r.text}")
        if r.status_code == 401:
            raise UnauthorizedException(f"Unauthorized access: {r.text}")
        if r.status_code != 200:
            raise SolrException(f"Search failed: {r.status_code} {r.text}")
        data = r.json()
        return data

    def delete(self, ids):
        r = requests.post(f"{self.base_url}/update", json={"delete": {"id": ids}}, timeout=30)
        if r.status_code != 200:
            raise SolrException(f"Failed to delete documents: {r.text}")

    def delete_all(self):
        r = requests.post(f"{self.base_url}/update", json={"delete": {"query": "*:*"}}, timeout=30)
        if r.status_code != 200:
            raise SolrException(f"Failed to delete all documents: {r.text}")

    def get_record_count(self):
        r = requests.get(f"{self.base_url}/select", params={"q": "*:*", "rows": 0}, timeout=30)
        if r.status_code != 200:
            raise SolrException(f"Failed to get record count: {r.text}")
        return r.json()["response"]["numFound"]

    def close(self): ...


class SolrNamespace(Namespace):
    def __init__(self, client: SolrClient, namespace: str, skip_populate: bool):
        self.client = client

    def insert_batch(self, batch: RecordList):
        dicts = [dict(rec) for rec in batch]
        if not self.skip_populate:
            self.client.add(dicts)

    def update_batch(self, batch: list[Record]):
        self.insert_batch(batch)

    def search(self, request: SearchRequest) -> list[str]:
        @retry(
            wait=wait_exponential_jitter(initial=0.1, jitter=0.1),
            stop=stop_after_attempt(5),
            after=after_log(logger, logging.DEBUG),
        )
        def do_query_with_retry():
            return self.client.search(
                q=request.values, rows=request.top_k, fq=request.filter
            )

        result = do_query_with_retry()
        matches = [m["id"] for m in result["response"]["docs"]]
        return matches

    def fetch_batch(self, request: list[str]) -> list[Record]:
        return self.client.get(request)

    def delete_batch(self, request: list[str]):
        self.client.delete(request)


class SolrDB(DB):
    def __init__(
        self,
        record_count: int,
        dimensions: int,
        metric: DistanceMetric,
        name: str,
        config: dict,
    ):
        self.skip_populate = config["skip_populate"]
        self.index_name = config["solr_index_name"]
        if self.index_name is None:
            self.index_name = f"vsb-{name}"
        self.client = SolrClient(config["solr_url"], self.index_name, self.skip_populate)
        self.overwrite = config["overwrite"]
        spec = config["solr_index_config"]
        if not self.skip_populate:
            try:
                self.client.create_index(
                    name=self.index_name,
                    dimension=dimensions,
                    metric=metric.value,
                    spec=spec,
                )
                self.created_index = True
            except UnauthorizedException:
                logger.critical(
                    f"SolrDB: UnauthorizedException when attempting to connect to index '{self.index_name}'"
                )
                raise StopUser()
            except NotFoundException:
                logger.info(
                    f"SolrDB: Specified index '{self.index_name}' was not found. Creating new index."
                )
                self.client.create_index(
                    name=self.index_name,
                    dimension=dimensions,
                    metric=metric.value,
                    spec=spec,
                )
                self.created_index = True
            except UnauthorizedException:
                logger.critical(
                    f"SolrDB: UnauthorizedException when attempting to connect to index '{self.index_name}'"
                )
                raise StopUser()
            except NotFoundException:
                logger.info(
                    f"SolrDB: Specified index '{self.index_name}' was not found. Creating new index."
                )
                self.client.create_index(
                    name=self.index_name,
                    dimension=dimensions,
                    metric=metric.value,
                    spec=spec,
                )
                self.created_index = True

    def close(self):
        self.client.close()

    def get_batch_size(self, sample_record: Record) -> int:
        max_id = 512
        max_values = len(sample_record.values) * 4
        max_metadata = 40 * 1024 if sample_record.metadata else 0
        max_record_size = max_id + max_metadata + max_values
        max_namespace = 500
        size_based_batch_size = ((2 * 1024 * 1024) - max_namespace) // max_record_size
        max_batch_size = 1000
        batch_size = min(size_based_batch_size, max_batch_size)
        logger.debug(f"SolrDB.get_batch_size() - Using batch size of {batch_size}")
        return batch_size

    def get_namespace(self, namespace: str) -> Namespace:
        return SolrNamespace(self.client, namespace, self.skip_populate)

    def initialize_population(self):
        if self.skip_populate:
            return
        if not self.created_index and not self.overwrite:
            msg = (
                f"SolrDB: Index '{self.index_name}' already exists - refusing to overwrite existing data."
            )
            logger.critical(msg)
            raise StopUser()
        try:
            logger.info(
                f"SolrDB: Deleting existing index '{self.index_name}' before population (--overwrite=True)"
            )
            self.client.delete_all()
        except SolrException as e:
            pass

    def finalize_population(self, record_count: int):
        if self.skip_populate:
            return
        logger.debug(f"SolrDB: Waiting for record count to reach {record_count}")
        with vsb.logging.progress_task(
            "  Finalize population", "  [32m[1m[4m[0m Finalize population", total=record_count
        ) as finalize_id:
            while True:
                index_count = self.client.get_record_count()
                if vsb.progress:
                    vsb.progress.update(finalize_id, completed=index_count)
                if index_count >= record_count:
                    logger.debug(
                        f"SolrDB: Index record count reached {index_count}, finalize is complete"
                    )
                    break
                time.sleep(1)

    def skip_refinalize(self):
        return self.skip_populate

    def get_record_count(self) -> int:
        return self.client.get_record_count()
