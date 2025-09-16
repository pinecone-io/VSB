import logging
from locust.exception import StopUser
import vsb
from vsb import logger
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, after_log
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..base import DB, Namespace
from ...vsb_types import Record, SearchRequest, DistanceMetric, RecordList

import requests
import numpy as np
import json


# Exceptions
class SolrException(Exception): ...


class NotFoundException(SolrException): ...


class UnauthorizedException(SolrException): ...


class SolrClient:
    def __init__(
        self,
        base_url: str,
        core: str | None = None,
        *,
        skip_populate: bool = False,
        **kwargs,
    ):
        """
        base_url: 'http://localhost:8983/solr'
        core:     optional core name; base_url becomes '<base>/<core>' if provided.
        """
        self._root = base_url.rstrip("/")
        self._core = core
        self.base_url = f"{self._root}/{core}" if core else self._root
        self.skip_populate = skip_populate
        self.start_from = kwargs.get("start_from", 0)
        self.overwrite = kwargs.get("overwrite", False)
        self.max_retries = kwargs.get("max_retries", 3)
        self.resume_offset = 0
        self._ensured_fields: set[str] = set()

        # Set up persistent session
        self._sess = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
        )
        self._sess.mount("http://", HTTPAdapter(pool_maxsize=64, max_retries=retry))

    # -------------------- HTTP helpers --------------------

    def _root_get(self, path: str, **kw):
        return self._sess.get(f"{self._root}{path}", timeout=30, **kw)

    def _core_get(self, path: str, **kw):
        return self._sess.get(f"{self._root}/{self._core}{path}", timeout=30, **kw)

    def _core_post_json(
        self, path: str, payload: dict, *, params: dict | None = None, timeout: int = 30
    ):
        return self._sess.post(
            f"{self._root}/{self._core}{path}",
            params=params,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )

    def _core_admin(self, timeout: int = 30, **params):
        return self._sess.get(
            f"{self._root}/admin/cores", params=params, timeout=timeout
        )

    def _expect_ok(self, resp, where: str):
        if resp.status_code != 200:
            raise SolrException(f"{where} failed {resp.status_code}: {resp.text}")

    def _parse_dim_sim(self, attrs):
        # attrs can be dict or list of "k=v" strings
        dim = sim = None
        if isinstance(attrs, dict):
            dim = attrs.get("vectorDimension")
            sim = attrs.get("similarityFunction")
        elif isinstance(attrs, list):
            for a in attrs:
                if isinstance(a, str) and a.startswith("vectorDimension="):
                    dim = a.split("=", 1)[1]
                if isinstance(a, str) and a.startswith("similarityFunction="):
                    sim = a.split("=", 1)[1]
        return (
            int(dim) if dim is not None else None,
            str(sim) if sim is not None else None,
        )

    def _get_fieldtype(self, name: str = "knn_vector"):
        r = self._core_get(f"/schema/fieldtypes/{name}?wt=json")
        if r.status_code == 404:
            return False, None
        if r.status_code != 200:
            raise SolrException(f"GET fieldtype failed {r.status_code}: {r.text}")
        return True, r.json().get("fieldType", {})

    def _upsert_knn_fieldtype(self, dim: int, sim: str):
        exists, ft = self._get_fieldtype("knn_vector")
        if exists:
            curr_dim, curr_sim = self._parse_dim_sim(ft.get("attributes", {}))
            if curr_dim != int(dim) or (curr_sim and curr_sim != sim):
                r = self._core_post_json(
                    "/schema",
                    {
                        "replace-field-type": {
                            "name": "knn_vector",
                            "class": "solr.DenseVectorField",
                            "vectorDimension": int(dim),
                            "similarityFunction": sim,
                        }
                    },
                )
                self._expect_ok(r, "replace-field-type")
            return
        # Not present → add
        r = self._core_post_json(
            "/schema",
            {
                "add-field-type": {
                    "name": "knn_vector",
                    "class": "solr.DenseVectorField",
                    "vectorDimension": int(dim),
                    "similarityFunction": sim,
                }
            },
        )
        self._expect_ok(r, "add-field-type")

    def _set_resume_offset_from_index(self):
        """Prime resume offset from current index count."""
        try:
            self.resume_offset = self.get_record_count()
        except Exception as e:
            logger.warning(f"RTG: Error getting record count: {e}")
            self.resume_offset = 0

    def _apply_resume_skip(self, docs: list[dict]) -> list[dict]:
        """Drop the first `resume_offset` docs; decrement offset atomically."""
        if self.resume_offset <= 0 or not docs:
            return docs
        if self.resume_offset >= len(docs):
            self.resume_offset -= len(docs)
            return []
        # partial consume
        out = docs[self.resume_offset :]
        self.resume_offset = 0
        return out

    def _already_exists(self, resp) -> bool:
        return resp.status_code in (400, 500) and "already exists" in resp.text.lower()

    def _wait_for_core_open(self, deadline_s: int = 60):
        # be tolerant of just-created cores
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            try:
                r = self._core_get("/admin/ping?wt=json")
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        # fall back to luke check
        self._wait_for_core_loaded()

    def _core_exists(self) -> bool:
        r = self._core_admin(action="STATUS", core=self._core, wt="json")
        if r.status_code != 200:
            raise SolrException(f"STATUS failed {r.status_code}: {r.text}")
        st = r.json().get("status", {})
        # present when core exists; empty when it doesn't
        return bool(st.get(self._core))

    def _add_or_replace_knn(self, dim: int, sim: str):
        add_payload = {
            "add-field-type": {
                "name": "knn_vector",
                "class": "solr.DenseVectorField",
                "vectorDimension": int(dim),
                "similarityFunction": sim,
            }
        }
        r = self._core_post_json("/schema", add_payload)
        if r.status_code == 200:
            return
        # If the type already exists (e.g., polluted _default), replace it in place.
        if r.status_code in (400, 409) and "already exists" in r.text.lower():
            rep_payload = {"replace-field-type": add_payload["add-field-type"]}
            rr = self._core_post_json("/schema", rep_payload)
            self._expect_ok(rr, "replace-field-type")
        else:
            self._expect_ok(r, "add-field-type")

    def _rtg_ids(self, ids: list[str]) -> set[str]:
        """Return the subset of ids that already exist in the core (fast RTG)."""
        logger.info(f"RTG: Checking {len(ids)} ids to get existing ids")
        have = set()
        # chunk to keep URLs/bodies sane
        CHUNK = 512
        for i in range(0, len(ids), CHUNK):
            chunk = ids[i : i + CHUNK]
            # Prefer POST with JSON to avoid long query strings
            params = {("fl", "id"), ("wt", "json"), [("id", i) for i in chunk]}
            r = self._core_get("/select", params=params)
            if r.status_code != 200:
                raise SolrException(f"RTG failed {r.status_code}: {r.text}")
            docs = r.json().get("docs", []) or []
            for d in docs:
                _id = d.get("id")
                if _id is not None:
                    have.add(str(_id))
        return have

    # -------------------- Waiters --------------------

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

    def _wait_for_core_loaded(self, deadline_s: int = 60):
        t0 = time.time()
        while time.time() - t0 < deadline_s:
            try:
                r = self._core_get("/admin/luke?wt=json")
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.5)
        raise SolrException(f"Core '{self._core}' did not load (no /admin/luke)")

    # -------------------- Core ops --------------------

    def _unload_hard(self):
        return self._core_admin(
            action="UNLOAD",
            core=self._core,
            deleteIndex="true",
            deleteDataDir="true",
            deleteInstanceDir="true",
        )

    def _recreate_core(self):
        self._unload_hard()
        r = self._core_admin(
            action="CREATE", name=self._core, configSet="_default", wt="json"
        )
        self._expect_ok(r, "CREATE")
        self._wait_for_core_loaded()

    # -------------------- Schema helpers --------------------

    def _ensure_field(self, name: str, ftype: str, extra: dict | None = None):
        r = self._core_get(f"/schema/fields/{name}")
        if r.status_code == 404:
            payload = {
                "add-field": {
                    "name": name,
                    "type": ftype,
                    "stored": True,
                    "indexed": True,
                }
            }
            if name == "id":
                payload["add-field"]["required"] = True
            if extra:
                payload["add-field"].update(extra)
            self._expect_ok(
                self._core_post_json("/schema", payload), f"add-field {name}"
            )
        elif r.status_code != 200:
            raise SolrException(f"GET field {name} failed {r.status_code}: {r.text}")

    def _normalize_docs(self, docs: list[dict]) -> list[dict]:
        out = []
        for doc in docs:
            ndoc = {}
            for key, value in doc.items():
                if key == "id":
                    ndoc["id"] = str(value)
                elif key == "values":
                    ndoc["values"] = [float(x) for x in value]
                else:
                    flat = self._flatten(
                        key, value
                    )  # dict of {field_name: field_value}
                    for fname, fval in flat.items():
                        tname, is_multi = self._infer_solr_type(fval)
                        self._ensure_field_exists(fname, tname, is_multi)
                        ndoc[fname] = fval
            out.append(ndoc)
        return out

    def _infer_solr_type(self, value):
        """Return (type_name, is_multi). Types: 'pint','pfloat','boolean','string'."""
        is_multi = isinstance(value, list)
        sample = value[0] if (is_multi and value) else value

        if isinstance(sample, bool):
            return "boolean", is_multi
        # treat ints before floats to avoid ints becoming pfloat
        try:
            if isinstance(sample, (int,)) or (
                isinstance(sample, str) and sample.isdigit()
            ):
                return "pint", is_multi
            float(sample)  # raises if not numeric
            return "pfloat", is_multi
        except Exception:
            return "string", is_multi

    def _ensure_field_exists(self, field_name: str, type_name: str, multi: bool):
        if field_name in self._ensured_fields:
            return
        r = self._core_get(f"/schema/fields/{field_name}")
        if r.status_code == 200:
            self._ensured_fields.add(field_name)
            return
        if r.status_code != 404:
            raise SolrException(
                f"GET field {field_name} failed {r.status_code}: {r.text}"
            )

        payload = {
            "add-field": {
                "name": field_name,
                "type": type_name,
                "stored": True,
                "indexed": True,
                "multiValued": bool(multi),
            }
        }
        rr = self._core_post_json("/schema", payload)
        if rr.status_code != 200:
            raise SolrException(
                f"add-field {field_name} failed {rr.status_code}: {rr.text}"
            )
        self._ensured_fields.add(field_name)

    def _ensure_dynamic_fields(self):
        """Ensure typed, multiValued dynamic fields exist. One-time per core."""
        want = [
            {
                "name": "*_s",
                "type": "string",
                "indexed": True,
                "stored": True,
                "multiValued": True,
            },
            {
                "name": "*_i",
                "type": "pint",
                "indexed": True,
                "stored": True,
                "multiValued": True,
            },
            {
                "name": "*_f",
                "type": "pfloat",
                "indexed": True,
                "stored": True,
                "multiValued": True,
            },
            {
                "name": "*_b",
                "type": "boolean",
                "indexed": True,
                "stored": True,
                "multiValued": True,
            },
        ]
        r = self._core_get("/schema/dynamicfields?wt=json")
        if r.status_code != 200:
            raise SolrException(f"GET dynamicfields failed {r.status_code}: {r.text}")
        have = {df["name"] for df in r.json().get("dynamicFields", [])}

        for df in want:
            if df["name"] in have:
                continue
            resp = self._core_post_json("/schema", {"add-dynamic-field": df})
            self._expect_ok(resp, f"add-dynamic-field {df['name']}")

    def _flatten(self, prefix: str, value):
        """
        Return dict of flattened fields:
        - dict → dotted keys: a.b.c
        - list of scalars → keep as list on same key
        - list of dicts → index each item: key.0.sub, key.1.sub
        - scalar → {prefix: value}
        """
        out = {}
        if isinstance(value, dict):
            for k, v in value.items():
                out.update(self._flatten(f"{prefix}.{k}", v))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                for i, item in enumerate(value):
                    out.update(self._flatten(f"{prefix}.{i}", item))
            else:
                out[prefix] = value
        else:
            out[prefix] = value
        return out

    def _to_solr_fq(self, filters: dict) -> str:
        """Turn {"tags":[851,769], "author":"alice"} into Solr fq."""
        clauses = []
        for key, val in filters.items():

            def clause(field_base, value):
                if isinstance(value, bool):
                    field = f"{field_base}_b"
                    return f"{field}:{str(value).lower()}"
                if isinstance(value, int):
                    field = f"{field_base}_i"
                    return f"{field}:{value}"
                if isinstance(value, float):
                    field = f"{field_base}_f"
                    return f"{field}:{value}"
                # strings or fallback
                field = f"{field_base}_s"
                return f'{field}:"{value}"'

            if isinstance(val, list):
                if not val:
                    continue
                parts = [clause(key, v) for v in val]
                # AND all values by default; change to OR if that’s your contract
                clauses.append("(" + " AND ".join(parts) + ")")
            elif isinstance(val, dict):
                # flatten inline: key_subKey …
                for sub_k, sub_v in val.items():
                    base = f"{key}_{sub_k}"
                    if isinstance(sub_v, list):
                        parts = [clause(base, v) for v in sub_v]
                        clauses.append("(" + " AND ".join(parts) + ")")
                    else:
                        clauses.append(clause(base, sub_v))
            else:
                clauses.append(clause(key, val))
        return " AND ".join(clauses) if clauses else "*:*"

    def _filter_existing_ids(self, docs: list[dict]) -> list[dict]:
        ids = [str(d["id"]) for d in docs]
        # Build an id:( "a" "b" ... ) query safely via JSON body
        terms = " ".join(f'"{i}"' for i in ids)
        payload = {"query": f"id:({terms})", "fields": "id", "limit": len(ids)}
        r = self._core_post_json("/select", payload, params={"wt": "json"})
        if r.status_code != 200:
            raise SolrException(
                f"Failed to filter existing ids: {r.status_code} {r.text}"
            )
        existing = {d["id"] for d in r.json()["response"]["docs"]}
        logger.debug(f"RTG: {len(existing)} ids already exist")
        return [d for d in docs if str(d["id"]) not in existing]

    # -------------------- Public: index creation --------------------

    def create_index(
        self, name: str, dimension: int, metric: str, spec: dict | None = None
    ):
        if not self._core:
            self._core = name
        if self.skip_populate:
            return

        self._wait_for_solr()

        # 1) Must not exist
        if self._core_exists():
            raise SolrException(f"Core '{self._core}' already exists")

        # 2) Create
        cr = self._core_admin(
            action="CREATE", name=self._core, configSet="_default", wt="json"
        )
        if cr.status_code != 200:
            # If a parallel creator raced us, treat as "exists" since our contract says 'shouldn't have been called'
            if "already exists" in cr.text.lower():
                raise SolrException(f"Core '{self._core}' already exists")
            raise SolrException(f"CREATE failed {cr.status_code}: {cr.text}")

        # 3) Wait for it to be live
        self._wait_for_core_loaded()

        # 4) Vector type + fields
        sim = {
            "cosine": "cosine",
            "dot": "dot_product",
            "dot_product": "dot_product",
            "ip": "dot_product",
            "euclidean": "euclidean",
            "l2": "euclidean",
        }.get(str(metric).lower(), "cosine")

        # Add or replace field type (handles polluted _default cleanly)
        self._add_or_replace_knn(dimension, sim)
        self._ensure_dynamic_fields()

        # Fields (one vector field only)
        self._ensure_field("id", "string")
        self._ensure_field("values", "knn_vector")

        # Clean up any accidental 'vector' field if it exists
        rv = self._core_get("/schema/fields/vector")
        if rv.status_code == 200:
            self._expect_ok(
                self._core_post_json("/schema", {"delete-field": {"name": "vector"}}),
                "delete-field vector",
            )

        # Point client at this core
        self.base_url = f"{self._root}/{self._core}"

    # -------------------- Data ops --------------------

    def core_exists(self, core: str) -> bool:
        r = self._core_admin(action="STATUS", core=core, wt="json")
        if r.status_code != 200:
            return False
        st = r.json().get("status", {})
        return core in st and st[core].get("name") == core

    def commit(self):
        r = self._core_get("/update?commit=true")
        if r.status_code != 200:
            raise SolrException(f"Failed to commit changes: {r.text}")

    def add(self, documents, start_from: int = 0, overwrite: bool = False):
        if not documents:
            return
        # Get the list of ids that already exist
        # existing_ids = self._rtg_ids([str(d["id"]) for d in documents])
        # logger.debug(f"RTG: {len(existing_ids)} ids already exist")
        # Filter out the documents that already exist
        # docs = [d for d in documents if str(d["id"]) not in existing_ids]
        documents = self._filter_existing_ids(documents)
        if not documents:
            return
        if len(documents) == 0:
            return
        if overwrite:
            self.delete_all()
        logger.debug(f"RTG: {len(documents)} ids to add")

        # Normalize payload for Solr
        docs = self._normalize_docs(documents)

        # Retry logic for connection errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                r = self._core_post_json(
                    "/update",
                    {"add": docs},
                    params={"commitWithin": 60000, "wt": "json"},
                    timeout=(10, 300),
                )
                if r.status_code != 200:
                    raise SolrException(
                        f"Failed to add documents: {r.status_code} {r.text}"
                    )
                return
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(300)
                else:
                    raise SolrException(
                        "Failed to add documents after multiple attempts due to connection errors."
                    )
            except Exception as e:
                logger.warning(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(300)
                else:
                    raise SolrException(
                        "Failed to add documents after multiple attempts due to errors."
                    )

    def search(self, q, rows=10, fq=None, ef=200):
        vector_query = ",".join(map(str, q))
        params = {
            "q": f"{{!knn f=values topK={rows}}}[{vector_query}]",
            "rows": str(rows),
            "fl": "id,score",
            "ef": ef,
            "wt": "json",
        }
        if fq:
            params["fq"] = self._to_solr_fq(fq)
        r = self._core_get("/select", params=params)
        if r.status_code == 404:
            raise NotFoundException(f"Document not found: {r.url} {r.text}")
        if r.status_code == 401:
            raise UnauthorizedException(f"Unauthorized access: {r.text}")
        if r.status_code != 200:
            raise SolrException(f"Search failed: {r.status_code} {r.text}")

        return r.json()

    def delete(self, ids):
        r = self._core_post_json("/update", {"delete": {"id": ids}})
        if r.status_code != 200:
            raise SolrException(f"Failed to delete documents: {r.text}")

    def delete_all(self):
        r = self._core_post_json("/update", {"delete": {"query": "*:*"}})
        if r.status_code != 200:
            raise SolrException(f"Failed to delete all documents: {r.text}")

    def delete_index(self):
        r = self._core_admin(
            action="UNLOAD", core=self._core, deleteIndex="true", deleteDataDir="true"
        )
        if r.status_code != 200:
            raise SolrException(f"Failed to delete index: {r.text}")

    def get_record_count(self):
        r = self._core_get("/select", params={"q": "*:*", "rows": 0})
        if r.status_code != 200:
            raise SolrException(f"Failed to get record count: {r.text}")
        return r.json()["response"]["numFound"]

    def close(self): ...


class SolrNamespace(Namespace):
    def __init__(
        self,
        client: SolrClient,
        namespace: str,
        skip_populate: bool = False,
        start_from: int = 0,
        overwrite: bool = False,
    ):
        self.client = client
        self.skip_populate = skip_populate
        self.start_from = start_from
        self.overwrite = overwrite

    def insert_batch(self, batch: RecordList):
        if self.skip_populate:
            return
        dicts = [dict(rec) for rec in batch]
        self.client.add(dicts, self.start_from, self.overwrite)

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
        return [m["id"] for m in result["response"]["docs"]]

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
        self.index_name = config["solr_index_name"] or f"vsb-{name}"

        # FIX: pass args by name so skip_populate isn't mis-slotted
        self.client = SolrClient(
            config["solr_url"],
            core=self.index_name,
            skip_populate=self.skip_populate,
            start_from=config["start_from"],
            overwrite=config["overwrite"],
            max_retries=config["solr_max_retries"],
            retry_delay=config["solr_retry_delay"],
        )

        self.overwrite = config["overwrite"]
        spec = config["solr_index_config"]

        exists = self.client.core_exists(self.index_name)

        if not self.skip_populate:
            if exists and self.overwrite:
                logger.info(
                    f"SolrDB: Overwrite=True → dropping core '{self.index_name}'"
                )
                self.client.drop_core(self.index_name)
                self.client.create_index(
                    name=self.index_name,
                    dimension=dimensions,
                    metric=metric.value,
                    spec=spec,
                )
                self.created_index = True
            elif not exists:
                logger.info(f"SolrDB: Creating core '{self.index_name}'")
                self.client.create_index(
                    name=self.index_name,
                    dimension=dimensions,
                    metric=metric.value,
                    spec=spec,
                )
                self.created_index = True
            else:
                logger.info(
                    f"SolrDB: Core '{self.index_name}' exists → resume (no recreate)"
                )
                self.created_index = False

    def close(self):
        self.client.close()

    def get_batch_size(self, sample_record: Record) -> int:
        max_id = 512
        max_values = len(sample_record.values) * 4
        max_metadata = 40 * 1024 if sample_record.metadata else 0
        max_record_size = max_id + max_metadata + max_values
        max_namespace = 500
        size_based_batch_size = ((2 * 1024 * 1024) - max_namespace) // max_record_size
        max_batch_size = 100
        batch_size = min(size_based_batch_size, max_batch_size)
        logger.debug(f"SolrDB.get_batch_size() - Using batch size of {batch_size}")
        return batch_size

    def get_namespace(self, namespace: str) -> Namespace:
        return SolrNamespace(self.client, namespace, self.skip_populate)

    def initialize_population(self):
        if self.skip_populate:
            return
        if self.overwrite:
            logger.info(f"SolrDB: Clearing '{self.index_name}' before load")
            try:
                self.client.delete_all()
            except SolrException:
                pass
        else:
            logger.info(f"SolrDB: Resume mode → keeping existing docs")

    def finalize_population(self, record_count: int):
        if self.skip_populate:
            return
        logger.debug(f"SolrDB: Waiting for record count to reach {record_count}")
        with vsb.logging.progress_task(
            "  Finalize population",
            "  \x1b[32m\x1b[1m\x1b[4m\x1b[0m Finalize population",
            total=record_count,
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
        self.client.commit()

    def skip_refinalize(self):
        return self.skip_populate

    def get_record_count(self) -> int:
        return self.client.get_record_count()
