import argparse
import pathlib
import random
import time
from functools import wraps
import gevent
import json
import pandas
from google.cloud.storage import Bucket, Client, transfer_manager
import grpc.experimental.gevent as grpc_gevent
from locust import FastHttpUser, User, events, tag, task
from locust.env import Environment
from locust.exception import StopUser
from locust.runners import Runner, WorkerRunner
from locust.user.task import DefaultTaskSet, TaskSet
import logging
import numpy as np
from dotenv import load_dotenv
from pyarrow.parquet import ParquetDataset
import os
from pinecone import Pinecone
from pinecone.grpc import PineconeGRPC
import psutil
import tempfile
import tabulate
from tqdm import tqdm, trange
import sys

# patch grpc so that it uses gevent instead of asyncio. This is required to
# allow the multiple coroutines used by locust to run concurrently. Without it
# (using default asyncio) will block the whole Locust/Python process,
# in practice limiting to running a single User per worker process.
grpc_gevent.init_gevent()

load_dotenv()  # take environment variables from .env.

colornames_file = pathlib.Path(__file__).parent / "colornames.txt"
word_list = [x.strip() for x in open(colornames_file, "r")]
includeMetadataValue = True
includeValuesValue = False
apikey = os.environ['PINECONE_API_KEY']

# Configure pyarrow's embedded jemalloc library to immediately return
# deallocted memory back to the OS.
# (We only use pyarrow to load the dataset's Parquet files into
# pandas.DateFrames, which requires ~1GB RAM per 100k ~1000 dimension
# vectors; with the default setting pyarrow can hold this memory
# resident for extended periods after we have finished with the
# associated DataFrame, resulting in excessive process RSS,
# particulary with high process counts).
import pyarrow
try:
    pyarrow.jemalloc_set_decay_ms(0)
except NotImplementedError:
    # Raised if jemalloc is not supported by the pyarrow installation - skip
    pass


@events.init_command_line_parser.add_listener
def _(parser):
    pc_options = parser.add_argument_group("Pinecone-specific options")
    pc_options.add_argument("--pinecone-topk", type=int, metavar="<int>", default=10,
                            help=("Number of results to return from a Pinecone "
                                  "query() request. Defaults to 10."))
    pc_options.add_argument("--pinecone-mode", choices=["rest", "sdk", "sdk+grpc"],
                            default="sdk+grpc",
                            help="How to connect to the Pinecone index (default: %(default)s). Choices: "
                                 "'rest': Pinecone REST API (via a normal HTTP client). "
                                 "'sdk': Pinecone Python SDK ('pinecone-client'). "
                                 "'sdk+grpc': Pinecone Python SDK using gRPC as the underlying transport.")
    pc_options.add_argument("--pinecone-dataset", type=str, metavar="<dataset_name> | 'list' | 'list-details'", default=None,
                            help="The dataset to use for index population and/or query generation. "
                                 "Pass the value 'list' to list available datasets, or pass 'list-details' to"
                                 " list full details of available datasets.")
    pc_options.add_argument("--pinecone-dataset-ignore-queries", action=argparse.BooleanOptionalAction,
                            help="Ignore and do not load the 'queries' table from the specified dataset.")
    pc_options.add_argument("--pinecone-dataset-docs-sample-for-query", type=float, default=0.01,
                            metavar="<fraction> (0.0 - 1.0)",
                            help="Specify the fraction of docs which should be sampled when the documents vectorset "
                                 "is used for queries (default: %(default)s).")
    pc_options.add_argument("--pinecone-populate-index", choices=["always", "never", "if-count-mismatch"],
                            default="if-count-mismatch",
                            help="Should the index be populated with the dataset before issuing requests. Choices: "
                                 "'always': Always populate from dataset. "
                                 "'never': Never populate from dataset. "
                                 "'if-count-mismatch': Populate if the number of items in the index differs from the "
                                 "number of items in th dataset, otherwise skip population. "
                                 "(default: %(default)s).")
    pc_options.add_argument("--pinecone-dataset-cache", type=str, default=".dataset_cache",
                            help="Path to directory to cache downloaded datasets (default: %(default)s).")

    # iterations option included from locust-plugins
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        help="Run at most this number of task iterations and terminate once they have finished",
        default=0,
    )


@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    if not isinstance(environment.runner, WorkerRunner):
        # For Worker runners dataset setup is deferred until the test starts,
        # to avoid multiple processes trying to downlaod at the same time.
        setup_dataset(environment)


def setup_dataset(environment: Environment, skip_download_and_populate: bool = False):
    """
    Sets up the dataset specified via the --pinecone_dataset argument:
     - downloads it if not already present in local cache
     - reads 'documents' data (if needed for population)
     - reads 'queries' data if present.
     - populates the index with the documents data (if requested).
    The Dataset is assigned to `environment.dataset` for later use by Users
    making requests.
    """
    dataset_name = environment.parsed_options.pinecone_dataset
    if not dataset_name:
        environment.dataset = Dataset()
        return
    if dataset_name in ("list", "list-details"):
        # Print out the list of available datasets, then exit.
        print("Fetching list of available datasets for --pinecone-dataset...")
        available = Dataset.list()
        if dataset_name == "list":
            # Discard the more detailed fields before printing, however
            # copy the 'dimensions' model field from 'dense_model' into the top
            # level as that's a key piece of information (needed to create an
            # index).
            brief = []
            for a in available:
                a['dimension'] = a['dense_model']['dimension']
                summary = {key.capitalize(): a[key] for key in ['name', 'documents', 'queries', 'dimension']}
                brief.append(summary)
            available = brief
        print(tabulate.tabulate(available, headers="keys", tablefmt="simple"))
        print()
        sys.exit(1)

    logging.info(f"Loading Dataset {dataset_name} into memory for Worker {os.getpid()}...")
    environment.dataset = Dataset(dataset_name, environment.parsed_options.pinecone_dataset_cache)
    ignore_queries = environment.parsed_options.pinecone_dataset_ignore_queries
    sample_ratio = environment.parsed_options.pinecone_dataset_docs_sample_for_query
    environment.dataset.load(skip_download=skip_download_and_populate,
                             load_queries=not ignore_queries,
                             doc_sample_fraction=sample_ratio)
    populate = environment.parsed_options.pinecone_populate_index
    if not skip_download_and_populate and populate != "never":
        logging.info(
            f"Populating index {environment.host} with {len(environment.dataset.documents)} vectors from dataset '{dataset_name}'")
        environment.dataset.upsert_into_index(environment.host,
                                              skip_if_count_identical=(populate == "if-count-mismatch"))
    # We no longer need documents - if we were populating then that is
    # finished, and if not populating then we don't need documents for
    # anything else.
    environment.dataset.prune_documents()


@events.test_start.add_listener
def setup_worker_dataset(environment, **_kwargs):
    # happens only once in headless runs, but can happen multiple times in web ui-runs
    # in a distributed run, the master does not typically need any test data
    if isinstance(environment.runner, WorkerRunner):
        # Make the Dataset available for WorkerRunners (non-Worker will have
        # already setup the dataset via on_locust_init).
        #
        # We need to perform this work in a background thread (not in
        # the current gevent greenlet) as otherwise we block the
        # current greenlet (pandas data loading is not
        # gevent-friendly) and locust's master / worker heartbeating
        # thinks the worker has gone missing and can terminate it.
        pool = gevent.get_hub().threadpool
        environment.setup_dataset_greenlet = pool.apply_async(setup_dataset,
                                                              kwds={'environment':environment,
                                                                    'skip_download_and_populate':True})


@events.test_start.add_listener
def set_up_iteration_limit(environment: Environment, **kwargs):
    options = environment.parsed_options
    if options.iterations:
        runner: Runner = environment.runner
        runner.iterations_started = 0
        runner.iteration_target_reached = False
        logging.debug(f"Iteration limit set to {options.iterations}")

        def iteration_limit_wrapper(method):
            @wraps(method)
            def wrapped(self, task):
                if runner.iterations_started == options.iterations:
                    if not runner.iteration_target_reached:
                        runner.iteration_target_reached = True
                        logging.info(
                            f"Iteration limit reached ({options.iterations}), stopping Users at the start of their next task run"
                        )
                    if runner.user_count == 1:
                        logging.info("Last user stopped, quitting runner")
                        if isinstance(runner, WorkerRunner):
                            runner._send_stats()  # send a final report
                        # need to trigger this in a separate greenlet, in case test_stop handlers do something async
                        gevent.spawn_later(0.1, runner.quit)
                    raise StopUser()
                runner.iterations_started = runner.iterations_started + 1
                method(self, task)

            return wrapped

        # monkey patch TaskSets to add support for iterations limit. Not ugly at all :)
        TaskSet.execute_task = iteration_limit_wrapper(TaskSet.execute_task)
        DefaultTaskSet.execute_task = iteration_limit_wrapper(DefaultTaskSet.execute_task)


@events.test_stop.add_listener
def on_stop_mem_usage(**kwargs):
    print_mem_usage("On test stop")


def print_mem_usage(label=""):
    if label:
        label = " - " + label
    process = psutil.Process()
    info = process.memory_info()

    def mb(b):
        return f"{int(b / 1024 / 1024)}MB"

    arrow_alloc = pyarrow.total_allocated_bytes()
    logging.debug(f"Memory usage for pid:{process.pid} RSS:{mb(info.rss)} "
                  f"VSZ:{mb(info.vms)} "
                  f"pyarrow.allocated:{mb(arrow_alloc)}{label}")


class Dataset:
    """
    Represents a Dataset used as the source of documents and/or queries for
    Pinecone index operations.
    The set of datasets are taken from the Pinecone public datasets
    (https://docs.pinecone.io/docs/using-public-datasets), which reside in a
    Google Cloud Storage bucket and are downloaded on-demand on first access,
    then cached on the local machine.
    """
    gcs_bucket = "pinecone-datasets-dev"

    def __init__(self, name: str = "", cache_dir: str = ""):
        self.name = name
        self.cache = pathlib.Path(cache_dir)
        self.documents = None
        self.queries = pandas.DataFrame()

    @staticmethod
    def list():
        """
        List all available datasets on the GCS bucket.
        :return: A list of dict objects, one for each dataset.
        """
        client = Client.create_anonymous_client()
        bucket: Bucket = client.bucket(Dataset.gcs_bucket)
        metadata_blobs = bucket.list_blobs(match_glob="*/metadata.json")
        datasets = []
        for m in metadata_blobs:
            datasets.append(json.loads(m.download_as_string()))
        return datasets

    def load(self, skip_download: bool = False, load_queries: bool = True,
             doc_sample_fraction: float = 1.0):
        """
        Load the dataset, populating the 'documents' and 'queries' DataFrames.
        """
        if not skip_download:
            self._download_dataset_files()

        # Load all the parquet dataset (made up of one or more parquet files),
        # to use for documents into a pandas dataframe.
        self.documents = self._load_parquet_dataset("documents")

        # If there is an explicit 'queries' dataset, then load that and use
        # for querying, otherwise use documents directly.
        if load_queries:
            self.queries = self._load_parquet_dataset("queries")
        if self.queries.empty:
            logging.debug("Using complete documents dataset for query data")
            # Queries expect a different schema than documents.
            # Documents looks like:
            #    ["id", "values", "sparse_values", "metadata"]
            # Queries looks like:
            #    ["vector", "sparse_vector", "filter", "top_k"]
            #
            # Extract 'values' and rename to query schema (only
            # 'vector' field of queries is currently used).
            self.queries = self.documents[["values"]].copy()
            self.queries.rename(columns={"values": "vector"}, inplace=True)

            # Use a sampling of documents for queries (to avoid
            # keeping a large complete dataset in memory for each
            # worker process).
            self.queries = self.queries.sample(frac=doc_sample_fraction, random_state=1)

    def upsert_into_index(self, index_host, skip_if_count_identical: bool = False):
        """
        Upsert the datasets' documents into the specified index.
        :param index_host: Pinecone index to upsert into (must already exist)
        :param skip_if_count_identical: If true then skip upsert if the index already contains the same number of
               vectors as the dataset.
        """
        pinecone = PineconeGRPC(apikey)
        index = pinecone.Index(host=index_host)
        if skip_if_count_identical:
            if index.describe_index_stats()['total_vector_count'] == len(self.documents):
                logging.info(
                    f"Skipping upsert as index already has same number of documents as dataset ({len(self.documents)}")
                return

        upserted_count = self._upsert_from_dataframe(index)
        if upserted_count != len(self.documents):
            logging.warning(
                f"Not all records upserted successfully. Dataset count:{len(self.documents)},"
                f" upserted count:{upserted_count}")

    def prune_documents(self):
        """
        Discard the contents of self.documents once it is no longer required
        (it can consume a significant amount of memory).
        """
        del self.documents
        logging.debug(f"After pruning, 'queries' memory usage:{self.queries.memory_usage()}")

    def _download_dataset_files(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        logging.debug(f"Checking for existence of dataset '{self.name}' in dataset cache '{self.cache}'")
        client = Client.create_anonymous_client()
        bucket: Bucket = client.bucket(Dataset.gcs_bucket)
        blobs = [b for b in bucket.list_blobs(prefix=self.name + "/")]
        # Ignore directories (blobs ending in '/') as we don't explicilty need them
        # (non-empty directories will have their files downloaded
        # anyway).
        blobs = [b for b in blobs if not b.name.endswith("/")]
        logging.debug(f"Dataset consists of files:{[b.name for b in blobs]}")

        def should_download(blob):
            path = self.cache / blob.name
            if not path.exists():
                return True
            # File exists - check size, assume same size is same file.
            # (Ideally would check hash (md5), but using hashlib.md5() to
            # calculate the local MD5 does not match remove; maybe due to
            # transmission as compressed file?
            local_size = path.stat().st_size
            remote_size = blob.size
            return local_size != remote_size

        to_download = [b for b in filter(lambda b: should_download(b), blobs)]
        if not to_download:
            return
        pbar = tqdm(desc="Downloading datset",
                    total=sum([b.size for b in to_download]),
                    unit="Bytes",
                    unit_scale=True)
        for blob in to_download:
            logging.debug(f"Dataset file '{blob.name}' not found in cache - will be downloaded")
            dest_path = self.cache / blob.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(self.cache / blob.name)
            pbar.update(blob.size)

    def _load_parquet_dataset(self, kind):
        parquet_files = [f for f in (self.cache / self.name).glob(kind + '/*.parquet')]
        if not len(parquet_files):
            return pandas.DataFrame

        dataset = ParquetDataset(parquet_files)
        # Read only the columns that Pinecone SDK makes use of.
        if kind == "documents":
            columns = ["id", "values", "sparse_values", "metadata"]
            metadata_column = "metadata"
        elif kind == "queries":
            columns = ["vector", "sparse_vector",  "filter", "top_k", "blob"]
            metadata_column = "filter"
        else:
            raise ValueError(f"Unsupported kind '{kind}' - must be one of (documents, queries)")
        # Note: We to specify pandas.ArrowDtype as the types mapper to use pyarrow datatypes in the
        # resulting DataFrame. This is significant as (for reasons unknown) it allows subsequent
        # samples() of the DataFrame to be "disconnected" from the original underlying pyarrow data,
        # and hence significantly reduces memory usage when we later prune away the underlying
        # parrow data (see prune_documents).
        df = dataset.read(columns=columns).to_pandas(types_mapper=pandas.ArrowDtype)

        # And drop any columns which all values are missing - e.g. not all
        # datasets have sparse_values, but the parquet file may still have
        # the (empty) column present.
        df.dropna(axis='columns', how="all", inplace=True)

        if metadata_column in df:
            def cleanup_null_values(metadata):
                # Null metadata values are not supported, remove any key
                # will a null value.
                if not metadata:
                    return None
                return {k: v for k, v in metadata.items() if v}

            def convert_metadata_to_dict(metadata) -> dict:
                # metadata is expected to be a dictionary of key-value pairs;
                # however it may be encoded as a JSON string in which case we
                # need to convert it.
                if metadata is None:
                    return None
                if isinstance(metadata, dict):
                    return metadata
                if isinstance(metadata, str):
                    return json.loads(metadata)
                raise TypeError(f"metadata must be a string or dict (found {type(metadata)})")

            def prepare_metadata(metadata):
                return cleanup_null_values(convert_metadata_to_dict(metadata))

            df[metadata_column] = df[metadata_column].apply(prepare_metadata)
        logging.debug(f"Loaded {len(df)} vectors of kind '{kind}'")
        return df

    def _upsert_from_dataframe(self, index):
        """
        Note: using PineconeGRPC.Index.upsert_from_dataframe() directly
        results in intermittent failures against serverless indexes as
        we can hit the request limit:
            grpc._channel._MultiThreadedRendezvous: < _MultiThreadedRendezvous of RPC that terminated with:
               status = StatusCode.RESOURCE_EXHAUSTED
               details = "Too many requests. Please retry shortly"
        I haven't observed this with the HTTP Pinecone.Index, however the
        gRPC one is so much faster for bulk loads we really want to keep using
        gRPC. As such, we have our own version of upsert from dataframe which
        handles this error with backoff and retry.
        """

        # Solution is somewhat naive - simply chunk the dataframe into
        # chunks of a smaller size, and pass each chunk to upsert_from_dataframe.
        # We still end up with multiple vectors in progress at once, but we
        # limit it to a finite amount and not the entire dataset.
        def split_dataframe(df, batch_size):
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i: i + batch_size]
                yield batch

        pbar = tqdm(desc="Populating index", unit=" vectors", total=len(self.documents))
        upserted_count = 0
        for sub_frame in split_dataframe(self.documents, 10000):
            # The 'values' column in the DataFrame is a pyarrow type (list<item: double>[pyarrow])
            # as it was read using the pandas.ArrowDtype types_mapper (see _load_parquet_dataset).
            # This _can_ be automatically converted to a Python list object inside upsert_from_dataframe,
            # but it is slow, as at that level the DataFrame is iterated row-by-row and the conversion
            # happens one element at a time.
            # However, converting the entire sub-frame's column back to a Python object before calling
            # upsert_from_dataframe() is significantly faster, such that the overall upsert throughput
            # (including the actual server-side work) is around 2x greater if we pre-convert.
            converted = sub_frame.astype(dtype={'values': object})
            resp = index.upsert_from_dataframe(converted,
                                               batch_size=200,
                                               show_progress=False)
            upserted_count += resp.upserted_count
            pbar.update(len(sub_frame))
        return upserted_count


class PineconeUser(User):
    def __init__(self, environment):
        super().__init__(environment)

        # Determine the dimensions of our index
        self.pinecone = Pinecone(apikey)
        self.index = self.pinecone.Index(host=self.host)
        self.dimensions = self.index.describe_index_stats()['dimension']

        # Set test properties from command-line args
        self.top_k = environment.parsed_options.pinecone_topk
        self.mode = environment.parsed_options.pinecone_mode
        if self.mode == "rest":
            self.client = PineconeRest(self.environment)
        elif self.mode == "sdk":
            self.client = PineconeSdk(self.environment)
        elif self.mode == "sdk+grpc":
            self.client = PineconeSdk(self.environment, use_grpc=True)
        else:
            raise Exception(f"Invalid pinecone_mode {self.mode}")

        if isinstance(self.environment.runner, WorkerRunner):
            # Wait until the datset has been loaded for this environment (Runner)
            environment.setup_dataset_greenlet.join()

    @tag('query')
    @task
    def vectorQuery(self):
        self.client.query(name="Vector (Query only)",
                          q_vector=self._query_vector(), top_k=self.top_k)

    @tag('fetch')
    @task
    def fetchQuery(self):
        randId = str(random.randint(0,85794))
        self.client.fetch(randId)

    @tag('delete')
    @task
    def deleteById(self):
        randId = str(random.randint(0,85794))
        self.client.delete(randId)

    @tag('query_meta')
    @task
    def vectorMetadataQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.query(name="Vector + Metadata",
                          q_vector=self._query_vector(),
                          top_k=self.top_k,
                          q_filter={"color": metadata['color'][0]})

    @tag('query_namespace')
    @task
    def vectorNamespaceQuery(self):
        self.client.query(name="Vector + Namespace (namespace1)",
                          q_vector=self._query_vector(),
                          top_k=self.top_k,
                          namespace="namespace1")

    @tag('query_meta_namespace')
    @task
    def vectorMetadataNamespaceQuery(self):
        metadata = dict(color=random.choices(word_list))
        self.client.query(name="Vector + Metadata + Namespace (namespace1)",
                          q_vector=self._query_vector(),
                          top_k=self.top_k,
                          q_filter={"color": metadata['color'][0]},
                          namespace="namespace1")

    def _query_vector(self):
        """
        Sample a random entry from the queries set (if non-empty), otherwise
        generate a random query vector in the range [-1.0, 1.0].
        """
        if not self.environment.dataset.queries.empty:
            record = self.environment.dataset.queries.sample(n=1).iloc[0]
            return record['vector']
        return ((np.random.random_sample(self.dimensions) * 2.0) - 1.0).tolist()


class PineconeRest(FastHttpUser):
    """Pinecone REST user, responsible for making requests to Pinecone via
    the base REST API.

    This is not directly instantiated / called by the locust framework, instead
    it is called via 'PineconeUser' if --pinecone-mode=rest.
    """
    abstract = True

    def __init__(self, environment):
        self.host = environment.host
        super().__init__(environment)

    def query(self, name: str, q_vector: list, top_k: int, q_filter=None, namespace=None):
        json = {"vector": q_vector,
                "topK": top_k,
                "includeMetadata": includeMetadataValue,
                "includeValues": includeValuesValue}
        if q_filter:
            json['filter'] = q_filter
        if namespace:
            json['namespace'] = namespace
        self.client.post("/query", name=name,
                         headers={"Api-Key": apikey},
                         json=json)

    def fetch(self, id : str):
        self.client.get("/vectors/fetch?ids=" + id, name=f"Fetch",
                        headers={"Api-Key": apikey})

    def delete(self, id : str):
        self.client.post("/vectors/delete", name=f"Delete",
                        headers={"Api-Key": apikey},
                        json={"ids": [id]})


class PineconeSdk(User):
    """Pinecone SDK client user, responsible for making requests to Pinecone via
    the 'pinecone-client' Python library.

    This is not directly instantiated / called by the locust framework, instead
    it is called via 'PineconeUser' if --pinecone-mode=rest.
    """
    abstract = True

    def __init__(self, environment, use_grpc : bool = False):
        super().__init__(environment)
        self.host = environment.host

        if use_grpc:
            self.request_type="Pine gRPC"
            self.pinecone = PineconeGRPC(apikey)
        else:
            self.request_type="Pine"
            self.pinecone = Pinecone(apikey)
        self.index = self.pinecone.Index(host=self.host)

    def query(self, name: str, q_vector: list, top_k: int, q_filter=None, namespace=None):
        args = {'vector': q_vector,
                  'top_k': top_k,
                  'include_values': includeValuesValue,
                  'include_metadata': includeValuesValue}
        if q_filter:
            args['filter'] = q_filter
        if namespace:
            args['namespace'] = namespace

        start = time.time()
        result = self.index.query(**args)
        stop = time.time()

        response_time = (stop - start) * 1000.0
        match_count = len(result.matches)

        events.request.fire(request_type=self.request_type,
                            name=name,
                            response_length=match_count,
                            response_time=response_time)

    def fetch(self, id : str):
        start = time.time()
        self.index.fetch(ids=[id])
        stop = time.time()

        response_time = (stop - start) * 1000.0
        events.request.fire(request_type=self.request_type,
                            name="Fetch",
                            response_length=0,
                            response_time=response_time)

    def delete(self, id : str):
        start = time.time()
        self.index.delete(ids=[id])
        stop = time.time()

        response_time = (stop - start) * 1000.0
        events.request.fire(request_type=self.request_type,
                            name="Delete",
                            response_length=0,
                            response_time=response_time)
