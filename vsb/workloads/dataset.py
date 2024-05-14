from collections.abc import Iterator

import numpy
import pyarrow.parquet
from google.cloud.storage import Bucket, Client, transfer_manager
import json
import logging
import pandas
import pathlib
from pinecone.grpc import PineconeGRPC
import pyarrow.dataset as ds
from pyarrow.parquet import ParquetDataset


class Dataset:
    """
    Represents a Dataset used as the source of documents and/or queries for
    Vector Search operations.
    The set of datasets are taken from the Pinecone public datasets
    (https://docs.pinecone.io/docs/using-public-datasets), which reside in a
    Google Cloud Storage bucket and are downloaded on-demand on first access,
    then cached on the local machine.
    """

    gcs_bucket = "pinecone-datasets-dev"

    @staticmethod
    def split_dataframe(df: pandas.DataFrame, batch_size) -> Iterator[pandas.DataFrame]:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            yield batch

    @staticmethod
    def recall(actual_matches: list, expected_matches: list):
        # Recall@K : how many relevant items were returned against how many
        # relevant items exist in the entire dataset. Defined as:
        #     truePositives / (truePositives + falseNegatives)

        # Handle degenerate case of zero matches.
        if not actual_matches:
            return 0

        # To allow us to calculate Recall when the count of actual_matches from
        # the query differs from expected_matches (e.g. when Query is
        # executed with a top_k different to what the Dataset was built with),
        # limit denominator to the minimum of the expected & actual.
        # (This allows use to use a Dataset with say 100 exact nearest
        # neighbours and still test the quality of results when querying at
        # top_k==10 as-if only the 10 exact nearest neighbours had been
        # provided).
        relevant_size = min(len(actual_matches), len(expected_matches))
        expected_matches = expected_matches[:relevant_size]
        true_positives = len(set(expected_matches).intersection(set(actual_matches)))
        recall = true_positives / relevant_size
        return recall

    def __init__(self, name: str = "", cache_dir: str = "", limit: int = 0):
        self.name = name
        self.cache = pathlib.Path(cache_dir)
        self.limit = limit
        self.documents = pandas.DataFrame()
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
            datasets.append(json.loads(m.download_as_bytes()))
        return datasets

    def load_documents(self, skip_download: bool = False):
        """
        Load the dataset, populating the 'documents' and 'queries' DataFrames.
        """
        if not skip_download:
            self._download_dataset_files()

        # Load the parquet dataset (made up of one or more parquet files),
        # to use for documents into a pandas dataframe.
        self.documents = self._load_parquet_dataset("passages", limit=self.limit)

    def get_batch_iterator(
        self, num_chunks: int, chunk_id: int, batch_size: int
    ) -> Iterator[pyarrow.RecordBatch]:
        """Split the dataset's documents into num_chunks of approximately the
        same size, returning an Iterator over the Nth chunk which yields
        batches of Records of at most batch_size.

        :param num_chunks: Number of chunks to split the dataset into.
        :param chunk_id: Which chunk to return an interator for.
        :param batch_size: Preferred size of each batch returned.
        """
        # If we are working with a complete dataset then we partition based
        # on the set of files which make up the dataset, returning an iterator
        # over the given chunk using pyarrow's Dataset.to_batches() API - this
        # is memory-efficient as it only has to load each file / row group into
        # memory at a time.
        # If the dataset is not complete (has been limited to N rows), then we
        # cannot use Dataset.to_batches() as it has no direct way to limit to N
        # rows. Given specifying a limit is normally used for a significantly
        # reduced subset of the dataset (e.g. 'test' variant) and hence memory
        # usage _should_ be low, we implement in a different manner - read the
        # first N rows into DataFrame, then split / iterate the dataframe.
        assert chunk_id >= 0
        assert chunk_id < num_chunks
        pq_files = list((self.cache / self.name).glob("passages/*.parquet"))
        if self.limit:
            first_n = ds.dataset(pq_files).head(self.limit)
            # Calculate start / end for this chunk, then split the table
            # and create an iterator over it.
            quotient, remainder = divmod(self.limit, num_chunks)
            chunks = [quotient + (1 if r < remainder else 0) for r in range(num_chunks)]
            # Determine start position based on sum of size of all chunks prior
            # to ours.
            start = sum(chunks[:chunk_id])
            user_chunk = first_n.slice(offset=start, length=chunks[chunk_id])

            def table_to_batches(table) -> Iterator[pyarrow.RecordBatch]:
                for batch in table.to_batches(batch_size):
                    yield batch

            return table_to_batches(user_chunk)
        else:
            # Need split the parquet files into `num_users` subset of files,
            # then return an iterator over the `user_id`th subset.
            self._download_dataset_files()
            chunks = numpy.array_split(pq_files, num_chunks)
            my_chunks = list(chunks[chunk_id])
            if not my_chunks:
                # No chunks for this user - nothing to do.
                return []
            docs_pq_dataset = ds.dataset(my_chunks)
            return docs_pq_dataset.to_batches(batch_size=batch_size)

    def setup_queries(
        self, load_queries: bool = True, doc_sample_fraction: float = 1.0, query_limit=0
    ):
        # If there is an explicit 'queries' dataset, then load that and use
        # for querying, otherwise use documents directly.
        if load_queries:
            self._download_dataset_files()
            self.queries = self._load_parquet_dataset("queries", limit=query_limit)
        if not self.queries.empty:
            logging.info(
                f"Using {len(self.queries)} query vectors loaded from dataset 'queries' table"
            )
        else:
            # Queries expect a different schema than documents.
            # Documents looks like:
            #    ["id", "values", "sparse_values", "metadata"]
            # Queries looks like:
            #    ["vector", "sparse_vector", "filter", "top_k"]
            #
            # Extract 'values' and rename to query schema (only
            # 'vector' field of queries is currently used).
            if self.documents.empty:
                self.load_documents()
            assert (
                not self.documents.empty
            ), "Cannot sample 'documents' to use for queries as it is empty"
            self.queries = self.documents[["values"]].copy()
            self.queries.rename(columns={"values": "vector"}, inplace=True)

            # Use a sampling of documents for queries (to avoid
            # keeping a large complete dataset in memory for each
            # worker process).
            self.queries = self.queries.sample(frac=doc_sample_fraction, random_state=1)
            logging.info(
                f"Using {doc_sample_fraction * 100}% of documents' dataset "
                f"for query data ({len(self.queries)} sampled)"
            )

    def upsert_into_index(
        self, index_host, api_key, skip_if_count_identical: bool = False
    ):
        """
        Upsert the datasets' documents into the specified index.
        :param index_host: Pinecone index to upsert into (must already exist)
        :param skip_if_count_identical: If true then skip upsert if the index already contains the same number of
               vectors as the dataset.
        """
        pinecone = PineconeGRPC(api_key)
        index = pinecone.Index(host=index_host)
        if skip_if_count_identical:
            if index.describe_index_stats()["total_vector_count"] == len(
                self.documents
            ):
                logging.info(
                    f"Skipping upsert as index already has same number of documents as dataset ({len(self.documents)}"
                )
                return

        upserted_count = self._upsert_from_dataframe(index)
        if upserted_count != len(self.documents):
            logging.warning(
                f"Not all records upserted successfully. Dataset count:{len(self.documents)},"
                f" upserted count:{upserted_count}"
            )

    def prune_documents(self):
        """
        Discard the contents of self.documents once it is no longer required
        (it can consume a significant amount of memory).
        """
        del self.documents
        logging.debug(
            f"After pruning, 'queries' memory usage:{self.queries.memory_usage()}"
        )

    def _download_dataset_files(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        logging.debug(
            f"Checking for existence of dataset '{self.name}' in dataset cache '{self.cache}'"
        )
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

        for blob in to_download:
            logging.debug(
                f"Dataset file '{blob.name}' not found in cache - will be downloaded"
            )
            dest_path = self.cache / blob.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(self.cache / blob.name)

    def _load_parquet_dataset(self, kind, limit=0):
        parquet_files = [f for f in (self.cache / self.name).glob(kind + "/*.parquet")]
        if not len(parquet_files):
            return pandas.DataFrame()

        dataset = ParquetDataset(parquet_files)
        # Read only the columns that Pinecone SDK makes use of.
        if kind == "documents":
            columns = ["id", "values", "sparse_values", "metadata"]
            metadata_column = "metadata"
        elif kind == "passages":
            # 'passages' format used by benchmarking datasets (e.g. mnist,
            # nq-769-tasb, yfcc, ...). Always has 'id' and 'values' fields;
            # may optionally have `sparse_values` and `metadata`.
            # Validate required fields are present.
            required = set(["id", "values"])
            fields = set(dataset.schema.names)
            missing = fields.difference(required)
            if len(missing) > 0:
                raise ValueError(
                    f"Missing required fields ({missing}) for passages from dataset '{self.name}'"
                )
            # Also load in supported optional fields.
            optional = set(["sparse_values", "metadata"])
            columns = list(required.union((fields.intersection(optional))))
            metadata_column = "metadata"
        elif kind == "queries":
            # 'queries' format which consists of query input parameters
            # and expected results.
            # * Required fields:
            #   - top_k
            #   - values or vector: dense search vector
            #   - ground truth nearest neighbours (stored in 'blob' field)
            # * Optional fields:
            #   - id: query identifier.
            #   - sparse_vector: sparse search vector.
            #   - filter: metadata filter
            fields = set(dataset.schema.names)
            # Validate required fields are present.
            required = set(["top_k", "blob"])
            missing = required.difference(fields)
            if len(missing) > 0:
                raise ValueError(
                    f"Missing required fields ({missing}) for queries from dataset '{self.name}'"
                )
            value_field = set(["values", "vector"]).intersection(fields)
            match len(value_field):
                case 0:
                    raise ValueError(
                        f"Missing required search vector field ('values' or 'vector') queries from dataset '{self.name}'"
                    )
                case 2:
                    raise ValueError(
                        f"Multiple search vector fields ('values' and 'vector') present in queries from dataset '{self.name}'"
                    )
                case 1:
                    required = required | value_field
            # Also load in supported optional fields.
            optional = set(["id", "sparse_vector", "filter"])
            columns = list(required.union((fields.intersection(optional))))
            metadata_column = "filter"
        else:
            raise ValueError(
                f"Unsupported kind '{kind}' - must be one of (documents, queries)"
            )
        # Note: We to specify pandas.ArrowDtype as the types mapper to use pyarrow datatypes in the
        # resulting DataFrame. This is significant as (for reasons unknown) it allows subsequent
        # samples() of the DataFrame to be "disconnected" from the original underlying pyarrow data,
        # and hence significantly reduces memory usage when we later prune away the underlying
        # parrow data (see prune_documents).
        df = dataset.read(columns=columns).to_pandas(types_mapper=pandas.ArrowDtype)
        if limit:
            df = df.iloc[:limit]

        # And drop any columns which all values are missing - e.g. not all
        # datasets have sparse_values, but the parquet file may still have
        # the (empty) column present.
        df.dropna(axis="columns", how="all", inplace=True)

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
                raise TypeError(
                    f"metadata must be a string or dict (found {type(metadata)})"
                )

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
        upserted_count = 0
        for sub_frame in Dataset.split_dataframe(self.documents, 10000):
            # The 'values' column in the DataFrame is a pyarrow type (list<item: double>[pyarrow])
            # as it was read using the pandas.ArrowDtype types_mapper (see _load_parquet_dataset).
            # This _can_ be automatically converted to a Python list object inside upsert_from_dataframe,
            # but it is slow, as at that level the DataFrame is iterated row-by-row and the conversion
            # happens one element at a time.
            # However, converting the entire sub-frame's column back to a Python object before calling
            # upsert_from_dataframe() is significantly faster, such that the overall upsert throughput
            # (including the actual server-side work) is around 2x greater if we pre-convert.
            converted = sub_frame.astype(dtype={"values": object})
            resp = index.upsert_from_dataframe(
                converted, batch_size=200, show_progress=False
            )
            upserted_count += resp.upserted_count
        return upserted_count
