from collections.abc import Iterator

import numpy
import pyarrow.parquet
from google.cloud.storage import Bucket, Client, transfer_manager
import json
import pandas
import pathlib
from pinecone.grpc import PineconeGRPC
import pyarrow.dataset as ds
from pyarrow.parquet import ParquetDataset, ParquetFile

import vsb
from vsb import logger
from vsb.logging import ProgressIOWrapper


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
        self._download_dataset_files()
        pq_files = list((self.cache / self.name).glob("passages/*.parquet"))

        if self.limit:
            dset = ds.dataset(pq_files)
            columns = self._get_set_of_passages_columns_to_read(dset)
            first_n = dset.head(self.limit, columns=columns)

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
            # then return a batch iterator over the `user_id`th subset.
            chunks = numpy.array_split(pq_files, num_chunks)
            my_chunks = list(chunks[chunk_id])
            if not my_chunks:
                # No chunks for this user - nothing to do.
                return []
            docs_pq_dataset = ds.dataset(my_chunks)
            columns = self._get_set_of_passages_columns_to_read(docs_pq_dataset)

            def files_to_batches(files: list):
                """Given a list of parquet files, return an iterator over
                batches of the given size across all files.
                """
                for f in files:
                    parquet = ParquetFile(f)
                    for batch in parquet.iter_batches(
                        columns=columns, batch_size=batch_size
                    ):
                        yield batch

            return files_to_batches(my_chunks)

    def setup_queries(self, query_limit=0):
        self._download_dataset_files()
        self.queries = self._load_parquet_dataset("queries", limit=query_limit)
        logger.debug(
            f"Using {len(self.queries)} query vectors loaded from dataset 'queries' table"
        )

    def _download_dataset_files(self):
        self.cache.mkdir(parents=True, exist_ok=True)
        logger.debug(
            f"Checking for existence of dataset '{self.name}' in dataset cache '{self.cache}'"
        )
        client = Client.create_anonymous_client()
        bucket: Bucket = client.bucket(Dataset.gcs_bucket)
        blobs = [b for b in bucket.list_blobs(prefix=self.name + "/")]
        # Ignore directories (blobs ending in '/') as we don't explicilty need them
        # (non-empty directories will have their files downloaded
        # anyway).
        blobs = [b for b in blobs if not b.name.endswith("/")]
        logger.debug(
            f"Dataset consists of {len(blobs)} files" f":{[b.name for b in blobs]}"
        )

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

        logger.debug(
            f"Parquet dataset: downloading {len(to_download)} files belonging to "
            f"dataset '{self.name}'"
        )
        with vsb.logging.progress_task(
            "  Downloading dataset files",
            "  ✔ Dataset download complete",
            total=len(to_download),
        ) as download_task:
            for blob in to_download:
                logger.debug(
                    f"Dataset file '{blob.name}' not found in cache - will be downloaded"
                )
                dest_path = self.cache / blob.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_file(
                    ProgressIOWrapper(
                        dest=dest_path, progress=vsb.progress, total=blob.size, indent=2
                    )
                )
                if vsb.progress:
                    vsb.progress.update(download_task, advance=1)

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
            columns = self._get_set_of_passages_columns_to_read(dataset)
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
        logger.debug(f"Loaded {len(df)} vectors of kind '{kind}'")
        return df

    def _get_set_of_passages_columns_to_read(self, dset: ds.Dataset):
        # 'passages' format used by benchmarking datasets (e.g. mnist,
        # nq-769-tasb, yfcc, ...) always have 'id' and 'values' fields;
        # may optionally have `sparse_values` and `metadata`.
        # Validate required fields are present.
        required = set(["id", "values"])
        fields = set(dset.schema.names)
        missing = required.difference(fields)
        if len(missing) > 0:
            raise ValueError(
                f"Missing required fields ({missing}) for passages from dataset '{self.name}'"
            )
        # Also load in supported optional fields.
        optional = set(["sparse_values", "metadata"])
        return list(required.union((fields.intersection(optional))))
