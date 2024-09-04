"""
A collection of classes that generate synthetic workloads, with
pseudo-randomly generated records and requests in some order.

SyntheticWorkload and SyntheticRunbook steps are generated eagerly
at initialization, and records/requests are held in memory with InMemoryWorkload,
so in-place recall calculation is possible. SyntheticProportionalWorkload generates 
a workload with a specified proportion of queries, upserts, deletes, and fetches, 
data of which is generated at request time with a specified distribution. 
"""

from abc import ABC
from collections.abc import Iterator
from typing import Generator

import pandas
import numpy as np
import random
import re
import string

from vsb import logger
from ..base import VectorWorkload, VectorWorkloadSequence
from ...vsb_types import (
    QueryRequest,
    SearchRequest,
    InsertRequest,
    UpdateRequest,
    FetchRequest,
    DeleteRequest,
    RecordList,
    Record,
    DistanceMetric,
    Vector,
)
from ..parquet_workload.parquet_workload import ParquetSubsetWorkload
from ...databases.pgvector.filter_util import FilterUtil


class InMemoryWorkload(VectorWorkload, ABC):
    """A workload that stores records and queries in memory."""

    def __init__(
        self,
        name: str,
    ):
        # Set up records and queries in a subclass constructor.
        super().__init__(name)
        self.records = None
        self.queries = None

    def get_sample_record(self) -> Record:
        return Record(**self.records.head(1).to_dict("records")[0])

    def get_record_batch_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, RecordList]]:

        if self.records is None:
            logger.warning(
                f"Had to lazy load records for {self.name} workload - "
                f"load_on_init is intended for MasterRunners to skip loading,"
                f"this shouldn't happen."
            )
            self.setup_records()

        # Calculate start / end for this record chunk, then split the table
        # and create an iterator over it.
        quotient, remainder = divmod(self.records.shape[0], num_users)
        chunks = [quotient + (1 if r < remainder else 0) for r in range(num_users)]
        # Determine start position based on sum of size of all chunks prior
        # to ours.
        start = sum(chunks[:user_id])
        end = start + chunks[user_id]
        # partition our record DataFrame into chunks of max size batch_size
        user_chunk: pandas.DataFrame = self.records.iloc[start:end]

        # Need to convert the pyarrow RecordBatch into a pandas DataFrame with
        # correctly formatted fields for consumption by the database.
        def dataframe_to_recordlist(
            records: pandas.DataFrame,
        ) -> Generator[tuple[str, list[dict]], None, None]:
            for batch in np.array_split(
                records, np.ceil(records.shape[0] / batch_size)
            ):
                yield "", RecordList(batch.to_dict("records"))

        return dataframe_to_recordlist(user_chunk)

    def get_query_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, SearchRequest]]:
        if self.queries is None:
            logger.warning(
                f"Had to lazy load queries for {self.name} workload - "
                f"load_on_init is intended for MasterRunners to skip loading,"
                f"this shouldn't happen."
            )
            self.queries = self.setup_queries()
        # Calculate start / end for this query chunk, then split the table
        # and create an iterator over it.
        quotient, remainder = divmod(self.queries.shape[0], num_users)
        chunks = [quotient + (1 if r < remainder else 0) for r in range(num_users)]
        # Determine start position based on sum of size of all chunks prior
        # to ours.
        start = sum(chunks[:user_id])
        end = start + chunks[user_id]

        # Return an iterator for the Nth chunk.
        def make_query_iter(start, end):
            for index in range(start, end):
                # Iterate over indexes because we're yielding one at a time, and
                # .iat is faster than .iloc for single values.
                query = {
                    "values": self.queries["values"].iat[index],
                    "top_k": self.queries["top_k"].iat[index],
                    "neighbors": self.queries["neighbors"].iat[index],
                    # TODO: Add metadata support
                }

                # TODO: Add multiple tenant support.
                yield "", SearchRequest(**query)

        return make_query_iter(start, end)

    # Information methods can't be static because they vary per instance.
    # Upon calling these methods statically, VectorWorkload's
    # NotImplementedError will be raised.

    def dimensions(self) -> int:
        return self._dimensions

    def metric(self) -> DistanceMetric:
        return self._metric

    def record_count(self) -> int:
        return self._record_count

    def request_count(self) -> int:
        return self._request_count


class SyntheticWorkload(InMemoryWorkload, ABC):
    """A workload in which records and queries are generated pseudo-randomly."""

    def __init__(
        self,
        name: str,
        options,
        load_on_init: bool = True,
        **kwargs,
    ):
        super().__init__(name)
        self._record_count = options.synthetic_records
        self._record_distribution = options.synthetic_record_distribution
        self._query_distribution = options.synthetic_query_distribution
        self._request_count = options.synthetic_requests
        self._dimensions = options.synthetic_dimensions
        self._metric = DistanceMetric(options.synthetic_metric)
        self._top_k = options.synthetic_top_k
        seed = int(options.synthetic_seed)
        if seed:
            self.rng = np.random.default_rng(np.random.SeedSequence(seed))
        else:
            ss = np.random.SeedSequence()
            self.seed = ss.entropy  # 128-bit integer, easy enough to copy
            self.rng = np.random.default_rng(ss)

        if load_on_init:
            self.setup_records()
            self.setup_queries()
        else:
            self.records = None
            self.queries = None

    def get_random_vector(self) -> Vector:
        # Generate a pseudo-random vector from the workload's distribution
        match self._metric:
            case DistanceMetric.Cosine | DistanceMetric.DotProduct:
                bounds = (-1, 1)
            case DistanceMetric.Euclidean:
                bounds = (0, 255)
        match self._record_distribution:
            case "uniform":
                return self.rng.uniform(bounds[0], bounds[1], self._dimensions)
            case "normal":
                # Center is usually offset from midpoint by a certain amount
                offset = self.rng.uniform(-0.2, 0.2)
                center = (bounds[0] + bounds[1]) / 2 + offset * (bounds[1] - bounds[0])
                # Use a heuristic 4.5 stdev range of bounds - seems to
                # mimic embeddings well
                stdev = (bounds[1] - bounds[0]) / 4.5
                return self.rng.normal(center, stdev, self._dimensions)
            case _:
                raise ValueError(
                    f"Unsupported record distribution: {self.record_distribution}"
                )

    def get_random_query_idx(self, num_idxs: int) -> int:
        # Pick a random record from our records to use as a query,
        # based on the query distribution.
        match self._query_distribution:
            case "uniform":
                return self.rng.integers(0, self._request_count, num_idxs)
            case "zipfian":
                idxs = []
                while len(idxs) < num_idxs:
                    if (offset := self.rng.zipf(1.1)) < self._request_count:
                        idxs.append(offset)
                return idxs
            case _:
                raise ValueError(
                    f"Unsupported query distribution: {self.query_distribution}"
                )

    def setup_records(self):
        # Pseudo-randomly generate the full RecordList of records
        # If dot product or cosine, generate each dimension as [0, 1]
        # If euclidean, use [0, 255]
        self.records = pandas.DataFrame(
            {
                "id": np.arange(self._record_count).astype(str),
                "values": [self.get_random_vector() for _ in range(self._record_count)],
            }
        )

    def setup_queries(self):
        # Pseudo-randomly generate the full RecordList of queries
        # Query will be generated with the same distribution as records
        self.queries = pandas.DataFrame(
            {
                "values": [
                    self.records["values"].iat[i]
                    for i in self.get_random_query_idx(self._request_count)
                ],
                "top_k": np.full(self._request_count, self._top_k),
                # TODO: Add metadata support
            }
        )

        # Recalculate ground truth neighbors for each query
        self.queries["neighbors"] = ParquetSubsetWorkload.recalculate_neighbors(
            self.records, self.queries, self._metric
        )


class SyntheticRunbook(VectorWorkloadSequence, ABC):
    """A synthetic workload sequence that simulates a series of "steps" of
    various operations (upsert, search, delete?) on a database over time.
    Each step contains a populate phase and a request phase; the total
    set of records and requests is divided evenly among the steps.
    """

    class WorkloadSketch(VectorWorkload, ABC):
        """Represents a workload's metadata without storing the actual records or queries.
        Necessary to generate because workloads are lazy-loaded in the MasterRunner.
        """

        def __init__(
            self,
            name: str,
            dimensions: int,
            metric: DistanceMetric,
            record_count: int,
            request_count: int,
            no_aggregate_stats: bool,
        ):
            self._name = name
            self._dimensions = dimensions
            self._metric = metric
            self._record_count = record_count
            self._request_count = request_count
            self._no_aggregate_stats = no_aggregate_stats

        @property
        def name(self) -> str:
            return self._name

        def dimensions(self) -> int:
            return self._dimensions

        def metric(self) -> DistanceMetric:
            return self._metric

        def record_count(self) -> int:
            return self._record_count

        def request_count(self) -> int:
            return self._request_count

        def get_sample_record(self) -> Record:
            # MasterRunner should never call this method.
            raise NotImplementedError

        def get_record_batch_iter(
            self, num_users: int, user_id: int, batch_size: int
        ) -> Iterator[tuple[str, RecordList]]:
            raise NotImplementedError

        def get_query_iter(
            self, num_users: int, user_id: int, batch_size: int
        ) -> Iterator[tuple[str, SearchRequest]]:
            raise NotImplementedError

        def get_stats_prefix(self) -> str:
            return self._name if self._no_aggregate_stats else ""

    def __init__(
        self,
        name: str,
        options,
        load_on_init: bool = True,
        **kwargs,
    ):
        super().__init__(name)
        self._record_count = options.synthetic_records
        self._record_distribution = options.synthetic_record_distribution
        self._query_distribution = options.synthetic_query_distribution
        self._request_count = options.synthetic_requests
        self._dimensions = options.synthetic_dimensions
        self._metric = DistanceMetric(options.synthetic_metric)
        self._metadata_gen = (
            SyntheticProportionalWorkload.parse_synthetic_metadata_template(
                options.synthetic_metadata
            )
        )
        self._top_k = options.synthetic_top_k
        self._steps = options.synthetic_steps
        self._no_aggregate_stats = options.synthetic_no_aggregate_stats
        seed = int(options.synthetic_seed)
        if seed:
            self.rng = np.random.default_rng(np.random.SeedSequence(seed))
        else:
            ss = np.random.SeedSequence()
            self.seed = ss.entropy
            self.rng = np.random.default_rng(ss)

        if load_on_init:
            self.setup_records()
            self.setup_queries()
            self.setup_workloads()
        else:
            self.records = None
            self.queries = None
            self.setup_workload_sketches()

    def workload_count(self) -> int:
        return self._steps

    # Auxiliary methods for reporting purposes, not part of the VectorWorkloadSequence interface
    def record_count(self) -> int:
        return self._record_count

    def request_count(self) -> int:
        return self._request_count

    def get_random_vector(self) -> Vector:
        # Generate a pseudo-random vector from the workload's distribution
        match self._metric:
            case DistanceMetric.Cosine | DistanceMetric.DotProduct:
                bounds = (-1, 1)
            case DistanceMetric.Euclidean:
                bounds = (0, 255)
        match self._record_distribution:
            case "uniform":
                return self.rng.uniform(bounds[0], bounds[1], self._dimensions)
            case "normal":
                # Center is usually offset from midpoint by a certain amount
                offset = self.rng.uniform(-0.2, 0.2)
                center = (bounds[0] + bounds[1]) / 2 + offset * (bounds[1] - bounds[0])
                # Use a heuristic 4.5 stdev range of bounds - seems to
                # mimic embeddings well
                stdev = (bounds[1] - bounds[0]) / 4.5
                return self.rng.normal(center, stdev, self._dimensions)
            case _:
                raise ValueError(
                    f"Unsupported record distribution: {self.record_distribution}"
                )

    def get_random_query_idx(self, num_idxs: int) -> int:
        # Pick a random record from our records to use as a query,
        # based on the query distribution.
        match self._query_distribution:
            case "uniform":
                return self.rng.integers(0, self._request_count, num_idxs)
            case "zipfian":
                idxs = []
                while len(idxs) < num_idxs:
                    if (offset := self.rng.zipf(1.1)) < self._request_count:
                        idxs.append(offset)
                return idxs
            case _:
                raise ValueError(
                    f"Unsupported query distribution: {self.query_distribution}"
                )

    def setup_workload_sketches(self):
        self.workloads = []
        for i in range(self._steps):
            workload_name = (
                f"{self.name}_step_{i+1}" if self._no_aggregate_stats else self.name
            )
            record_count = self._record_count // self._steps * (i + 1) + (
                i < self._record_count % self._steps
            )
            request_count = self._request_count // self._steps * (i + 1) + (
                i < self._request_count % self._steps
            )
            self.workloads.append(
                self.WorkloadSketch(
                    name=workload_name,
                    dimensions=self._dimensions,
                    metric=self._metric,
                    record_count=record_count,
                    request_count=request_count,
                    no_aggregate_stats=self._no_aggregate_stats,
                )
            )
        return self.workloads

    def setup_workloads(self):
        assert self.records is not None
        assert self.queries is not None
        self.workloads = []
        cumulative_records = pandas.DataFrame(columns=["id", "values"])
        record_workload_chunks = np.array_split(self.records, self._steps)
        query_workload_chunks = np.array_split(self.queries, self._steps)
        for i in range(self._steps):
            records = record_workload_chunks[i]
            # TODO: make more efficient, don't use concat
            cumulative_records = pandas.concat([cumulative_records, records])
            workload_name = (
                # Add step number to workload name to make it unique if not aggregating stats
                f"{self.name}_step_{i+1}"
                if self._no_aggregate_stats
                else self.name
            )
            workload = CumulativeSubsetWorkload(
                workload_name,
                records,
                query_workload_chunks[i],
                cumulative_records,
                self._dimensions,
                self._metric,
                self._top_k,
                self._no_aggregate_stats,
            )
            self.workloads.append(workload)
        # clear memory
        self.records = None
        self.queries = None
        return self.workloads

    def setup_records(self):
        # Pseudo-randomly generate the full RecordList of records
        # If dot product or cosine, generate each dimension as [0, 1]
        # If euclidean, use [0, 255]
        self.records = pandas.DataFrame(
            {
                "id": np.arange(self._record_count).astype(str),
                "values": [self.get_random_vector() for _ in range(self._record_count)],
                "metadata": [
                    {key: value(self.rng) for key, value in self._metadata_gen}
                    for _ in range(self._record_count)
                ],
            }
        )

    def setup_queries(self):
        # Pseudo-randomly generate the full RecordList of queries
        self.queries = pandas.DataFrame(
            {
                "values": [
                    self.records["values"].iat[i]
                    for i in self.get_random_query_idx(self._request_count)
                ],
                "top_k": np.full(self._request_count, self._top_k),
                # TODO: Add metadata support
            }
        )


class CumulativeSubsetWorkload(InMemoryWorkload, ABC):
    """A workload that takes in a set of records and queries alongside
    a cumulative record set, and recalculates neighbors for each query
    based on the cumulative record set.
    """

    def __init__(
        self,
        name: str,
        records: pandas.DataFrame,
        queries: pandas.DataFrame,
        cumulative_records: pandas.DataFrame,
        dimensions: int,
        metric: DistanceMetric,
        top_k: int,
        no_aggregate_stats: bool,
    ):
        super().__init__(name)
        self._no_aggregate_stats = no_aggregate_stats
        self._record_count = records.shape[0]
        self._request_count = queries.shape[0]
        self._dimensions = dimensions
        self._metric = metric
        self._top_k = top_k
        self.records = records
        self.queries = queries
        self.queries["neighbors"] = ParquetSubsetWorkload.recalculate_neighbors(
            cumulative_records, self.queries, self._metric
        )

    def get_stats_prefix(self) -> str:
        return self._name if self._no_aggregate_stats else ""


class SyntheticProportionalWorkload(InMemoryWorkload, ABC):
    """A workload that populates an initial set of records,
    then executes populate/search/delete/fetch operations
    in a specified proportion."""

    def __init__(
        self,
        name: str,
        options,
        **kwargs,
    ):
        super().__init__(name)
        self._record_count = options.synthetic_records
        self._request_count = options.synthetic_requests
        self._dimensions = options.synthetic_dimensions
        self._metric = DistanceMetric(options.synthetic_metric)
        self._metadata_gen = self.parse_synthetic_metadata_template(
            options.synthetic_metadata
        )
        self._top_k = options.synthetic_top_k
        self._batch_size = options.synthetic_batch_size
        self._query_proportion = options.synthetic_query_ratio
        self._insert_proportion = options.synthetic_insert_ratio
        self._update_proportion = options.synthetic_update_ratio
        self._delete_proportion = options.synthetic_delete_ratio
        self._fetch_proportion = options.synthetic_fetch_ratio
        self._query_distribution = options.synthetic_query_distribution
        self._record_distribution = options.synthetic_record_distribution
        self.seed = int(options.synthetic_seed)
        self.rng = np.random.default_rng(np.random.SeedSequence(self.seed))

    def id_to_vec(self, id: int, version: int) -> dict:
        # Generate a pseudo-random vector based on the id, version, and seed
        # Instead of storing vectors, we can store ids and lazy generate vectors.
        # The same id/version pair will always generate the same vector.
        seed_seq = np.random.SeedSequence([self.seed, id, version])
        rng = np.random.default_rng(seed_seq)
        metadata = self.generate_synthetic_metadata(rng)

        def get_vector(bounds: tuple[int, int], dimensions: int) -> Vector:
            match self._record_distribution:
                case "uniform":
                    return rng.uniform(bounds[0], bounds[1], dimensions)
                case "normal":
                    # Center is usually offset from midpoint by a certain amount
                    offset = rng.uniform(-0.2, 0.2)
                    center = (bounds[0] + bounds[1]) / 2 + offset * (
                        bounds[1] - bounds[0]
                    )
                    # Use a heuristic 4.5 stdev range of bounds - seems to
                    # mimic embeddings well
                    stdev = (bounds[1] - bounds[0]) / 4.5
                    return rng.normal(center, stdev, dimensions)
                case _:
                    raise ValueError(
                        f"Unsupported record distribution: {self.record_distribution}"
                    )

        match self._metric:
            case DistanceMetric.Cosine | DistanceMetric.DotProduct:
                return {
                    "values": get_vector((-1, 1), self._dimensions),
                    "metadata": metadata,
                }
            case DistanceMetric.Euclidean:
                return {
                    "values": get_vector((0, 255), self._dimensions),
                    "metadata": metadata,
                }

    @staticmethod
    def parse_synthetic_metadata_template(template: list[str] | None) -> dict:
        if template is None:
            return {}
        generator_dict = {}
        for entry in template:
            key, value = entry.split(":")
            match value[-1]:
                case "s":
                    # string of length n
                    generator_dict[key] = lambda rng: "".join(
                        rng.choice(list(string.ascii_letters + string.digits))
                        for _ in range(int(value[:-1]))
                    )
                case "l":
                    # list of n strings of length m
                    regmatch = re.match(r"(\d+)s(\d+)l", value)
                    m, n = int(regmatch.group(1)), int(regmatch.group(2))
                    generator_dict[key] = lambda rng: [
                        "".join(
                            rng.choice(list(string.ascii_letters + string.digits))
                            for _ in range(m)
                        )
                        for _ in range(n)
                    ]
                case "n":
                    # n digit number
                    generator_dict[key] = lambda rng: int(
                        rng.integers(0, 10 ** int(value[:-1]))
                    )
                case "b":
                    # boolean
                    generator_dict[key] = lambda rng: bool(rng.choice([True, False]))
        return generator_dict

    def generate_synthetic_metadata(self, rng) -> dict:
        return {key: gen(rng) for key, gen in self._metadata_gen.items()}

    def query_distributor(self, num_available_indexes, samples) -> list[int]:
        if num_available_indexes == 0:
            return []
        match self._query_distribution:
            case "uniform":
                return self.rng.choice(num_available_indexes, samples, replace=False)
            case "zipfian":
                query_idx = []
                while len(query_idx) < samples:
                    if (offset := self.rng.zipf(1.1) - 1) < num_available_indexes:
                        query_idx.append(offset)
                return query_idx
            case _:
                raise ValueError(
                    f"Unsupported query distribution: {self.query_distribution}"
                )

    def get_sample_record(self):
        return Record(
            id="0",
            **self.id_to_vec(0, 0),
        )

    def get_record_batch_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, RecordList]]:
        user_n_records = self._record_count // num_users + (
            user_id < self._record_count % num_users
        )
        # User-unique upsert id range to avoid conflicts
        insert_index = self._record_count // num_users * user_id + (
            min(self._record_count % num_users, user_id)
        )

        def make_record_iter(num_records, insert_index):
            for next_i in range(0, num_records, batch_size):
                # Yield in batches of max batch_size
                insert_n = min(batch_size, num_records - next_i)
                yield "", RecordList(
                    root=[
                        Record(
                            id=str(i),
                            **self.id_to_vec(i, 0),
                        )
                        for i in range(insert_index, insert_index + insert_n)
                    ]
                )
                # Update insert_index for next batch
                insert_index += insert_n

        return make_record_iter(user_n_records, insert_index)

    def get_query_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, QueryRequest]]:
        user_n_queries = self._request_count // num_users + (
            user_id < self._request_count % num_users
        )
        user_n_records = self._record_count // num_users + (
            user_id < self._record_count % num_users
        )
        # User-unique upsert id range to avoid conflicts
        insert_index = self._record_count + user_id * (user_n_queries + 1)
        # User-unique delete/fetch id range to avoid conflicts
        original_index_start = self._record_count // num_users * user_id + (
            min(self._record_count % num_users, user_id)
        )
        original_index_end = original_index_start + user_n_records
        # We maintain a deque of available indexes for upserts, deletions, fetches,
        # and searches. We delete from the front, and upsert to the back. Fetches
        # and searches will be taken by specified distribution (zipfian, etc.)
        available_indexes = [
            (i, 0) for i in range(original_index_start, original_index_end)
        ]

        def make_query_iter(num_queries, insert_index, available_indexes):
            # Generate queries in batches. These batches will be homogenous, but a
            # single query iter may contain multiple types of queries.
            for query_num in range(0, num_queries, self._batch_size):
                # In case num_queries is not a multiple of batch_size
                curr_batch_size = min(self._batch_size, num_queries - query_num)
                upsert_batch_size = min(curr_batch_size, batch_size)
                # Choose a random request type based on proportions, and
                # do _batch_size requests of that type
                p = [
                    self._query_proportion,
                    self._insert_proportion,
                    self._update_proportion,
                    self._delete_proportion,
                    self._fetch_proportion,
                ]
                # Normalize probabilities to sum to 1
                p = [x / sum(p) for x in p]
                req_type = self.rng.choice(
                    ["search", "insert", "update", "delete", "fetch"],
                    p=p,
                )
                match req_type:
                    case "search":
                        for _ in range(curr_batch_size):
                            idx = self.query_distributor(len(available_indexes), 1)
                            if not idx:
                                # No available indexes left
                                break
                            idx = idx[0]
                            vector = self.id_to_vec(
                                available_indexes[idx][0], available_indexes[idx][1]
                            )["values"]
                            yield "", SearchRequest(
                                values=vector,
                                top_k=self._top_k,
                                neighbors=[],
                            )
                    case "insert":
                        for next_i in range(0, curr_batch_size, upsert_batch_size):
                            # Yield in batches of max upsert_batch_size
                            insert_n = min(upsert_batch_size, curr_batch_size - next_i)
                            yield "", InsertRequest(
                                records=RecordList(
                                    root=[
                                        Record(id=str(i), **self.id_to_vec(i, 0))
                                        for i in range(
                                            insert_index, insert_index + insert_n
                                        )
                                    ]
                                )
                            )
                            # Update available_indexes for deletions
                            available_indexes.extend(
                                [
                                    (i, 0)
                                    for i in range(
                                        insert_index, insert_index + insert_n
                                    )
                                ]
                            )
                            # Update insert_index for next batch
                            insert_index += insert_n
                    case "update":
                        # Update a sample of existing records
                        idxs = self.query_distributor(
                            len(available_indexes), curr_batch_size
                        )
                        for i in idxs:
                            available_indexes[i] = (
                                available_indexes[i][0],
                                available_indexes[i][1] + 1,
                            )
                        updated_records = [
                            Record(
                                id=str(available_indexes[i][0]),
                                **self.id_to_vec(
                                    available_indexes[i][0], available_indexes[i][1]
                                ),
                            )
                            for i in idxs
                        ]
                        for next_i in range(0, curr_batch_size, upsert_batch_size):
                            update_n = min(upsert_batch_size, curr_batch_size - next_i)
                            yield "", UpdateRequest(
                                records=RecordList(
                                    root=updated_records[next_i : next_i + update_n]
                                )
                            )

                    case "delete":
                        # Delete an arbitrary subset of records
                        delete_ids = [
                            str(i) for i, _ in available_indexes[:curr_batch_size]
                        ]
                        available_indexes = available_indexes[curr_batch_size:]
                        yield "", DeleteRequest(ids=delete_ids)
                    case "fetch":
                        idxs = self.query_distributor(
                            len(available_indexes), curr_batch_size
                        )
                        fetch_ids = [str(available_indexes[i][0]) for i in idxs]

                        yield "", FetchRequest(ids=fetch_ids)

        return make_query_iter(user_n_queries, insert_index, available_indexes)

    def recall_available(self) -> bool:
        return False
