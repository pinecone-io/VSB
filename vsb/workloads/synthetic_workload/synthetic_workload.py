import json
from abc import ABC
from collections.abc import Iterator
from typing import Generator

import pandas
import numpy as np
import pyarrow

import itertools

from vsb import logger
from ..base import VectorWorkload, VectorWorkloadSequence
from ..parquet_workload.parquet_workload import ParquetSubsetWorkload
from ...vsb_types import (
    QueryRequest,
    SearchRequest,
    UpsertRequest,
    FetchRequest,
    DeleteRequest,
    RecordList,
    Record,
    DistanceMetric,
    Vector,
)
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
        return self._query_count


class SyntheticWorkload(InMemoryWorkload, ABC):
    """A workload in which records and queries are generated pseudo-randomly."""

    def __init__(
        self,
        name: str,
        cache_dir: str,
        record_count: int,
        query_count: int,
        dimensions: int,
        metric: DistanceMetric,
        top_k: int,
        seed: int = None,
        load_on_init: bool = True,
        **kwargs,
    ):
        super().__init__(name)
        self._record_count = record_count
        self._query_count = query_count
        self._dimensions = dimensions
        self._metric = metric
        self._top_k = top_k
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

    def setup_records(self):
        # Pseudo-randomly generate the full RecordList of records
        # If dot product or cosine, generate each dimension as [0, 1]
        # If euclidean, use [0, 255]
        # TODO: Add custom distribution support (normal, hypergeo, zipfian, etc.)
        self.records = pandas.DataFrame(
            {
                "id": np.arange(self._record_count).astype(str),
                "values": [
                    (
                        self.rng.uniform(size=self._dimensions)
                        if self._metric != DistanceMetric.Euclidean
                        else self.rng.uniform(0, 256, self._dimensions)
                    )
                    for _ in range(self._record_count)
                ],
            }
        )

    def setup_queries(self):
        # Pseudo-randomly generate the full RecordList of queries
        # Query will be generated with the same distribution as records
        self.queries = pandas.DataFrame(
            {
                "values": [
                    (
                        self.rng.uniform(size=self._dimensions)
                        if self._metric != DistanceMetric.Euclidean
                        else self.rng.uniform(0, 256, self._dimensions)
                    )
                    for _ in range(self._query_count)
                ],
                "top_k": np.full(self._query_count, self._top_k),
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
            query_count: int,
        ):
            self._name = name
            self._dimensions = dimensions
            self._metric = metric
            self._record_count = record_count
            self._query_count = query_count

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
            return self._query_count

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

    def __init__(
        self,
        name: str,
        cache_dir: str,
        record_count: int,
        query_count: int,
        dimensions: int,
        metric: DistanceMetric,
        top_k: int,
        steps: int,
        no_aggregate_stats: bool,
        seed: int = None,
        load_on_init: bool = True,
        **kwargs,
    ):
        super().__init__(name)
        self._record_count = record_count
        self._query_count = query_count
        self._dimensions = dimensions
        self._metric = metric
        self._top_k = top_k
        self._steps = steps
        self._no_aggregate_stats = no_aggregate_stats
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

    def query_count(self) -> int:
        return self._query_count

    def setup_workload_sketches(self):
        self.workloads = []
        for i in range(self._steps):
            workload_name = (
                f"{self.name}_step_{i+1}" if self._no_aggregate_stats else self.name
            )
            record_count = self._record_count // self._steps * (i + 1) + (
                i < self._record_count % self._steps
            )
            query_count = self._query_count // self._steps * (i + 1) + (
                i < self._query_count % self._steps
            )
            self.workloads.append(
                self.WorkloadSketch(
                    name=workload_name,
                    dimensions=self._dimensions,
                    metric=self._metric,
                    record_count=record_count,
                    query_count=query_count,
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
        # TODO: Add custom distribution support (normal, hypergeo, zipfian, etc.)
        self.records = pandas.DataFrame(
            {
                "id": np.arange(self._record_count).astype(str),
                "values": [
                    (
                        self.rng.uniform(size=self._dimensions)
                        if self._metric != DistanceMetric.Euclidean
                        else self.rng.uniform(0, 256, self._dimensions)
                    )
                    for _ in range(self._record_count)
                ],
            }
        )

    def setup_queries(self):
        # Pseudo-randomly generate the full RecordList of queries
        # Query will be generated with the same distribution as records
        self.queries = pandas.DataFrame(
            {
                "values": [
                    (
                        self.rng.uniform(size=self._dimensions)
                        if self._metric != DistanceMetric.Euclidean
                        else self.rng.uniform(0, 256, self._dimensions)
                    )
                    for _ in range(self._query_count)
                ],
                "top_k": np.full(self._query_count, self._top_k),
                # TODO: Add metadata support
            }
        )

        # Recalculate ground truth neighbors for each query
        self.queries["neighbors"] = ParquetSubsetWorkload.recalculate_neighbors(
            self.records, self.queries, self._metric
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
    ):
        super().__init__(name)
        self._record_count = records.shape[0]
        self._query_count = queries.shape[0]
        self._dimensions = dimensions
        self._metric = metric
        self._top_k = top_k
        self.records = records
        self.queries = queries
        self.queries["neighbors"] = ParquetSubsetWorkload.recalculate_neighbors(
            cumulative_records, self.queries, self._metric
        )


class SyntheticProportionalWorkload(InMemoryWorkload, ABC):
    """A workload that populates an initial set of records,
    then executes populate/search/delete/fetch operations
    in a specified proportion."""

    def __init__(
        self,
        name: str,
        cache_dir: str,
        record_count: int,
        query_count: int,
        dimensions: int,
        metric: DistanceMetric,
        top_k: int,
        batch_size: int,
        query_proportion: float,
        upsert_proportion: float,
        delete_proportion: float,
        fetch_proportion: float,
        query_distribution: str,
        seed: int = None,
        load_on_init: bool = True,
        **kwargs,
    ):
        super().__init__(name)
        self._record_count = record_count
        self._query_count = query_count
        self._dimensions = dimensions
        self._metric = metric
        self._top_k = top_k
        self._batch_size = batch_size
        self._query_proportion = query_proportion
        self._upsert_proportion = upsert_proportion
        self._delete_proportion = delete_proportion
        self._fetch_proportion = fetch_proportion
        self._query_distribution = query_distribution
        if seed:
            self.rng = np.random.default_rng(np.random.SeedSequence(seed))
        else:
            ss = np.random.SeedSequence()
            self.seed = ss.entropy
            self.rng = np.random.default_rng(ss)

        if load_on_init:
            self.setup_records()
        else:
            self.records = None

    def id_to_vec(self, id: int, version: int) -> Vector:
        # Generate a pseudo-random vector based on the id, version, and seed
        # Instead of storing vectors, we can store ids and lazy generate vectors.
        # The same id/version pair will always generate the same vector.
        seed_seq = np.random.SeedSequence([self.seed, id, version])
        rng = np.random.default_rng(seed_seq)
        match self._metric:
            case DistanceMetric.Cosine | DistanceMetric.DotProduct:
                return rng.uniform(size=self._dimensions)
            case DistanceMetric.Euclidean:
                return rng.uniform(0, 256, self._dimensions)

    def query_distributor(self, num_available_indexes, samples) -> list[int]:
        match self._query_distribution:
            case "uniform":
                return self.rng.choice(num_available_indexes, samples, replace=False)
            case "zipfian":
                query_idx = []
                while len(query_idx) < samples:
                    if (offset := self.rng.zipf(1.1)) < num_available_indexes:
                        query_idx.append(offset)
                return query_idx
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
                "values": [self.id_to_vec(id, 0) for id in range(self._record_count)],
            }
        )

    def get_query_iter(
        self, num_users: int, user_id: int, batch_size: int
    ) -> Iterator[tuple[str, QueryRequest]]:
        user_n_queries = self._query_count // num_users + (
            user_id < self._query_count % num_users
        )
        user_n_records = self._record_count // num_users + (
            user_id < self._record_count % num_users
        )
        # User-unique upsert id range to avoid conflicts
        upsert_index = self._record_count + user_id * (user_n_queries + 1)
        # User-unique delete/fetch id range to avoid conflicts
        original_index_start = self._record_count // num_users * user_id + (
            min(self._record_count % num_users, user_id)
        )
        original_index_end = original_index_start + user_n_records
        # We maintain a deque of available indexes for upserts, deletions, fetches,
        # and searches. We delete from the front, and upsert to the back. Fetches
        # and searches will be taken by specified distribution (zipfian, etc.)
        available_indexes = list(range(original_index_start, original_index_end))

        def make_query_iter(num_queries, upsert_index, available_indexes):
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
                    self._upsert_proportion,
                    self._delete_proportion,
                    self._fetch_proportion,
                ]
                # Normalize probabilities to sum to 1
                p = [x / sum(p) for x in p]
                req_type = self.rng.choice(
                    ["search", "upsert", "delete", "fetch"],
                    p=p,
                )
                match req_type:
                    case "search":
                        for _ in range(curr_batch_size):
                            idx = self.query_distributor(len(available_indexes), 1)[0]
                            # TODO: change when we have versioning w/ updates
                            vector = self.id_to_vec(available_indexes[idx], 0)
                            yield "", SearchRequest(
                                values=vector,
                                top_k=self._top_k,
                                neighbors=[],
                            )
                    case "upsert":
                        for next_i in range(0, curr_batch_size, upsert_batch_size):
                            # Yield in batches of max upsert_batch_size
                            upsert_n = min(upsert_batch_size, curr_batch_size - next_i)
                            yield "", UpsertRequest(
                                records=RecordList(
                                    root=[
                                        Record(
                                            id=str(i),
                                            values=self.rng.uniform(
                                                size=self._dimensions
                                            ),
                                        )
                                        for i in range(
                                            upsert_index, upsert_index + upsert_n
                                        )
                                    ]
                                )
                            )
                            # Update available_indexes for deletions
                            available_indexes.extend(
                                range(upsert_index, upsert_index + upsert_n)
                            )
                            # Update upsert_index for next batch
                            upsert_index += upsert_n

                    case "delete":
                        # Delete an arbitrary subset of records
                        delete_ids = [
                            str(i) for i in available_indexes[:curr_batch_size]
                        ]
                        available_indexes = available_indexes[curr_batch_size:]
                        yield "", DeleteRequest(ids=delete_ids)
                    case "fetch":
                        idxs = self.query_distributor(
                            len(available_indexes), curr_batch_size
                        )
                        fetch_ids = [str(available_indexes[i]) for i in idxs]

                        yield "", FetchRequest(ids=fetch_ids)

        return make_query_iter(user_n_queries, upsert_index, available_indexes)
