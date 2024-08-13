import json
from abc import ABC
from collections.abc import Iterator
from typing import Generator

import pandas
import numpy as np
import pyarrow

from vsb import logger
from ..base import VectorWorkload
from ..parquet_workload.parquet_workload import ParquetSubsetWorkload
from ...vsb_types import SearchRequest, RecordList, Record, DistanceMetric, Vector
from ...databases.pgvector.filter_util import FilterUtil


class SyntheticWorkload(VectorWorkload, ABC):
    """A static workload which is implemented by reading records and query from
    two sets of parquet files.
    The initial records for the workload are loaded from one set of parquet
    files, then the run phase of the workload consists of queries loaded
    from a second set of parquet files.
    """

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
        self, num_users: int, user_id: int
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
