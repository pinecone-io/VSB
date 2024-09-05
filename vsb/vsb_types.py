from enum import Enum

from typing import Any
from pydantic import (
    BaseModel,
    RootModel,
    field_validator,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
)
import numpy as np

VectorInt = list[int]
VectorFloat = list[float]
Vector = VectorInt | VectorFloat


class Record(BaseModel):
    id: str
    values: Vector
    metadata: dict = None

    # We need to override Pydantic's validator, which wastes time validating
    # each value in values, for each Record, since it assumes the list can be heterogeneous.
    # Records are usually constructed by Pandas DataFrame, which is homogeneous.
    # We can check the type once per Record in this case.
    @field_validator("values", mode="wrap")
    @classmethod
    def check_array_type(
        cls, a: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> Vector:
        if isinstance(a, np.ndarray):
            if np.issubdtype(a.dtype, np.integer) or np.issubdtype(
                a.dtype, np.floating
            ):
                return a
            raise ValueError(
                f"Record: values: NumPy array of type {np.dtype} was received, but int or float type was expected"
            )
        # use Pydantic's normal validator
        return handler(a)


class RecordList(RootModel):
    root: list[Record]

    def __len__(self):
        return len(self.root)

    def __iter__(self):
        return iter(self.root)

    def __getitem__(self, item):
        return self.root[item]


class SearchRequest(BaseModel):
    values: Vector
    top_k: int
    filter: dict = None
    neighbors: list[str] = None


class InsertRequest(BaseModel):
    records: RecordList


class UpdateRequest(BaseModel):
    records: RecordList


class DeleteRequest(BaseModel):
    ids: list[str]


class FetchRequest(BaseModel):
    ids: list[str]


QueryRequest = (
    SearchRequest | InsertRequest | UpdateRequest | DeleteRequest | FetchRequest
)


class DistanceMetric(Enum):
    Cosine = "cosine"
    Euclidean = "euclidean"
    DotProduct = "dotproduct"
