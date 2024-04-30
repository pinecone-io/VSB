from enum import Enum

from pydantic import BaseModel

Vector = list[float]


class Record(BaseModel):
    id: str
    values: Vector


class SearchRequest(BaseModel):
    values: Vector
    top_k: int


class DistanceMetric(Enum):
    Cosine = "cosine"
    Euclidean = "euclidean"
    DotProduct = "dotproduct"
