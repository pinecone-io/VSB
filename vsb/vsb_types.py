from enum import Enum

from pydantic import BaseModel, RootModel

VectorInt = list[int]
VectorFloat = list[float]
Vector = VectorInt | VectorFloat


class Record(BaseModel):
    id: str
    values: Vector
    metadata: dict = None


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


class DistanceMetric(Enum):
    Cosine = "cosine"
    Euclidean = "euclidean"
    DotProduct = "dotproduct"
