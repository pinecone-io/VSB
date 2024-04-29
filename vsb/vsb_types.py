from pydantic import BaseModel

Vector = list[float]


class Record(BaseModel):
    id: str
    values: Vector


class SearchRequest(BaseModel):
    values: Vector
    top_k: int
