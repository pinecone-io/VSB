from pydantic import BaseModel

Vector = list[float]


Record = dict[str, ]
class Record(BaseModel):
    id: str
    values: Vector


class SearchRequest(BaseModel):
    values: Vector
    top_k: int
