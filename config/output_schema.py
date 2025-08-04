from pydantic import BaseModel

class Evaluator(BaseModel):
    behaviors: list[str]

class Standard(BaseModel):
    response: str

class Review(BaseModel):
    safety: bool
    adjustments: list[list[str, str]]
