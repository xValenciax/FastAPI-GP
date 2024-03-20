from enum import Enum
from pydantic import BaseModel


class TranslationInput(BaseModel):
    text: str | None
    voice_record: str | None


class TranslationOutput(BaseModel):
    translation: str
    time_taken: int


class ModelName(str, Enum):
    Translation = "translate"
    Speech = "speech"
