from enum import Enum
from pydantic import BaseModel


class ReportResultStatus(Enum):
    COMPLETE = "complete"
    ERROR = "error"


class ScoreRatio(BaseModel):
    recognition: float
    speed: float
    intonation: float
