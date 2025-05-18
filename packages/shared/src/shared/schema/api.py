from enum import Enum
from typing import Optional, Union, List
from fastapi import File, UploadFile
from pydantic import BaseModel, Field, model_validator
import json


class ModelType(str, Enum):
    WAVE2VEC250 = "wave2vec250"
    FORMER = "former"


class TranscribeRequest(BaseModel):
    model: ModelType = ModelType.FORMER

    # Wave2Vec250
    use_beam_search: bool = True
    beam_width: int = 500

    @model_validator(mode="before")
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


class TimestampTranscription(BaseModel):
    content: str
    start: Union[float, str]
    end: Union[float, str]


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint"""

    transcription: Optional[Union[str, List[TimestampTranscription]]] = Field(
        default=None, description="The transcribed text from the audio file"
    )
    execution_time: Optional[Union[float, str]] = Field(
        ..., description="Time taken to process the file in seconds"
    )
    error: Optional[str] = Field(default=None, description="Error message if any")


class ErrorResponse(BaseModel):
    """Response model for error cases"""

    transcription: None = Field(
        default=None, description="No transcription available due to error"
    )
    execution_time: str = Field(..., description="Time taken before error occurred")
    error: str = Field(..., description="Error message")
