import fastapi
from fastapi import UploadFile, File
import tempfile
from pathlib import Path
import time
from wave2vec_250.processor import get_processor
from shared.logger import get_logger
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import dotenv
import os

logger = get_logger("voice_recog", file_name="./logs/")

dotenv.load_dotenv()


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint"""

    transcription: Optional[str] = Field(
        default=None, description="The transcribed text from the audio file"
    )
    execution_time: str = Field(
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


# Initialize processor once at startup
processor = get_processor()

app = fastapi.FastAPI(
    title="Voice Recognition API",
    description="API for transcribing audio files using Wave2Vec2",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        200: {
            "description": "Successful transcription",
            "model": TranscriptionResponse,
        },
        500: {"description": "Error during processing", "model": ErrorResponse},
    },
)
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe an uploaded audio file.

    Args:
        file: The uploaded audio file

    Returns:
        TranscriptionResponse containing:
        - transcription: The transcribed text
        - execution_time: Time taken to process the file
        - error: Error message if any
    """
    start_time = time.time()

    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Process the audio file
        greedy_output, beam_output = processor.transcribe(temp_file_path)

        # Calculate execution time
        execution_time = time.time() - start_time

        # Clean up the temporary file
        Path(temp_file_path).unlink()

        return TranscriptionResponse(
            transcription=beam_output if beam_output else greedy_output,
            execution_time=f"{execution_time:.2f} seconds",
            error=None,
        )

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return ErrorResponse(
            transcription=None,
            execution_time=f"{time.time() - start_time:.2f} seconds",
            error=str(e),
        )


def start():
    """Start the FastAPI application"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server at host: %s, port: %s", host, port)
    uvicorn.run("voice_recog.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    start()
