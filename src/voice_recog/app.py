from fastapi import File, UploadFile, Body, FastAPI
import tempfile
from pathlib import Path
import time
from shared.schema.api import ModelType
from shared.schema.api import TranscribeRequest
from shared.schema.api import TranscriptionResponse
from shared.schema.api import ErrorResponse
from wave2vec_250.processor import get_processor as get_wave2vec_processor
from former.processor import get_processor as get_former_processor
from shared.logger import get_logger
import uvicorn
import dotenv
import os
from typing import Dict
from shared.audio.processor import AudioProcessor

logger = get_logger("voice_recog", file_name="./logs/")

dotenv.load_dotenv()


# Initialize processor once at startup

PROCESSORS: Dict[ModelType, AudioProcessor] = {
    ModelType.WAVE2VEC250: get_wave2vec_processor(),
    ModelType.FORMER: get_former_processor(),
}

app = FastAPI(
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
async def transcribe(
    file: UploadFile = File(...),
    config: TranscribeRequest = Body(...),
):
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
    model = config.model
    logger.info(f"Model: {model}")

    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(file.filename).suffix
        ) as temp_file:
            # Write the uploaded file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        processor = PROCESSORS[model]

        if model == ModelType.FORMER:
            # Process the audio file
            transcription = processor.transcribe(temp_file_path)

        elif model == ModelType.WAVE2VEC250:
            # Process the audio file
            use_beam_search = config.use_beam_search
            beam_width = config.beam_width

            if use_beam_search and not beam_width:
                raise ValueError("Beam width is required when using beam search")
            elif not use_beam_search and beam_width:
                raise ValueError("Beam width is not allowed when using greedy search")

            greedy_output, beam_output = processor.transcribe(
                temp_file_path,
                use_beam_search=use_beam_search,
                beam_width=beam_width,
            )

            transcription = beam_output if beam_output else greedy_output

            # Calculate execution time
        execution_time = time.time() - start_time

        # Clean up the temporary file
        Path(temp_file_path).unlink()

        return TranscriptionResponse(
            transcription=transcription,
            execution_time=f"{execution_time:.2f} seconds",
            error=None,
        )

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        # raise ErrorResponse(
        return ErrorResponse(
            transcription=None,
            execution_time=f"{time.time() - start_time:.2f} seconds",
            error=str(e),
        )


def start():
    """Start the FastAPI application"""
    import sys

    dir_path = Path(os.path.abspath(__file__)).parent.parent
    print(f"Start at: {dir_path}")
    sys.path.append(str(dir_path))

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting server at host: %s, port: %s", host, port)
    uvicorn.run("voice_recog.app:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    start()
