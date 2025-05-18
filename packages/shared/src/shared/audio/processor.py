"""Abstract base classes for audio processing and speech recognition"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
from shared.logger import get_logger, log_execution_time
import logging

from shared.schema.api import TimestampTranscription


class BaseAudioPreprocessor(ABC):
    """Abstract base class for audio preprocessing"""

    def __init__(self, target_sr: int = 16000, target_db: float = -20):
        """
        Initialize the base audio preprocessor.

        Args:
            target_sr: Target sample rate in Hz
            target_db: Target decibel level for normalization
        """
        self.target_sr = target_sr
        self.target_db = target_db
        self.logger = get_logger(__name__, file_name="./logs/")

    @abstractmethod
    def preprocess(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        visualize: bool = False,
    ) -> str:
        """
        Preprocess an audio file.

        Args:
            audio_path: Path to the input audio file
            output_path: Path to save the processed audio (optional)
            visualize: Whether to visualize the audio before and after processing

        Returns:
            Path to the processed audio file
        """
        pass

    @abstractmethod
    def _reduce_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Reduce background noise in audio"""
        pass

    @abstractmethod
    def _normalize_volume(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Normalize the volume of the audio"""
        pass

    @abstractmethod
    def _enhance_voice(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Enhance voice frequencies in the audio"""
        pass


class BaseSpeechRecognizer(ABC):
    """Abstract base class for speech recognition"""

    def __init__(self):
        """
        Initialize the base speech recognizer.

        Args:
            cache_dir: Directory to cache model files
        """
        self.logger = get_logger(__name__, file_name="./logs/")

    @abstractmethod
    def transcribe(
        self,
        audio_path: Union[str, Path],
        **kwargs,
    ) -> Union[str, None, List[str], List[TimestampTranscription]]:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to the audio file
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of (greedy_search_output, beam_search_output)
        """
        pass

    @abstractmethod
    def _initialize_models(self) -> Tuple[Any, Any, Any]:
        """Initialize speech recognition models"""
        pass


class AudioPreprocessor(BaseAudioPreprocessor):
    """
    Concrete implementation of audio preprocessing.
    This class handles all audio preprocessing steps.
    """

    def __init__(
        self,
        target_sr: int = 16000,
        target_db: float = -20,
        use_voice_normalization: bool = True,
    ):
        """
        Initialize the audio preprocessor.

        Args:
            target_sr: Target sample rate in Hz
            target_db: Target decibel level for normalization
            use_voice_normalization: Whether to apply voice-focused normalization
        """
        super().__init__(target_sr, target_db)
        self.use_voice_normalization = use_voice_normalization

    def preprocess(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        visualize: bool = False,
    ) -> str:
        """Implementation of audio preprocessing"""
        # Implementation will be moved from wave2vec processor
        pass

    def _reduce_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Implementation of noise reduction"""
        # Implementation will be moved from wave2vec processor
        pass

    def _normalize_volume(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Implementation of volume normalization"""
        # Implementation will be moved from wave2vec processor
        pass

    def _enhance_voice(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Implementation of voice enhancement"""
        # Implementation will be moved from wave2vec processor
        pass


class AudioProcessor:
    """
    Concrete implementation of audio processing and speech recognition.
    This class combines preprocessing and speech recognition capabilities through composition.
    """

    def __init__(
        self,
        preprocessor: BaseAudioPreprocessor,
        recognizer: BaseSpeechRecognizer,
    ):
        """
        Initialize the audio processor.

        Args:
            preprocessor: Audio preprocessor instance
            recognizer: Speech recognizer instance
        """
        self.preprocessor = preprocessor
        self.recognizer = recognizer
        self.logger = get_logger(__name__, file_name="./logs/")

    def preprocess(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        visualize: bool = False,
    ) -> str:
        """Preprocess audio using the configured preprocessor"""
        return self.preprocessor.preprocess(audio_path, output_path, visualize)

    def transcribe(
        self,
        audio_path: Union[str, Path],
        **kwargs,
    ) -> Tuple[str, Optional[str]]:
        """Transcribe audio using the configured recognizer"""
        return self.recognizer.transcribe(audio_path, **kwargs)
