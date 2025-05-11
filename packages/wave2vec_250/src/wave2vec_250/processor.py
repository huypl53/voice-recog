"""Wave2Vec2-based audio processor for Vietnamese speech recognition"""

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import kenlm
from pyctcdecode import Alphabet, BeamSearchDecoderCTC, LanguageModel
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shared.audio.processor import (
    AudioProcessor,
    BaseSpeechRecognizer,
    AudioPreprocessor,
)
from shared.logger import get_logger, log_execution_time
from typing import Union, Optional, Tuple, Any

logger = get_logger(__name__, file_name="./logs/")


class Wave2VecPreprocessor(AudioPreprocessor):
    """Wave2Vec2-specific implementation of audio preprocessing"""

    def preprocess(
        self,
        audio_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        visualize: bool = False,
    ) -> str:
        """Implementation of audio preprocessing"""
        audio_path = Path(audio_path)
        if output_path is None:
            output_path = f"{audio_path.stem}_processed{audio_path.suffix}"
        output_path = Path(output_path)

        # Load and process audio
        self.logger.info(f"Loading audio from {audio_path}")
        y, sr = librosa.load(str(audio_path), sr=None)

        # Apply processing steps
        y_denoised = self._reduce_noise(y, sr)
        y_normalized = self._normalize_volume(y_denoised, sr)

        if self.use_voice_normalization:
            y_enhanced = self._enhance_voice(y_normalized, sr)
        else:
            y_enhanced = y_normalized

        # Resample if needed
        if sr != self.target_sr:
            y_final = librosa.resample(y_enhanced, orig_sr=sr, target_sr=self.target_sr)
            out_sr = self.target_sr
        else:
            y_final = y_enhanced
            out_sr = sr

        # Trim silence
        y_final, _ = librosa.effects.trim(y_final, top_db=20)

        # Visualize if requested
        if visualize:
            self._visualize_processing(y, y_final, sr, out_sr)

        # Save processed audio
        sf.write(str(output_path), y_final, out_sr, subtype="PCM_24")
        self.logger.info(f"Saved processed audio to {output_path}")

        return str(output_path)

    def _reduce_noise(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Implementation of noise reduction"""
        return librosa.effects.preemphasis(y)

    def _normalize_volume(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Implementation of volume normalization"""
        peak = np.max(np.abs(y))
        if peak > 0:
            y_norm = y / peak
            target_rms = 10 ** (self.target_db / 20.0)
            current_rms = np.sqrt(np.mean(y_norm**2))
            if current_rms > 0:
                y_norm = y_norm * (target_rms / current_rms)
            return y_norm
        return y

    def _enhance_voice(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Implementation of voice enhancement"""
        y_highpass = librosa.effects.high_pass_filter(y, sr=sr, cutoff=300)
        y_bandpassed = librosa.effects.low_pass_filter(y_highpass, sr=sr, cutoff=3000)
        return y_bandpassed * 1.2

    def _visualize_processing(self, y_orig, y_proc, sr_orig, sr_proc):
        """Visualize the audio before and after processing"""
        plt.figure(figsize=(14, 8))

        # Original waveform
        plt.subplot(2, 2, 1)
        plt.title("Original Waveform")
        librosa.display.waveshow(y_orig, sr=sr_orig)

        # Processed waveform
        plt.subplot(2, 2, 2)
        plt.title("Processed Waveform")
        librosa.display.waveshow(y_proc, sr=sr_proc)

        # Original spectrogram
        plt.subplot(2, 2, 3)
        plt.title("Original Spectrogram")
        D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
        librosa.display.specshow(D_orig, sr=sr_orig, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")

        # Processed spectrogram
        plt.subplot(2, 2, 4)
        plt.title("Processed Spectrogram")
        D_proc = librosa.amplitude_to_db(np.abs(librosa.stft(y_proc)), ref=np.max)
        librosa.display.specshow(D_proc, sr=sr_proc, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")

        plt.tight_layout()
        plt.show()


class Wave2VecRecognizer(BaseSpeechRecognizer):
    """Wave2Vec2-based implementation of speech recognition"""

    def __init__(self, cache_dir: Union[str, Path] = "./cache/"):
        """Initialize the Wave2Vec2 recognizer"""
        super().__init__(cache_dir)
        self.processor, self.model, self.ngram_lm_model = self._initialize_models()

    def _initialize_models(self) -> Tuple[Any, Any, Any]:
        """Initialize Wav2Vec2 models and language model"""
        processor = Wav2Vec2Processor.from_pretrained(
            "nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=self.cache_dir
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            "nguyenvulebinh/wav2vec2-base-vietnamese-250h", cache_dir=self.cache_dir
        )
        lm_file = str(self.cache_dir / "vi_lm_4grams.bin")
        ngram_lm_model = self._get_decoder_ngram_model(processor.tokenizer, lm_file)
        return processor, model, ngram_lm_model

    def _get_decoder_ngram_model(self, tokenizer, ngram_lm_path):
        """Initialize n-gram language model for decoding"""
        vocab_dict = tokenizer.get_vocab()
        sort_vocab = sorted((value, key) for (key, value) in vocab_dict.items())
        vocab = [x[1] for x in sort_vocab][:-2]
        vocab_list = vocab

        # Convert special tokens
        vocab_list[tokenizer.pad_token_id] = ""
        vocab_list[tokenizer.unk_token_id] = ""
        vocab_list[tokenizer.word_delimiter_token_id] = " "

        alphabet = Alphabet.build_alphabet(
            vocab_list, ctc_token_idx=tokenizer.pad_token_id
        )
        lm_model = kenlm.Model(ngram_lm_path)
        decoder = BeamSearchDecoderCTC(alphabet, language_model=LanguageModel(lm_model))
        return decoder

    @log_execution_time(logger=logger)
    def transcribe(
        self,
        audio_path: Union[str, Path],
        use_beam_search: bool = True,
        beam_width: int = 500,
    ) -> Tuple[str, Optional[str]]:
        """Implementation of speech recognition"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)

        # Prepare input for model
        input_values = self.processor(
            y, sampling_rate=sr, return_tensors="pt"
        ).input_values

        # Get model predictions
        with torch.no_grad():
            logits = self.model(input_values).logits[0]

        # Decode predictions
        pred_ids = torch.argmax(logits, dim=-1)
        greedy_search_output = self.processor.decode(pred_ids)

        if use_beam_search:
            beam_search_output = self.ngram_lm_model.decode(
                logits.cpu().detach().numpy(), beam_width=beam_width
            )
        else:
            beam_search_output = None

        return greedy_search_output, beam_search_output


def get_processor() -> AudioProcessor:
    preprocessor = Wave2VecPreprocessor()
    recognizer = Wave2VecRecognizer()
    processor = AudioProcessor(preprocessor, recognizer)
    return processor


def main():
    """Example usage of the Wave2Vec2 processor"""
    # Create preprocessor and recognizer
    preprocessor = Wave2VecPreprocessor()
    recognizer = Wave2VecRecognizer()

    # Create combined processor
    processor = AudioProcessor(preprocessor, recognizer)

    # Process and transcribe audio
    audio_file = (
        "E:/code/AI/voice/voice-recog-survey/samples/ssstik.io_1746777311077.mp3"
    )
    greedy_output, beam_output = processor.transcribe(audio_file)
    print("Greedy search output:", greedy_output)
    print("Beam search output:", beam_output)


if __name__ == "__main__":
    main()
