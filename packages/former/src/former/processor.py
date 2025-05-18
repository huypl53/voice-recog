from shared.schema.api import TimestampTranscription
import torch
import pandas as pd
import jiwer
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
from typing import List, Literal, Union, Optional, Tuple, Any
from tqdm import tqdm

import torchaudio.compliance.kaldi as kaldi
from former.model.utils.ctc_utils import get_output_with_timestamps, get_output
from former.util import DTYPE_MAP, init, load_audio, load_config
from contextlib import nullcontext

logger = get_logger(__name__, file_name="./logs/")


class FormerPreprocessor(AudioPreprocessor):
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


class FormerRecognizer(BaseSpeechRecognizer):
    """Wave2Vec2-based implementation of speech recognition"""

    def __init__(self, config=load_config()):
        """Initialize the Wave2Vec2 recognizer"""
        self.model = None
        self.char_dict = None
        self.model_checkpoint = config.model_checkpoint
        self.device = torch.device(config.device)
        self.chunk_size = config.chunk_size
        self.left_context_size = config.left_context_size
        self.right_context_size = config.right_context_size
        self.total_batch_duration = config.total_batch_duration
        self.dtype = DTYPE_MAP[config.dtype]

    def _initialize_models(self) -> Tuple[Any, Any, Any]:
        """Initialize Wav2Vec2 models and language model"""
        model, char_dict = init(self.model_checkpoint, self.device)
        self.model = model
        self.char_dict = char_dict

    @torch.no_grad()
    @log_execution_time(logger=logger)
    def transcribe(
        self,
        audio_path: Union[str, Path],
        mode: Literal["longform", "batch"] = "longform",
    ) -> List[TimestampTranscription]:
        with (
            torch.autocast(self.device.type, self.dtype)
            if self.dtype is not None
            else nullcontext()
        ):
            if mode == "longform":
                return self.endless_decode(audio_path)
            elif mode == "batch":
                return self.batch_decode(audio_path)
            else:
                raise ValueError(f"Invalid mode: {mode}")

    def endless_decode(self, audio_path: Union[str, Path]):

        if self.model is None or self.char_dict is None:
            self._initialize_models()

        model = self.model
        char_dict = self.char_dict

        def get_max_input_context(c, r, n):
            return r + max(c, r) * (n - 1)

        device = next(model.parameters()).device
        # model configuration
        subsampling_factor = model.encoder.embed.subsampling_factor
        conv_lorder = model.encoder.cnn_module_kernel // 2

        # get the maximum length that the gpu can consume
        max_length_limited_context = self.total_batch_duration
        max_length_limited_context = (
            int((max_length_limited_context // 0.01)) // 2
        )  # in 10ms second

        multiply_n = max_length_limited_context // self.chunk_size // subsampling_factor
        truncated_context_size = (
            self.chunk_size * multiply_n
        )  # we only keep this part for text decoding

        # get the relative right context size
        rel_right_context_size = get_max_input_context(
            self.chunk_size,
            max(self.right_context_size, conv_lorder),
            model.encoder.num_blocks,
        )
        rel_right_context_size = rel_right_context_size * subsampling_factor

        waveform = load_audio(str(audio_path))
        offset = torch.zeros(1, dtype=torch.int, device=device)

        # waveform = padding(waveform, sample_rate)
        xs = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000,
        ).unsqueeze(0)

        hyps = []
        att_cache = torch.zeros(
            (
                model.encoder.num_blocks,
                self.left_context_size,
                model.encoder.attention_heads,
                model.encoder._output_size * 2 // model.encoder.attention_heads,
            )
        ).to(device)
        cnn_cache = torch.zeros(
            (model.encoder.num_blocks, model.encoder._output_size, conv_lorder)
        ).to(
            device
        )  # print(context_size)
        for idx, _ in list(
            enumerate(
                range(0, xs.shape[1], truncated_context_size * subsampling_factor)
            )
        ):
            start = max(truncated_context_size * subsampling_factor * idx, 0)
            end = min(
                truncated_context_size * subsampling_factor * (idx + 1) + 7,
                xs.shape[1],
            )

            x = xs[:, start : end + rel_right_context_size]
            x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

            encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset = (
                model.encoder.forward_parallel_chunk(
                    xs=x,
                    xs_origin_lens=x_len,
                    chunk_size=self.chunk_size,
                    left_context_size=self.left_context_size,
                    right_context_size=self.right_context_size,
                    att_cache=att_cache,
                    cnn_cache=cnn_cache,
                    truncated_context_size=truncated_context_size,
                    offset=offset,
                )
            )
            encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[
                :, :encoder_lens
            ]
            if (
                self.chunk_size * multiply_n * subsampling_factor * idx
                + rel_right_context_size
                < xs.shape[1]
            ):
                encoder_outs = encoder_outs[
                    :, :truncated_context_size
                ]  # (B, maxlen, vocab_size) # exclude the output of rel right context
            offset = offset - encoder_lens + encoder_outs.shape[1]

            hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
            hyps.append(hyp)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (
                self.chunk_size * multiply_n * subsampling_factor * idx
                + rel_right_context_size
                >= xs.shape[1]
            ):
                break
        hyps = torch.cat(hyps)
        decode = get_output_with_timestamps([hyps], char_dict)[0]
        return [
            TimestampTranscription(
                content=item["decode"],
                start=item["start"],
                end=item["end"],
            )
            for item in decode
        ]

        # for item in decode:
        #     start = f"{Fore.RED}{item['start']}{Style.RESET_ALL}"
        #     end = f"{Fore.RED}{item['end']}{Style.RESET_ALL}"
        #     print(f"{start} - {end}: {item['decode']}")

    @torch.no_grad()
    def batch_decode(
        self, audio_list_path: str, output_path: Optional[Union[str, Path]] = None
    ):
        """
        Batch decode audio files.

        Args:
            audio_list_path (str): Path to the audio list file
            output_path (Optional[Union[str, Path]], optional): Path to the output file. Defaults to None.
        """
        model = self.model
        char_dict = self.char_dict
        df = pd.read_csv(audio_list_path, sep="\t")
        if not output_path:
            output_path = audio_list_path

        max_length_limited_context = self.total_batch_duration
        max_length_limited_context = (
            int((max_length_limited_context // 0.01)) // 2
        )  # in 10ms second    xs = []
        max_frames = max_length_limited_context
        chunk_size = self.chunk_size
        left_context_size = self.left_context_size
        right_context_size = self.right_context_size
        device = next(model.parameters()).device

        decodes = []
        xs = []
        xs_origin_lens = []
        for idx, audio_path in tqdm(enumerate(df["wav"].to_list())):
            waveform = load_audio(audio_path)
            x = kaldi.fbank(
                waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=16000,
            )

            xs.append(x)
            xs_origin_lens.append(x.shape[0])
            max_frames -= xs_origin_lens[-1]

            if (max_frames <= 0) or (idx == len(df) - 1):
                xs_origin_lens = torch.tensor(
                    xs_origin_lens, dtype=torch.int, device=device
                )
                offset = torch.zeros(len(xs), dtype=torch.int, device=device)
                encoder_outs, encoder_lens, n_chunks, _, _, _ = (
                    model.encoder.forward_parallel_chunk(
                        xs=xs,
                        xs_origin_lens=xs_origin_lens,
                        chunk_size=chunk_size,
                        left_context_size=left_context_size,
                        right_context_size=right_context_size,
                        offset=offset,
                    )
                )

                hyps = model.encoder.ctc_forward(encoder_outs, encoder_lens, n_chunks)
                decodes += get_output(hyps, char_dict)

                # reset
                xs = []
                xs_origin_lens = []
                max_frames = max_length_limited_context

        df["decode"] = decodes
        if "txt" in df:
            wer = jiwer.wer(df["txt"].to_list(), decodes)
            print("WER: ", wer)
        df.to_csv(output_path, sep="\t", index=False)


def get_processor() -> AudioProcessor:
    preprocessor = FormerPreprocessor()
    recognizer = FormerRecognizer()
    processor = AudioProcessor(preprocessor, recognizer)
    return processor


def main():
    """Example usage of the Wave2Vec2 processor"""
    # Create preprocessor and recognizer
    preprocessor = FormerPreprocessor()
    recognizer = FormerRecognizer()

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
