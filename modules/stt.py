"""
Speech-to-Text (STT) Module using Faster-Whisper.
Transcribes audio to text locally.
"""

import logging
import numpy as np
import os
from faster_whisper import WhisperModel
from typing import Optional, List
import io
import wave


class SpeechToText:
    """
    Speech-to-Text using Faster-Whisper for local transcription.
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "sv",
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize the STT engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v2, large-v3)
            language: Language code (e.g., 'sv' for Swedish, 'en' for English)
            device: Device to use ('cpu' or 'cuda')
            compute_type: Compute type (int8, int8_float16, float16, float32)
        """
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None

        self.logger.info(f"Initializing Faster-Whisper with model: {model_size}, "
                         f"language: {language}, device: {device}")

    def load_model(self) -> None:
        """
        Load the Whisper model.
        """
        try:
            self.logger.info(f"Loading Whisper model '{self.model_size}'... This may take a while on first run.")
            
            # For CPU devices, explicitly set num_workers=1 to avoid threading issues
            # and cpu_threads to optimize for single-threaded performance on embedded devices
            model_kwargs = {
                'device': self.device,
                'compute_type': self.compute_type
            }
            
            # Optimize for CPU-only devices like Raspberry Pi
            if self.device == 'cpu':
                # Use detected CPU count or conservative default of 2 for embedded devices
                # On Raspberry Pi 5, os.cpu_count() typically returns 4 (quad-core)
                model_kwargs['cpu_threads'] = os.cpu_count() or 2
                model_kwargs['num_workers'] = 1
            
            self.model = WhisperModel(
                self.model_size,
                **model_kwargs
            )
            self.logger.info("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> str:
        """
        Transcribe audio data to text.

        Args:
            audio_data: Audio data as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text
        """
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return ""

        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            max_val = np.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val

            self.logger.debug(f"Transcribing audio of length {len(audio_data)} samples")

            # Transcribe
            segments, info = self.model.transcribe(
                audio_data,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Collect all segments
            transcription = " ".join([segment.text for segment in segments]).strip()

            self.logger.info(f"Transcription: {transcription}")
            self.logger.debug(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            return transcription

        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            return ""

    def transcribe_file(self, audio_file_path: str) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text
        """
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return ""

        try:
            self.logger.info(f"Transcribing file: {audio_file_path}")

            segments, info = self.model.transcribe(
                audio_file_path,
                language=self.language,
                beam_size=5,
                vad_filter=True
            )

            transcription = " ".join([segment.text for segment in segments]).strip()

            self.logger.info(f"Transcription: {transcription}")
            return transcription

        except Exception as e:
            self.logger.error(f"Error transcribing file: {e}")
            return ""

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2
    ) -> str:
        """
        Transcribe raw audio bytes to text.

        Args:
            audio_bytes: Raw audio bytes (PCM)
            sample_rate: Sample rate
            channels: Number of channels
            sample_width: Sample width in bytes (2 for 16-bit)

        Returns:
            Transcribed text
        """
        try:
            # Convert bytes to numpy array
            dtype = np.int16 if sample_width == 2 else np.int32
            audio_array = np.frombuffer(audio_bytes, dtype=dtype)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / np.iinfo(dtype).max

            return self.transcribe_audio(audio_float, sample_rate)

        except Exception as e:
            self.logger.error(f"Error transcribing bytes: {e}")
            return ""

    def is_loaded(self) -> bool:
        """
        Check if the model is loaded.

        Returns:
            True if model is loaded, False otherwise
        """
        return self.model is not None

    def unload_model(self) -> None:
        """
        Unload the model to free memory.
        """
        if self.model is not None:
            del self.model
            self.model = None
            self.logger.info("Whisper model unloaded")
