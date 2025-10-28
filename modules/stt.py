"""
Speech-to-Text (STT) Module using Vosk.
Transcribes audio to text locally.
"""

import logging
import numpy as np
import os
import json
from vosk import Model, KaldiRecognizer
from typing import Optional
import wave
import io
from datetime import datetime
from pathlib import Path


class SpeechToText:
    """
    Speech-to-Text using Vosk for local transcription.
    Vosk is lightweight and optimized for Raspberry Pi.
    """

    def __init__(
        self,
        model_path: str = "models/vosk/vosk-model-small-sv-rhasspy-0.15",
        language: str = "sv",
        sample_rate: int = 16000,
        # Audio recording options
        save_recordings: bool = False,
        recordings_dir: str = "recordings",
        # Legacy parameters kept for backward compatibility but not used with Vosk
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        beam_size: Optional[int] = None,
        temperature: Optional[float] = None,
        initial_prompt: Optional[str] = None,
        vad_filter: Optional[bool] = None,
        vad_min_silence_duration: Optional[int] = None,
        condition_on_previous_text: Optional[bool] = None
    ):
        """
        Initialize the STT engine.

        Args:
            model_path: Path to Vosk model directory
            language: Language code (e.g., 'sv' for Swedish, 'en' for English)
            sample_rate: Audio sample rate (default: 16000 Hz)
            save_recordings: If True, save audio recordings to disk for debugging
            recordings_dir: Directory to save recordings in
            
        Note: Other parameters are kept for backward compatibility with Faster-Whisper
              but are not used by Vosk.
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.language = language
        self.sample_rate = sample_rate
        self.save_recordings = save_recordings
        self.recordings_dir = recordings_dir
        self.model: Optional[Model] = None
        self.recognizer: Optional[KaldiRecognizer] = None

        # Create recordings directory if needed
        if self.save_recordings:
            Path(self.recordings_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Audio recordings will be saved to: {self.recordings_dir}")

        self.logger.info(f"Initializing Vosk STT with model: {model_path}, "
                         f"language: {language}, sample_rate: {sample_rate}")

    def load_model(self) -> None:
        """
        Load the Vosk model.
        """
        try:
            if not os.path.exists(self.model_path):
                self.logger.error(f"Vosk model not found at: {self.model_path}")
                self.logger.error("Please download the Swedish Vosk model from:")
                self.logger.error("https://alphacephei.com/vosk/models")
                self.logger.error("For Swedish, download: vosk-model-small-sv-rhasspy-0.15")
                raise FileNotFoundError(f"Vosk model not found at: {self.model_path}")
            
            self.logger.info(f"Loading Vosk model from '{self.model_path}'...")
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)  # Enable word-level timestamps
            
            self.logger.info("Vosk model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Vosk model: {e}")
            raise

    def save_audio_to_wav(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        filename: Optional[str] = None
    ) -> str:
        """
        Save audio data to a WAV file for debugging.

        Args:
            audio_data: Audio data as numpy array (float32, normalized to [-1, 1])
            sample_rate: Sample rate of the audio
            filename: Optional filename (without path). If None, generates timestamp-based name.

        Returns:
            Path to the saved WAV file
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recording_{timestamp}.wav"
            
            # Ensure recordings directory exists
            Path(self.recordings_dir).mkdir(parents=True, exist_ok=True)
            
            # Full path to the file
            filepath = os.path.join(self.recordings_dir, filename)
            
            # Ensure audio is float32 normalized to [-1, 1]
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32767.0
                else:
                    audio_data = audio_data.astype(np.float32)
            
            # Convert to int16 for WAV file
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            self.logger.info(f"Audio saved to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving audio to WAV: {e}")
            return ""

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
        if self.model is None or self.recognizer is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return ""

        try:
            # Save audio recording if enabled
            if self.save_recordings:
                self.save_audio_to_wav(audio_data, sample_rate)
            
            # Ensure audio is float32 normalized to [-1, 1]
            if audio_data.dtype != np.float32:
                if audio_data.dtype == np.int16:
                    # Normalize int16 to float32 range [-1, 1]
                    audio_data = audio_data.astype(np.float32) / 32767.0
                else:
                    # For other types, just convert to float32
                    audio_data = audio_data.astype(np.float32)
            
            # Convert float32 audio to int16 for Vosk
            # Vosk expects int16 PCM data
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            self.logger.debug(f"Transcribing audio of length {len(audio_data)} samples")

            # Reset recognizer for new audio
            self.recognizer = KaldiRecognizer(self.model, sample_rate)
            self.recognizer.SetWords(True)

            # Process audio in chunks for better memory management
            chunk_size = 4000  # Process 4000 bytes at a time
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                self.recognizer.AcceptWaveform(chunk)

            # Get final result
            result = json.loads(self.recognizer.FinalResult())
            transcription = result.get('text', '').strip()

            self.logger.info(f"Transcription: {transcription}")

            return transcription

        except Exception as e:
            self.logger.error(f"Error during transcription: {e}")
            return ""

    def transcribe_file(self, audio_file_path: str) -> str:
        """
        Transcribe an audio file to text.

        Args:
            audio_file_path: Path to audio file (WAV format)

        Returns:
            Transcribed text
        """
        if self.model is None or self.recognizer is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return ""

        try:
            self.logger.info(f"Transcribing file: {audio_file_path}")

            # Open WAV file
            with wave.open(audio_file_path, 'rb') as wf:
                # Verify format
                if wf.getnchannels() != 1:
                    self.logger.error("Audio file must be mono (1 channel)")
                    return ""
                if wf.getsampwidth() != 2:
                    self.logger.error("Audio file must be 16-bit")
                    return ""
                if wf.getframerate() != self.sample_rate:
                    self.logger.warning(f"Audio sample rate {wf.getframerate()} != {self.sample_rate}")

                # Reset recognizer
                self.recognizer = KaldiRecognizer(self.model, wf.getframerate())
                self.recognizer.SetWords(True)

                # Process audio in chunks
                chunk_size = 4000
                while True:
                    data = wf.readframes(chunk_size)
                    if len(data) == 0:
                        break
                    self.recognizer.AcceptWaveform(data)

                # Get final result
                result = json.loads(self.recognizer.FinalResult())
                transcription = result.get('text', '').strip()

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
            audio_bytes: Raw audio bytes (PCM, int16)
            sample_rate: Sample rate
            channels: Number of channels (must be 1)
            sample_width: Sample width in bytes (must be 2 for 16-bit)

        Returns:
            Transcribed text
        """
        try:
            if channels != 1:
                self.logger.error("Only mono audio (1 channel) is supported")
                return ""
            if sample_width != 2:
                self.logger.error("Only 16-bit audio is supported")
                return ""

            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32767.0

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
        return self.model is not None and self.recognizer is not None

    def unload_model(self) -> None:
        """
        Unload the model to free memory.
        """
        if self.recognizer is not None:
            del self.recognizer
            self.recognizer = None
        if self.model is not None:
            del self.model
            self.model = None
            self.logger.info("Vosk model unloaded")
