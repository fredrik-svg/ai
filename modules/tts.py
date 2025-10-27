"""
Text-to-Speech (TTS) Module using Piper.
Converts text responses to spoken audio locally.
"""

import logging
import subprocess
import tempfile
import os
from typing import Optional
import sounddevice as sd
import soundfile as sf
import numpy as np


class TextToSpeech:
    """
    Text-to-Speech using Piper for local speech synthesis.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        speaker: int = 0,
        output_device: Optional[int] = None
    ):
        """
        Initialize the TTS engine.

        Args:
            model_path: Path to Piper ONNX model file
            config_path: Path to model config JSON file (optional, auto-detected if None)
            speaker: Speaker ID for multi-speaker models
            output_device: Output audio device index (None for default)
        """
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.config_path = config_path or f"{model_path}.json"
        self.speaker = speaker
        self.output_device = output_device

        # Verify files exist
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Config file not found: {self.config_path}")

        # Check if piper is installed
        self._check_piper_installation()

        self.logger.info(f"TTS initialized with model: {model_path}")

    def _check_piper_installation(self) -> None:
        """Check if Piper is installed and accessible."""
        try:
            result = subprocess.run(
                ["piper", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.logger.info(f"Piper version: {result.stdout.strip()}")
        except FileNotFoundError:
            self.logger.warning("Piper command not found. Attempting to use python module...")
        except Exception as e:
            self.logger.warning(f"Could not verify Piper installation: {e}")

    def synthesize_to_file(self, text: str, output_path: str) -> bool:
        """
        Synthesize text to an audio file.

        Args:
            text: Text to synthesize
            output_path: Path to save the audio file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Synthesizing text: {text}")

            # Build piper command
            cmd = [
                "piper",
                "--model", self.model_path,
                "--output_file", output_path
            ]

            if os.path.exists(self.config_path):
                cmd.extend(["--config", self.config_path])

            if self.speaker > 0:
                cmd.extend(["--speaker", str(self.speaker)])

            # Run piper with text input
            result = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                timeout=30
            )

            if result.returncode == 0:
                self.logger.info(f"Audio saved to: {output_path}")
                return True
            else:
                self.logger.error(f"Piper failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.logger.error("Piper command timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error during synthesis: {e}")
            return False

    def synthesize_and_play(self, text: str) -> bool:
        """
        Synthesize text and play it immediately.

        Args:
            text: Text to synthesize and speak

        Returns:
            True if successful, False otherwise
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided, skipping TTS")
            return False

        try:
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            # Synthesize to file
            if self.synthesize_to_file(text, temp_path):
                # Play the audio
                self.play_audio_file(temp_path)
                
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    self.logger.warning(f"Could not delete temp file: {e}")

                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Error in synthesize_and_play: {e}")
            return False

    def _trim_silence(self, data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio data.

        Args:
            data: Audio data array
            threshold: Amplitude threshold for silence detection

        Returns:
            Trimmed audio data
        """
        # Find first and last non-silent samples
        non_silent = np.abs(data) > threshold
        
        if not non_silent.any():
            # All silence, return original
            return data
        
        # Find indices of first and last non-silent samples
        first_idx = np.argmax(non_silent)
        last_idx = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        # Add small padding (50ms at 22050 Hz = ~1100 samples, scale to actual rate)
        padding = int(0.05 * 22050)  # 50ms padding
        first_idx = max(0, first_idx - padding)
        last_idx = min(len(data) - 1, last_idx + padding)
        
        return data[first_idx:last_idx + 1]

    def _normalize_audio(self, data: np.ndarray, target_level: float = 0.8) -> np.ndarray:
        """
        Normalize audio data to reduce volume variations.

        Args:
            data: Audio data array
            target_level: Target peak level (0.0 to 1.0)

        Returns:
            Normalized audio data
        """
        # Find peak amplitude
        peak = np.abs(data).max()
        
        if peak > 0:
            # Normalize to target level
            data = data * (target_level / peak)
        
        return data

    def play_audio_file(self, file_path: str) -> bool:
        """
        Play an audio file.

        Args:
            file_path: Path to audio file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"Playing audio: {file_path}")

            # Read the audio file
            data, sample_rate = sf.read(file_path)

            # Trim silence from beginning and end to reduce noise
            data = self._trim_silence(data)
            
            # Normalize audio for consistent volume
            data = self._normalize_audio(data)

            # Play the audio
            sd.play(data, sample_rate, device=self.output_device)
            sd.wait()  # Wait for playback to finish

            self.logger.info("Audio playback finished")
            return True

        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
            return False

    def speak(self, text: str) -> bool:
        """
        Convenience method to speak text immediately.

        Args:
            text: Text to speak

        Returns:
            True if successful, False otherwise
        """
        return self.synthesize_and_play(text)

    def test_audio_output(self) -> bool:
        """
        Test the audio output by playing a test message.

        Returns:
            True if successful, False otherwise
        """
        test_text = "Hej, jag är din röstassistent."
        self.logger.info("Testing audio output...")
        return self.speak(test_text)

    @staticmethod
    def list_audio_devices() -> None:
        """List all available audio output devices."""
        print("\nAvailable audio devices:")
        print(sd.query_devices())

    def set_output_device(self, device: Optional[int]) -> None:
        """
        Set the output audio device.

        Args:
            device: Device index (None for default)
        """
        self.output_device = device
        self.logger.info(f"Output device set to: {device}")
