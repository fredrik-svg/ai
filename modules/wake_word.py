"""
Wake Word Detection Module using Picovoice Porcupine.
Continuously listens for the wake word (e.g., "Hey Genio").
"""

import logging
import struct
import pvporcupine
from typing import Optional, Callable


class WakeWordDetector:
    """
    Wake word detection using Picovoice Porcupine.
    """

    def __init__(self, access_key: str, keyword: str = "hey-genio", sensitivity: float = 0.5):
        """
        Initialize the wake word detector.

        Args:
            access_key: Picovoice access key
            keyword: Wake word keyword to detect
            sensitivity: Detection sensitivity (0.0 to 1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.access_key = access_key
        self.keyword = keyword
        self.sensitivity = sensitivity
        self.porcupine: Optional[pvporcupine.Porcupine] = None

        self.logger.info(f"Initializing wake word detector with keyword: {keyword}")

    def start(self) -> None:
        """
        Start the wake word detector.
        """
        try:
            # Get available keywords
            keywords = pvporcupine.KEYWORDS
            
            # Use built-in keyword if available, otherwise use custom keyword path
            if self.keyword in keywords:
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=[self.keyword],
                    sensitivities=[self.sensitivity]
                )
            else:
                # For custom keywords, you would need to train them via Picovoice Console
                # and provide the .ppn file path
                self.logger.warning(f"Keyword '{self.keyword}' not found in built-in keywords. "
                                    f"Available keywords: {keywords}")
                # Fallback to a common keyword
                fallback_keyword = "jarvis" if "jarvis" in keywords else keywords[0]
                self.logger.info(f"Using fallback keyword: {fallback_keyword}")
                self.porcupine = pvporcupine.create(
                    access_key=self.access_key,
                    keywords=[fallback_keyword],
                    sensitivities=[self.sensitivity]
                )

            self.logger.info(f"Wake word detector started. Frame length: {self.porcupine.frame_length}")
            self.logger.info(f"Sample rate: {self.porcupine.sample_rate}")

        except Exception as e:
            self.logger.error(f"Failed to initialize Porcupine: {e}")
            raise

    def process_audio(self, audio_frame: bytes) -> bool:
        """
        Process an audio frame and check for wake word detection.

        Args:
            audio_frame: Audio frame data (PCM 16-bit)

        Returns:
            True if wake word detected, False otherwise
        """
        if self.porcupine is None:
            self.logger.error("Porcupine not initialized. Call start() first.")
            return False

        try:
            # Convert bytes to array of 16-bit integers
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, audio_frame)
            
            # Process the audio frame
            keyword_index = self.porcupine.process(pcm)
            
            if keyword_index >= 0:
                self.logger.info("Wake word detected!")
                return True

        except Exception as e:
            self.logger.error(f"Error processing audio frame: {e}")

        return False

    def stop(self) -> None:
        """
        Stop and cleanup the wake word detector.
        """
        if self.porcupine is not None:
            self.porcupine.delete()
            self.porcupine = None
            self.logger.info("Wake word detector stopped")

    @property
    def frame_length(self) -> int:
        """Get the required frame length for audio processing."""
        if self.porcupine is None:
            return 512  # Default frame length
        return self.porcupine.frame_length

    @property
    def sample_rate(self) -> int:
        """Get the required sample rate for audio processing."""
        if self.porcupine is None:
            return 16000  # Default sample rate
        return self.porcupine.sample_rate

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
