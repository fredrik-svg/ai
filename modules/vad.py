"""
Voice Activity Detection (VAD) Module.
Detects when a user is speaking to avoid recording silence.
"""

import logging
import collections
import webrtcvad
from typing import Optional


class VoiceActivityDetector:
    """
    Voice Activity Detection using WebRTC VAD.
    """
    
    # Maximum number of frames to buffer (prevents indefinite growth)
    MAX_BUFFER_FRAMES = 10

    def __init__(self, sample_rate: int = 16000, frame_duration: int = 30, mode: int = 3):
        """
        Initialize the VAD detector.

        Args:
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000 Hz)
            frame_duration: Frame duration in milliseconds (10, 20, or 30 ms)
            mode: Aggressiveness mode (0-3, where 3 is most aggressive)
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.mode = mode

        # Validate parameters
        if sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"Invalid sample rate: {sample_rate}. Must be 8000, 16000, 32000, or 48000.")
        
        if frame_duration not in [10, 20, 30]:
            raise ValueError(f"Invalid frame duration: {frame_duration}. Must be 10, 20, or 30 ms.")

        if mode not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid mode: {mode}. Must be 0, 1, 2, or 3.")

        self.vad = webrtcvad.Vad(mode)
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # Ring buffer for smoothing
        self.ring_buffer = collections.deque(maxlen=10)
        self.triggered = False
        
        # Audio buffer for accumulating incoming audio chunks
        self.audio_buffer = b''

        self.logger.info(f"VAD initialized: sample_rate={sample_rate}, "
                         f"frame_duration={frame_duration}ms, mode={mode}, "
                         f"frame_size={self.frame_size} samples ({self.frame_size * 2} bytes)")

    def is_speech(self, audio_frame: bytes) -> bool:
        """
        Check if the audio frame contains speech.

        Args:
            audio_frame: Audio frame data (PCM 16-bit), must be exactly the right size

        Returns:
            True if speech is detected, False otherwise
        """
        expected_bytes = self.frame_size * 2  # 2 bytes per sample for 16-bit PCM
        
        if len(audio_frame) != expected_bytes:
            self.logger.error(f"Invalid frame size: got {len(audio_frame)} bytes, expected {expected_bytes} bytes")
            return False
        
        try:
            return self.vad.is_speech(audio_frame, self.sample_rate)
        except Exception as e:
            self.logger.error(f"Error in VAD processing: {e} (frame size: {len(audio_frame)} bytes)")
            return False

    def process_frame(self, audio_chunk: bytes) -> tuple[bool, bool]:
        """
        Process an audio chunk with smoothing to detect speech start/stop.
        
        This method handles variable-sized audio chunks by buffering them and
        extracting properly-sized frames for VAD processing.

        Args:
            audio_chunk: Audio chunk data (PCM 16-bit), can be any size

        Returns:
            Tuple of (is_speaking, voice_detected)
            - is_speaking: Current speaking state
            - voice_detected: True if voice activity changed
        """
        # Add incoming audio to buffer
        self.audio_buffer += audio_chunk
        
        # Calculate required frame size in bytes
        required_bytes = self.frame_size * 2  # 2 bytes per sample for 16-bit PCM
        
        # Prevent buffer from growing indefinitely
        max_buffer_size = required_bytes * self.MAX_BUFFER_FRAMES
        if len(self.audio_buffer) > max_buffer_size:
            # Keep only the most recent data
            self.audio_buffer = self.audio_buffer[-max_buffer_size:]
            self.logger.debug(f"Buffer trimmed to {len(self.audio_buffer)} bytes")
        
        voice_detected = False
        
        # Process all complete frames in the buffer
        while len(self.audio_buffer) >= required_bytes:
            # Extract one frame
            frame = self.audio_buffer[:required_bytes]
            self.audio_buffer = self.audio_buffer[required_bytes:]
            
            # Check for speech in this frame
            is_speech = self.is_speech(frame)
            self.ring_buffer.append((frame, is_speech))

            num_voiced = sum(1 for f, speech in self.ring_buffer if speech)

            # If we're not triggered and we have enough voiced frames, start recording
            if not self.triggered:
                if num_voiced > 0.8 * self.ring_buffer.maxlen:
                    self.triggered = True
                    voice_detected = True
                    self.logger.debug("Voice activity started")
            # If we're triggered and we have mostly unvoiced frames, stop recording
            else:
                if num_voiced < 0.2 * self.ring_buffer.maxlen:
                    self.triggered = False
                    voice_detected = True
                    self.logger.debug("Voice activity stopped")

        return self.triggered, voice_detected

    def reset(self) -> None:
        """
        Reset the VAD state.
        """
        self.ring_buffer.clear()
        self.triggered = False
        self.audio_buffer = b''
        self.logger.debug("VAD state reset")

    @property
    def is_triggered(self) -> bool:
        """Check if VAD is currently triggered (detecting voice)."""
        return self.triggered

    def get_frame_size(self) -> int:
        """
        Get the frame size in samples.

        Returns:
            Number of samples per frame
        """
        return self.frame_size

    def get_frame_bytes(self) -> int:
        """
        Get the frame size in bytes (for 16-bit PCM).

        Returns:
            Number of bytes per frame
        """
        return self.frame_size * 2  # 2 bytes per sample for 16-bit PCM
