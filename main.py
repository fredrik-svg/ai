#!/usr/bin/env python3
"""
Main application for the local voice-controlled AI assistant.
Orchestrates all modules: Wake Word, VAD, STT, MQTT, and TTS.
"""

import os
import logging
import sys
import yaml
import signal
import time
import struct
import sounddevice as sd
import numpy as np
from pathlib import Path
from typing import Optional

# Configure ONNX Runtime environment before importing modules
# This prevents GPU device discovery warnings on CPU-only devices like Raspberry Pi
os.environ.setdefault('ORT_DISABLE_GPU_DEVICE_CHECK', '1')
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')  # 3 = ERROR level

from modules.wake_word import WakeWordDetector
from modules.vad import VoiceActivityDetector
from modules.stt import SpeechToText
from modules.mqtt_handler import MQTTHandler
from modules.tts import TextToSpeech


class VoiceAssistant:
    """
    Main voice assistant orchestrator.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the voice assistant.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self.wake_word: Optional[WakeWordDetector] = None
        self.vad: Optional[VoiceActivityDetector] = None
        self.stt: Optional[SpeechToText] = None
        self.mqtt: Optional[MQTTHandler] = None
        self.tts: Optional[TextToSpeech] = None
        
        # State
        self.running = False
        self.listening = False
        self.speaking = False  # Flag to indicate TTS is speaking
        self.audio_buffer = []
        self.wake_word_buffer = b''  # Buffer for accumulating audio for wake word detection
        
        self.logger.info("Voice Assistant initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file '{config_path}' not found.")
            print("Please copy 'config.example.yaml' to 'config.yaml' and configure it.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            sys.exit(1)

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )

    def initialize_modules(self) -> bool:
        """
        Initialize all modules.

        Returns:
            True if all modules initialized successfully
        """
        try:
            # Initialize Wake Word Detector
            self.logger.info("Initializing Wake Word Detector...")
            wake_config = self.config['wake_word']
            self.wake_word = WakeWordDetector(
                access_key=wake_config['access_key'],
                keyword=wake_config.get('keyword', 'hey-genio'),
                sensitivity=wake_config.get('sensitivity', 0.5),
                keyword_path=wake_config.get('keyword_path')
            )
            self.wake_word.start()

            # Initialize VAD
            self.logger.info("Initializing Voice Activity Detector...")
            vad_config = self.config['vad']
            self.vad = VoiceActivityDetector(
                sample_rate=vad_config.get('sample_rate', 16000),
                frame_duration=vad_config.get('frame_duration', 30),
                mode=vad_config.get('mode', 3)
            )

            # Initialize STT
            self.logger.info("Initializing Speech-to-Text...")
            stt_config = self.config['stt']
            self.stt = SpeechToText(
                model_path=stt_config.get('model_path', 'models/vosk/vosk-model-small-sv-rhasspy-0.15'),
                language=stt_config.get('language', 'sv'),
                sample_rate=stt_config.get('sample_rate', 16000),
                # Legacy Faster-Whisper parameters for backward compatibility
                model_size=stt_config.get('model', 'base'),
                device=stt_config.get('device', 'cpu'),
                compute_type=stt_config.get('compute_type', 'int8'),
                beam_size=stt_config.get('beam_size', 8),
                temperature=stt_config.get('temperature', 0.0),
                initial_prompt=stt_config.get('initial_prompt'),
                vad_filter=stt_config.get('vad_filter', True),
                vad_min_silence_duration=stt_config.get('vad_min_silence_duration', 500),
                condition_on_previous_text=stt_config.get('condition_on_previous_text', True)
            )
            self.stt.load_model()

            # Initialize MQTT
            self.logger.info("Initializing MQTT Handler...")
            mqtt_config = self.config['mqtt']
            self.mqtt = MQTTHandler(
                broker=mqtt_config['broker'],
                port=mqtt_config.get('port', 8883),
                username=mqtt_config.get('username', ''),
                password=mqtt_config.get('password', ''),
                topic_send=mqtt_config.get('topic_send', 'assistant/input'),
                topic_receive=mqtt_config.get('topic_receive', 'assistant/output'),
                use_tls=mqtt_config.get('use_tls', True),
                keepalive=mqtt_config.get('keepalive', 60),
                qos=mqtt_config.get('qos', 1)
            )
            if not self.mqtt.connect():
                self.logger.error("Failed to connect to MQTT broker")
                return False

            # Set MQTT response callback
            self.mqtt.set_response_callback(self._handle_mqtt_response)

            # Initialize TTS
            self.logger.info("Initializing Text-to-Speech...")
            tts_config = self.config['tts']
            self.tts = TextToSpeech(
                model_path=tts_config['model_path'],
                config_path=tts_config.get('config_path'),
                speaker=tts_config.get('speaker', 0),
                output_device=self.config['audio'].get('output_device')
            )

            self.logger.info("All modules initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error initializing modules: {e}")
            return False

    def _handle_mqtt_response(self, response: str) -> None:
        """
        Handle response from n8n via MQTT.

        Args:
            response: Response text
        """
        self.logger.info(f"Received response from n8n: {response}")
        
        # Speak the response
        if self.tts:
            # Set speaking flag to prevent wake word detection during TTS
            self.speaking = True
            try:
                self.tts.speak(response)
            finally:
                # Always clear speaking flag, even if TTS fails
                self.speaking = False
                # Clear wake word buffer to prevent TTS audio from triggering detection
                self.wake_word_buffer = b''
                self.logger.debug("TTS finished, wake word detection resumed")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback for audio input stream.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        # Convert to bytes for wake word detection
        audio_bytes = indata.tobytes()

        # Skip wake word detection if TTS is currently speaking
        # This prevents the assistant's own voice from triggering the wake word
        if self.speaking:
            return

        # If not listening, check for wake word
        if not self.listening:
            # Accumulate audio in buffer for wake word detection
            # Note: Audio callback receives chunks of size 'chunk_size' (default 1024 samples),
            # but Porcupine requires exact frames of 512 samples. We buffer and extract correctly sized frames.
            self.wake_word_buffer += audio_bytes
            
            # Process with wake word detector when we have enough data
            if self.wake_word:
                required_bytes = self.wake_word.frame_length * 2  # 2 bytes per sample (16-bit)
                
                # Safeguard: prevent buffer from growing too large (keep max 4 frames worth)
                max_buffer_size = required_bytes * 4
                if len(self.wake_word_buffer) > max_buffer_size:
                    # Keep only the most recent data
                    self.wake_word_buffer = self.wake_word_buffer[-max_buffer_size:]
                
                # Process all complete frames in the buffer
                while len(self.wake_word_buffer) >= required_bytes:
                    # Extract one frame
                    frame = self.wake_word_buffer[:required_bytes]
                    self.wake_word_buffer = self.wake_word_buffer[required_bytes:]
                    
                    # Process the frame
                    if self.wake_word.process_audio(frame):
                        self.logger.info("Wake word detected! Starting to listen...")
                        self.listening = True
                        self.audio_buffer = []
                        self.wake_word_buffer = b''  # Clear wake word buffer
                        self.vad.reset()
                        
                        # Play acknowledgment sound (optional)
                        # You could play a beep here
                        break  # Stop processing more frames once detected
        else:
            # Listening mode - use VAD and record
            if self.vad:
                is_speaking, voice_changed = self.vad.process_frame(audio_bytes)
                
                if is_speaking:
                    # Add to buffer
                    self.audio_buffer.append(indata.copy())
                elif voice_changed and not is_speaking:
                    # Voice stopped - process the recorded audio
                    self.logger.info("Voice activity stopped. Processing...")
                    self._process_recorded_audio()
                    self.listening = False
                    self.audio_buffer = []

    def _process_recorded_audio(self) -> None:
        """Process the recorded audio buffer."""
        if not self.audio_buffer:
            self.logger.warning("No audio recorded")
            return

        try:
            # Concatenate audio buffer
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            
            # Flatten if necessary
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()

            self.logger.info(f"Processing {len(audio_data)} samples")

            # Transcribe with STT
            if self.stt:
                text = self.stt.transcribe_audio(
                    audio_data,
                    sample_rate=self.config['audio']['sample_rate']
                )

                if text:
                    self.logger.info(f"User said: {text}")
                    
                    # Send to n8n via MQTT
                    if self.mqtt:
                        self.mqtt.send_message(text)
                        self.logger.info("Message sent to n8n, waiting for response...")
                else:
                    self.logger.warning("No text transcribed")

        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")

    def run(self) -> None:
        """
        Run the voice assistant main loop.
        """
        if not self.initialize_modules():
            self.logger.error("Failed to initialize modules. Exiting.")
            return

        self.running = True
        self.logger.info("Voice Assistant started. Listening for wake word...")
        self.logger.info(f"Say '{self.config['wake_word'].get('keyword', 'hey-genio')}' to activate")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Get audio configuration
            audio_config = self.config['audio']
            sample_rate = audio_config.get('sample_rate', 16000)
            channels = audio_config.get('channels', 1)
            chunk_size = audio_config.get('chunk_size', 1024)
            input_device = audio_config.get('input_device')

            # Start audio stream
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype='int16',
                blocksize=chunk_size,
                device=input_device,
                callback=self._audio_callback
            ):
                self.logger.info("Audio stream started")
                
                # Keep running until stopped
                while self.running:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self._cleanup()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}. Shutting down...")
        self.running = False

    def _cleanup(self) -> None:
        """Cleanup resources."""
        self.logger.info("Cleaning up...")
        
        if self.wake_word:
            self.wake_word.stop()
        
        if self.mqtt:
            self.mqtt.disconnect()
        
        if self.stt:
            self.stt.unload_model()
        
        self.logger.info("Cleanup complete")

    def stop(self) -> None:
        """Stop the voice assistant."""
        self.running = False


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Local Voice-Controlled AI Assistant for Raspberry Pi 5")
    print("=" * 60)
    print()

    # Check for config file
    config_path = "config.yaml"
    if not Path(config_path).exists():
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please copy 'config.example.yaml' to 'config.yaml' and configure it.")
        print()
        print("Steps:")
        print("  1. cp config.example.yaml config.yaml")
        print("  2. Edit config.yaml with your settings")
        print("  3. Run this script again")
        sys.exit(1)

    # Create and run assistant
    assistant = VoiceAssistant(config_path)
    assistant.run()


if __name__ == "__main__":
    main()
