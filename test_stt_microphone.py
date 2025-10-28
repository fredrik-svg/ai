#!/usr/bin/env python3
"""
Test script for STT (Speech-To-Text) with microphone.
This script tests the STT functionality in isolation without other modules.

Usage:
    python test_stt_microphone.py [OPTIONS]

Options:
    --model-path PATH    Path to Vosk model directory (default: from config.yaml or models/vosk/vosk-model-small-sv-rhasspy-0.15)
    --sample-rate RATE   Sample rate in Hz (default: 16000)
    --duration SECONDS   Recording duration in seconds (default: 5)
    --device INDEX       Audio input device index (default: system default)
    --language LANG      Language code (default: sv)
    --list-devices       List available audio devices and exit
    --save-audio         Save the recorded audio to a WAV file for debugging
    --play-audio         Play back the recorded audio after transcription

Example:
    # Test with default settings
    python test_stt_microphone.py

    # Test with 10 second recording
    python test_stt_microphone.py --duration 10

    # List available audio devices
    python test_stt_microphone.py --list-devices

    # Use specific device
    python test_stt_microphone.py --device 2
    
    # Save and play back audio for debugging
    python test_stt_microphone.py --save-audio --play-audio
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np
import sounddevice as sd
import yaml

# Configure ONNX Runtime environment
os.environ.setdefault('ORT_DISABLE_GPU_DEVICE_CHECK', '1')
os.environ.setdefault('ORT_LOGGING_LEVEL', '3')

from modules.stt import SpeechToText


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def list_audio_devices():
    """List all available audio input devices."""
    print("\n" + "=" * 70)
    print("Available Audio Input Devices:")
    print("=" * 70)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default = " (DEFAULT)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']}{default}")
            print(f"      Channels: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']} Hz")
    print("=" * 70 + "\n")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file if it exists."""
    try:
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load config file: {e}")
    return {}


def record_audio(duration: int, sample_rate: int, device: int = None) -> np.ndarray:
    """
    Record audio from microphone.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        device: Audio input device index (None for default)
    
    Returns:
        Audio data as numpy array
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print(f"Recording for {duration} seconds...")
    print("Please speak clearly into the microphone.")
    print("=" * 70 + "\n")
    
    try:
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32',
            device=device
        )
        sd.wait()  # Wait for recording to complete
        
        print("\n" + "=" * 70)
        print("Recording complete!")
        print("=" * 70 + "\n")
        
        return audio_data.flatten()
    
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        raise


def test_stt_microphone(
    model_path: str,
    sample_rate: int = 16000,
    duration: int = 5,
    device: int = None,
    language: str = "sv",
    save_audio: bool = False,
    play_audio: bool = False
):
    """
    Test STT with microphone input.
    
    Args:
        model_path: Path to Vosk model directory
        sample_rate: Sample rate in Hz
        duration: Recording duration in seconds
        device: Audio input device index
        language: Language code
        save_audio: If True, save the recorded audio to a WAV file
        play_audio: If True, play back the recorded audio after transcription
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 70)
    print("  STT (Speech-To-Text) Microphone Test")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Language: {language}")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Recording Duration: {duration} seconds")
    if device is not None:
        device_info = sd.query_devices(device)
        print(f"  Input Device: [{device}] {device_info['name']}")
    else:
        default_device = sd.default.device[0]
        device_info = sd.query_devices(default_device)
        print(f"  Input Device: [default] {device_info['name']}")
    if save_audio:
        print(f"  Save Audio: Yes")
    if play_audio:
        print(f"  Play Audio: Yes")
    print("=" * 70 + "\n")
    
    # Initialize STT
    print("Initializing STT module...")
    try:
        stt = SpeechToText(
            model_path=model_path,
            language=language,
            sample_rate=sample_rate,
            save_recordings=save_audio,
            recordings_dir="test_recordings"
        )
        stt.load_model()
        print("✓ STT module initialized successfully\n")
    except Exception as e:
        logger.error(f"Failed to initialize STT: {e}")
        print(f"\n✗ ERROR: Failed to initialize STT module")
        print(f"  {e}")
        print(f"\nPlease ensure the Vosk model is downloaded to: {model_path}")
        print(f"Download from: https://alphacephei.com/vosk/models")
        return False
    
    # Record audio
    try:
        audio_data = record_audio(duration, sample_rate, device)
    except Exception as e:
        logger.error(f"Failed to record audio: {e}")
        print(f"\n✗ ERROR: Failed to record audio")
        print(f"  {e}")
        return False
    
    # Save audio file path for playback
    saved_audio_path = None
    
    # Transcribe audio
    print("Transcribing audio...")
    try:
        transcription = stt.transcribe_audio(audio_data, sample_rate)
        
        # If save_audio is enabled but we want the path for playback
        if save_audio or play_audio:
            # Save audio manually to get the path
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_audio_path = stt.save_audio_to_wav(audio_data, sample_rate, f"test_{timestamp}.wav")
        
        print("\n" + "=" * 70)
        print("TRANSCRIPTION RESULT:")
        print("=" * 70)
        if transcription:
            print(f"  \"{transcription}\"")
            print("=" * 70)
            
            # Play back audio if requested
            if play_audio and saved_audio_path and os.path.exists(saved_audio_path):
                print("\nPlaying back recorded audio...")
                try:
                    # Read and play the WAV file
                    import wave
                    with wave.open(saved_audio_path, 'rb') as wf:
                        audio_data_playback = np.frombuffer(
                            wf.readframes(wf.getnframes()), 
                            dtype=np.int16
                        ).astype(np.float32) / 32767.0
                        
                        sd.play(audio_data_playback, sample_rate)
                        sd.wait()
                        print("✓ Playback complete")
                except Exception as e:
                    logger.error(f"Failed to play back audio: {e}")
                    print(f"⚠ WARNING: Could not play back audio: {e}")
            
            print("\n✓ STT test completed successfully!")
            if saved_audio_path:
                print(f"\n📁 Audio saved to: {saved_audio_path}")
                print(f"   You can play it back with: aplay {saved_audio_path}")
            return True
        else:
            print("  (no text detected)")
            print("=" * 70)
            print("\n⚠ WARNING: No text was transcribed")
            print("  This could mean:")
            print("  - No speech was detected in the audio")
            print("  - The microphone volume is too low")
            print("  - Background noise is too high")
            print("  - The language model doesn't match the spoken language")
            
            if saved_audio_path:
                print(f"\n📁 Audio saved to: {saved_audio_path}")
                print(f"   You can play it back to debug: aplay {saved_audio_path}")
            
            return False
    
    except Exception as e:
        logger.error(f"Failed to transcribe audio: {e}")
        print(f"\n✗ ERROR: Failed to transcribe audio")
        print(f"  {e}")
        return False
    
    finally:
        # Cleanup
        stt.unload_model()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test STT (Speech-To-Text) functionality with microphone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings
  %(prog)s

  # Test with 10 second recording
  %(prog)s --duration 10

  # List available audio devices
  %(prog)s --list-devices

  # Use specific device
  %(prog)s --device 2

  # Use custom model path
  %(prog)s --model-path models/vosk/vosk-model-sv-se-0.22
  
  # Save audio to file for debugging
  %(prog)s --save-audio
  
  # Play back audio after recording
  %(prog)s --play-audio
        """
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to Vosk model directory (default: from config.yaml or models/vosk/vosk-model-small-sv-rhasspy-0.15)'
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=16000,
        help='Sample rate in Hz (default: 16000)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=5,
        help='Recording duration in seconds (default: 5)'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='Audio input device index (default: system default)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='sv',
        help='Language code (default: sv for Swedish)'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    parser.add_argument(
        '--save-audio',
        action='store_true',
        help='Save the recorded audio to a WAV file for debugging'
    )
    parser.add_argument(
        '--play-audio',
        action='store_true',
        help='Play back the recorded audio after transcription (implies --save-audio)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return 0
    
    # Load config for default values
    config = load_config()
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    elif config and 'stt' in config and 'model_path' in config['stt']:
        model_path = config['stt']['model_path']
    else:
        model_path = "models/vosk/vosk-model-small-sv-rhasspy-0.15"
    
    # Run test
    success = test_stt_microphone(
        model_path=model_path,
        sample_rate=args.sample_rate,
        duration=args.duration,
        device=args.device,
        language=args.language,
        save_audio=args.save_audio or args.play_audio,
        play_audio=args.play_audio
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
