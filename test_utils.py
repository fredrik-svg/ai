#!/usr/bin/env python3
"""
Utility script to test individual components of the voice assistant.
"""

import sys
import argparse
import logging


def test_audio_devices():
    """List available audio input and output devices."""
    print("\n=== Testing Audio Devices ===\n")
    
    try:
        import sounddevice as sd
        print("Available audio devices:")
        print(sd.query_devices())
        
        print("\nDefault input device:")
        print(sd.query_devices(kind='input'))
        
        print("\nDefault output device:")
        print(sd.query_devices(kind='output'))
        
    except ImportError:
        print("Error: sounddevice not installed. Run: pip install sounddevice")
    except Exception as e:
        print(f"Error: {e}")


def test_microphone():
    """Test microphone by recording and playing back audio."""
    print("\n=== Testing Microphone ===\n")
    
    try:
        import sounddevice as sd
        import numpy as np
        
        duration = 3  # seconds
        sample_rate = 16000
        
        print(f"Recording {duration} seconds of audio...")
        print("Speak into your microphone!")
        
        recording = sd.rec(int(duration * sample_rate), 
                          samplerate=sample_rate, 
                          channels=1, 
                          dtype='int16')
        sd.wait()
        
        print("Recording complete!")
        print(f"Recording shape: {recording.shape}")
        print(f"Max amplitude: {np.max(np.abs(recording))}")
        
        print("\nPlaying back recording...")
        sd.play(recording, sample_rate)
        sd.wait()
        
        print("Playback complete!")
        
    except ImportError:
        print("Error: sounddevice not installed. Run: pip install sounddevice")
    except Exception as e:
        print(f"Error: {e}")


def test_tts():
    """Test Text-to-Speech."""
    print("\n=== Testing Text-to-Speech ===\n")
    
    try:
        import yaml
        from modules.tts import TextToSpeech
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        tts_config = config['tts']
        
        print(f"Loading TTS model: {tts_config['model_path']}")
        tts = TextToSpeech(
            model_path=tts_config['model_path'],
            config_path=tts_config.get('config_path'),
            speaker=tts_config.get('speaker', 0)
        )
        
        test_text = "Hej, jag är din röstassistent. Det här är ett test."
        print(f"\nSpeaking: {test_text}")
        
        if tts.speak(test_text):
            print("✓ TTS test successful!")
        else:
            print("✗ TTS test failed!")
        
    except FileNotFoundError:
        print("Error: config.yaml not found. Run setup.sh first.")
    except ImportError as e:
        print(f"Error: Missing dependency. {e}")
    except Exception as e:
        print(f"Error: {e}")


def test_stt():
    """Test Speech-to-Text with a sample audio file."""
    print("\n=== Testing Speech-to-Text ===\n")
    
    try:
        import yaml
        from modules.stt import SpeechToText
        
        # Load config
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        stt_config = config['stt']
        
        print(f"Loading STT model: {stt_config['model']}")
        stt = SpeechToText(
            model_size=stt_config.get('model', 'base'),
            language=stt_config.get('language', 'sv'),
            device=stt_config.get('device', 'cpu'),
            compute_type=stt_config.get('compute_type', 'int8')
        )
        
        print("Loading model... (this may take a while on first run)")
        stt.load_model()
        
        print("\n✓ STT model loaded successfully!")
        print("Note: To test transcription, you need an audio file.")
        print("Usage: python test_utils.py --test-stt-file <audio_file.wav>")
        
    except FileNotFoundError:
        print("Error: config.yaml not found. Run setup.sh first.")
    except ImportError as e:
        print(f"Error: Missing dependency. {e}")
    except Exception as e:
        print(f"Error: {e}")


def test_stt_file(audio_file: str):
    """Test Speech-to-Text with a specific audio file."""
    print(f"\n=== Testing Speech-to-Text with {audio_file} ===\n")
    
    try:
        import yaml
        from modules.stt import SpeechToText
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        stt_config = config['stt']
        
        print("Loading STT model...")
        stt = SpeechToText(
            model_size=stt_config.get('model', 'base'),
            language=stt_config.get('language', 'sv'),
            device=stt_config.get('device', 'cpu'),
            compute_type=stt_config.get('compute_type', 'int8')
        )
        stt.load_model()
        
        print(f"Transcribing {audio_file}...")
        text = stt.transcribe_file(audio_file)
        
        print(f"\nTranscription: {text}")
        
        if text:
            print("✓ STT test successful!")
        else:
            print("✗ No transcription generated")
        
    except Exception as e:
        print(f"Error: {e}")


def test_mqtt():
    """Test MQTT connection."""
    print("\n=== Testing MQTT Connection ===\n")
    
    try:
        import yaml
        from modules.mqtt_handler import MQTTHandler
        
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        mqtt_config = config['mqtt']
        
        print(f"Connecting to MQTT broker: {mqtt_config['broker']}:{mqtt_config['port']}")
        
        mqtt = MQTTHandler(
            broker=mqtt_config['broker'],
            port=mqtt_config.get('port', 8883),
            username=mqtt_config.get('username', ''),
            password=mqtt_config.get('password', ''),
            topic_send=mqtt_config.get('topic_send', 'assistant/input'),
            topic_receive=mqtt_config.get('topic_receive', 'assistant/output'),
            use_tls=mqtt_config.get('use_tls', True)
        )
        
        if mqtt.connect():
            print("✓ Connected to MQTT broker successfully!")
            
            # Send a test message
            print("\nSending test message...")
            if mqtt.send_message("Test message from voice assistant"):
                print("✓ Test message sent!")
            
            mqtt.disconnect()
        else:
            print("✗ Failed to connect to MQTT broker")
            print("Check your config.yaml settings")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Test utilities for Voice Assistant')
    parser.add_argument('--test-audio', action='store_true', 
                       help='List audio devices')
    parser.add_argument('--test-mic', action='store_true',
                       help='Test microphone recording and playback')
    parser.add_argument('--test-tts', action='store_true',
                       help='Test Text-to-Speech')
    parser.add_argument('--test-stt', action='store_true',
                       help='Test Speech-to-Text model loading')
    parser.add_argument('--test-stt-file', type=str,
                       help='Test Speech-to-Text with audio file')
    parser.add_argument('--test-mqtt', action='store_true',
                       help='Test MQTT connection')
    parser.add_argument('--test-all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if args.test_all:
        test_audio_devices()
        test_microphone()
        test_mqtt()
        test_stt()
        test_tts()
    else:
        if args.test_audio:
            test_audio_devices()
        if args.test_mic:
            test_microphone()
        if args.test_tts:
            test_tts()
        if args.test_stt:
            test_stt()
        if args.test_stt_file:
            test_stt_file(args.test_stt_file)
        if args.test_mqtt:
            test_mqtt()
        
        if not any(vars(args).values()):
            parser.print_help()


if __name__ == "__main__":
    main()
