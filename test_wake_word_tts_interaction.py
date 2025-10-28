#!/usr/bin/env python3
"""
Test to verify wake word detection is properly disabled during TTS playback.
This test simulates the scenario where TTS could trigger wake word detection.
"""

import sys
import os
import time
import logging
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock modules that aren't installed
sys.modules['sounddevice'] = Mock()
sys.modules['pvporcupine'] = Mock()
sys.modules['webrtcvad'] = Mock()
sys.modules['vosk'] = Mock()
sys.modules['paho'] = Mock()
sys.modules['paho.mqtt'] = Mock()
sys.modules['paho.mqtt.client'] = Mock()
sys.modules['soundfile'] = Mock()

from main import VoiceAssistant

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_speaking_flag_prevents_wake_word_detection():
    """
    Test that the speaking flag properly prevents wake word detection
    during TTS playback.
    """
    print("=" * 70)
    print("Test: Speaking flag prevents wake word detection")
    print("=" * 70)
    
    # Create a mock config
    mock_config = {
        'wake_word': {
            'access_key': 'test_key',
            'keyword': 'test',
            'sensitivity': 0.5,
            'keyword_path': None
        },
        'vad': {
            'sample_rate': 16000,
            'frame_duration': 30,
            'mode': 3
        },
        'stt': {
            'model_path': 'test/path',
            'language': 'sv',
            'sample_rate': 16000,
            'model': 'base',
            'device': 'cpu',
            'compute_type': 'int8',
            'beam_size': 8,
            'temperature': 0.0,
            'vad_filter': True,
            'vad_min_silence_duration': 500,
            'condition_on_previous_text': True
        },
        'mqtt': {
            'broker': 'test.broker',
            'port': 8883,
            'username': 'test',
            'password': 'test',
            'topic_send': 'test/input',
            'topic_receive': 'test/output',
            'use_tls': True,
            'keepalive': 60,
            'qos': 1
        },
        'tts': {
            'model_path': 'test/model.onnx',
            'config_path': 'test/model.json',
            'speaker': 0
        },
        'audio': {
            'input_device': None,
            'output_device': None,
            'sample_rate': 16000,
            'channels': 1,
            'chunk_size': 1024
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    # Patch the _load_config method
    with patch.object(VoiceAssistant, '_load_config', return_value=mock_config):
        assistant = VoiceAssistant()
    
    # Mock the TTS speak method to simulate speaking
    assistant.tts = Mock()
    assistant.tts.speak = Mock(return_value=True)
    
    # Initialize state
    assistant.speaking = False
    assistant.listening = False
    assistant.wake_word_buffer = b''
    
    # Create mock audio data
    mock_audio = np.zeros((512, 1), dtype=np.int16)
    
    print("\n1. Testing normal wake word detection (not speaking)")
    print("-" * 70)
    
    # Test 1: Audio callback should process wake word when not speaking
    assistant.speaking = False
    initial_buffer_len = len(assistant.wake_word_buffer)
    assistant._audio_callback(mock_audio, 512, None, None)
    
    # Buffer should have received audio data
    if len(assistant.wake_word_buffer) > initial_buffer_len:
        print("✓ Audio callback processes wake word detection when not speaking")
        test1_pass = True
    else:
        print("✗ FAIL: Audio callback did not process audio when not speaking")
        test1_pass = False
    
    print("\n2. Testing wake word detection blocked during TTS")
    print("-" * 70)
    
    # Test 2: Audio callback should skip wake word detection when speaking
    assistant.speaking = True
    buffer_before_speaking = len(assistant.wake_word_buffer)
    assistant._audio_callback(mock_audio, 512, None, None)
    buffer_after_speaking = len(assistant.wake_word_buffer)
    
    # Buffer should NOT have grown
    if buffer_before_speaking == buffer_after_speaking:
        print("✓ Audio callback skips wake word detection when speaking")
        test2_pass = True
    else:
        print("✗ FAIL: Audio callback processed audio during TTS playback")
        test2_pass = False
    
    print("\n3. Testing MQTT response handler sets speaking flag")
    print("-" * 70)
    
    # Test 3: _handle_mqtt_response should set and clear speaking flag
    assistant.speaking = False
    assistant.wake_word_buffer = b'some_old_audio_data'
    
    # Mock TTS to track when it's called and verify speaking flag
    call_order = []
    speaking_flag_during_call = []
    
    def mock_speak(text):
        call_order.append('speak_start')
        # Record the speaking flag state during the call
        speaking_flag_during_call.append(assistant.speaking)
        call_order.append('speak_end')
        return True
    
    assistant.tts.speak = mock_speak
    
    # Call the handler
    assistant._handle_mqtt_response("Test response")
    
    # Verify speaking flag is cleared after TTS
    if not assistant.speaking:
        print("✓ Speaking flag is cleared after TTS")
        test3a_pass = True
    else:
        print("✗ FAIL: Speaking flag not cleared after TTS")
        test3a_pass = False
    
    # Verify wake word buffer is cleared after TTS
    if len(assistant.wake_word_buffer) == 0:
        print("✓ Wake word buffer cleared after TTS")
        test3b_pass = True
    else:
        print("✗ FAIL: Wake word buffer not cleared after TTS")
        test3b_pass = False
    
    # Verify call order and that speaking flag was set during TTS
    if call_order == ['speak_start', 'speak_end'] and speaking_flag_during_call == [True]:
        print("✓ TTS was called with speaking flag set")
        test3c_pass = True
    else:
        print("✗ FAIL: Speaking flag was not properly set during TTS call")
        test3c_pass = False
    
    print("\n4. Testing speaking flag cleared even if TTS fails")
    print("-" * 70)
    
    # Test 4: Speaking flag should be cleared even if TTS raises exception
    assistant.speaking = False
    
    def mock_speak_with_error(text):
        raise RuntimeError("TTS error")
    
    assistant.tts.speak = mock_speak_with_error
    
    try:
        assistant._handle_mqtt_response("Test response")
    except RuntimeError as e:
        # Expected to raise RuntimeError from TTS
        if str(e) == "TTS error":
            pass  # This is the expected error
        else:
            print(f"✗ FAIL: Unexpected error: {e}")
            test4_pass = False
            return test4_pass
    
    # Verify speaking flag is still cleared
    if not assistant.speaking:
        print("✓ Speaking flag cleared even when TTS raises exception")
        test4_pass = True
    else:
        print("✗ FAIL: Speaking flag not cleared after TTS exception")
        test4_pass = False
    
    # Final results
    print("\n" + "=" * 70)
    all_tests = [test1_pass, test2_pass, test3a_pass, test3b_pass, test3c_pass, test4_pass]
    
    if all(all_tests):
        print("✓ ALL TESTS PASSED!")
        print("\nThe wake word detection correctly:")
        print("  • Processes audio when not speaking")
        print("  • Blocks detection during TTS playback")
        print("  • Clears buffer after TTS to prevent interference")
        print("  • Handles TTS errors gracefully")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  Passed: {sum(all_tests)}/{len(all_tests)}")
        return 1

def main():
    """Run the test"""
    print("\nWake Word TTS Interaction Test")
    print()
    
    try:
        return test_speaking_flag_prevents_wake_word_detection()
    except Exception as e:
        print(f"\n✗ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
