#!/usr/bin/env python3
"""
Test to verify that listening mode has a timeout and doesn't get stuck.
"""

import sys
import os
import time
import logging
from unittest.mock import Mock, MagicMock, patch

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
sys.modules['numpy'] = Mock()

# Mock numpy array
class MockArray:
    def __init__(self, data, dtype=None, shape=None):
        self.data = data
        self.dtype = dtype
        # If shape not provided, determine from data
        if shape is None:
            if isinstance(data, list):
                self.shape = (len(data),)
            else:
                self.shape = (1,)
        else:
            self.shape = shape
    
    def tobytes(self):
        return b'\x00' * 1024
    
    def copy(self):
        return MockArray(self.data, self.dtype, self.shape)

mock_np = sys.modules['numpy']

def mock_zeros(shape, dtype=None):
    # Handle both tuple and int shapes
    if isinstance(shape, tuple):
        # Calculate total size for multi-dimensional arrays
        size = 1
        for dim in shape:
            size *= dim
        return MockArray([0] * size, dtype, shape)
    else:
        return MockArray([0] * shape, dtype, (shape,))

mock_np.zeros = mock_zeros
mock_np.int16 = 'int16'

from main import VoiceAssistant

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_listening_timeout():
    """
    Test that listening mode has a timeout and automatically resets
    to wake word detection if VAD doesn't detect voice stop.
    """
    print("=" * 70)
    print("Test: Listening Mode Timeout")
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
    
    # Initialize state
    assistant.speaking = False
    assistant.listening = True  # Simulate being stuck in listening mode
    assistant.listening_start_time = time.time() - 11.0  # 11 seconds ago (past timeout)
    assistant.audio_buffer = []
    assistant.listening_timeout = 10.0
    
    # Mock VAD to always return False (no speech detected)
    mock_vad = Mock()
    mock_vad.process_frame = Mock(return_value=(False, False))
    mock_vad.reset = Mock()
    assistant.vad = mock_vad
    
    # Mock STT
    assistant.stt = Mock()
    
    # Create mock audio data
    mock_audio = MockArray([0] * 512, dtype='int16')
    
    print("\n1. Testing timeout detection and reset")
    print("-" * 70)
    
    # Call audio callback - should detect timeout and reset
    assistant._audio_callback(mock_audio, 512, None, None)
    
    # Check that listening mode was reset
    test1_pass = True
    if not assistant.listening:
        print("✓ Listening mode reset to False")
    else:
        print("✗ FAIL: Listening mode still True")
        test1_pass = False
    
    if assistant.listening_start_time is None:
        print("✓ Listening start time cleared")
    else:
        print("✗ FAIL: Listening start time not cleared")
        test1_pass = False
    
    if len(assistant.audio_buffer) == 0:
        print("✓ Audio buffer cleared")
    else:
        print("✗ FAIL: Audio buffer not cleared")
        test1_pass = False
    
    if mock_vad.reset.called:
        print("✓ VAD reset called")
    else:
        print("✗ FAIL: VAD reset not called")
        test1_pass = False
    
    print("\n2. Testing timeout with buffered audio")
    print("-" * 70)
    
    # Reset and add audio to buffer
    assistant.listening = True
    assistant.listening_start_time = time.time() - 11.0
    assistant.audio_buffer = [mock_audio.copy(), mock_audio.copy()]
    
    # Mock process_recorded_audio to verify it's called
    process_called = []
    original_process = assistant._process_recorded_audio
    assistant._process_recorded_audio = lambda: process_called.append(True)
    
    # Call audio callback
    assistant._audio_callback(mock_audio, 512, None, None)
    
    test2_pass = True
    if len(process_called) > 0:
        print("✓ Process recorded audio called before timeout reset")
    else:
        print("✗ FAIL: Process recorded audio not called")
        test2_pass = False
    
    if not assistant.listening:
        print("✓ Listening mode reset after processing audio")
    else:
        print("✗ FAIL: Listening mode still True")
        test2_pass = False
    
    # Restore original method
    assistant._process_recorded_audio = original_process
    
    print("\n3. Testing no timeout when time is within limit")
    print("-" * 70)
    
    # Reset with recent start time (5 seconds ago, within 10 second timeout)
    assistant.listening = True
    assistant.listening_start_time = time.time() - 5.0
    assistant.audio_buffer = []
    initial_listening = assistant.listening
    
    # Call audio callback
    assistant._audio_callback(mock_audio, 512, None, None)
    
    test3_pass = True
    if assistant.listening == initial_listening:
        print("✓ Listening mode unchanged when within timeout")
    else:
        print("✗ FAIL: Listening mode changed when it shouldn't")
        test3_pass = False
    
    if assistant.listening_start_time is not None:
        print("✓ Listening start time preserved")
    else:
        print("✗ FAIL: Listening start time cleared too early")
        test3_pass = False
    
    # Final results
    print("\n" + "=" * 70)
    all_tests = [test1_pass, test2_pass, test3_pass]
    
    if all(all_tests):
        print("✓ ALL TESTS PASSED!")
        print("\nListening mode timeout correctly:")
        print("  • Detects when timeout exceeded")
        print("  • Resets to wake word detection mode")
        print("  • Processes any buffered audio before reset")
        print("  • Clears all state variables")
        print("  • Does not timeout when within time limit")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  Passed: {sum(all_tests)}/{len(all_tests)}")
        return 1

def main():
    """Run the test"""
    print("\nListening Mode Timeout Test")
    print()
    
    try:
        return test_listening_timeout()
    except Exception as e:
        print(f"\n✗ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
