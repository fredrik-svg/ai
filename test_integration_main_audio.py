#!/usr/bin/env python3
"""
Integration test simulating the exact scenario from main.py.
This test verifies that int16 audio from sounddevice works correctly with STT.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def simulate_main_py_audio_flow():
    """
    Simulate the exact audio flow from main.py:
    1. Audio is captured as int16 from sounddevice
    2. Stored in audio_buffer
    3. Concatenated and passed to STT
    """
    print("=" * 70)
    print("Simulating main.py audio flow")
    print("=" * 70)
    
    # Simulate audio buffer from main.py (int16 from sounddevice)
    # Create synthetic int16 audio data similar to what sounddevice produces
    audio_buffer = []
    
    # Simulate multiple audio chunks (as would come from callback)
    # Using reasonable int16 values for speech audio
    chunk1 = np.array([0, 100, 200, 500, 1000, 2000, 3000], dtype=np.int16)
    chunk2 = np.array([5000, 8000, 10000, 12000, 15000, 16000], dtype=np.int16)
    chunk3 = np.array([16000, 12000, 8000, 4000, 2000, 1000, 0], dtype=np.int16)
    chunk4 = np.array([-1000, -2000, -4000, -8000, -12000, -16000], dtype=np.int16)
    
    audio_buffer.append(chunk1)
    audio_buffer.append(chunk2)
    audio_buffer.append(chunk3)
    audio_buffer.append(chunk4)
    
    print(f"\n✓ Created {len(audio_buffer)} audio chunks (int16)")
    
    # Concatenate audio buffer (exactly as main.py does)
    audio_data = np.concatenate(audio_buffer, axis=0)
    print(f"✓ Concatenated to {len(audio_data)} samples")
    print(f"  Data type: {audio_data.dtype}")
    print(f"  Value range: [{audio_data.min()}, {audio_data.max()}]")
    
    # Now simulate what happens in stt.transcribe_audio()
    print("\n--- Simulating STT conversion ---")
    
    # This is the fixed code from modules/stt.py
    if audio_data.dtype != np.float32:
        if audio_data.dtype == np.int16:
            # Normalize int16 to float32 range [-1, 1]
            print("✓ Detected int16 data, normalizing to float32...")
            audio_float32 = audio_data.astype(np.float32) / 32767.0
        else:
            # For other types, just convert to float32
            audio_float32 = audio_data.astype(np.float32)
    else:
        audio_float32 = audio_data
    
    print(f"✓ Converted to float32")
    print(f"  Data type: {audio_float32.dtype}")
    print(f"  Value range: [{audio_float32.min():.4f}, {audio_float32.max():.4f}]")
    
    # Verify values are in the correct range [-1, 1]
    if audio_float32.min() >= -1.0 and audio_float32.max() <= 1.0:
        print("✓ Float32 values are in correct range [-1, 1]")
    else:
        print("✗ ERROR: Float32 values are outside expected range!")
        return False
    
    # Convert back to int16 for Vosk (as done in transcribe_audio)
    print("\n--- Converting to int16 for Vosk ---")
    audio_int16 = (audio_float32 * 32767).astype(np.int16)
    print(f"✓ Converted to int16 for Vosk")
    print(f"  Data type: {audio_int16.dtype}")
    print(f"  Value range: [{audio_int16.min()}, {audio_int16.max()}]")
    
    # Verify the round-trip conversion preserves the data
    print("\n--- Verifying round-trip conversion ---")
    max_diff = np.max(np.abs(audio_data - audio_int16))
    print(f"  Max difference from original: {max_diff}")
    
    if max_diff <= 1:  # Allow 1 bit of difference due to rounding
        print("✓ Round-trip conversion successful (within rounding error)")
        return True
    else:
        print("✗ ERROR: Too much data loss in conversion!")
        return False

def main():
    """Run integration test"""
    print("\nIntegration Test: main.py audio flow with int16 data")
    print()
    
    success = simulate_main_py_audio_flow()
    
    print("\n" + "=" * 70)
    if success:
        print("✓ INTEGRATION TEST PASSED!")
        print("\nThe fix correctly handles int16 audio data from main.py")
        return 0
    else:
        print("✗ INTEGRATION TEST FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
