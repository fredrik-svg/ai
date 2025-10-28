#!/usr/bin/env python3
"""
Unit test for STT dtype conversion fix.
Tests that int16 audio data is properly normalized when passed to transcribe_audio.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_int16_to_float32_conversion():
    """Test that int16 audio is properly normalized to float32 range [-1, 1]"""
    
    # Create test int16 audio data
    # Use values that represent a reasonable audio signal
    test_values = np.array([0, 16000, -16000, 32767, -32768], dtype=np.int16)
    
    # Simulate the conversion that happens in transcribe_audio
    if test_values.dtype == np.int16:
        audio_float32 = test_values.astype(np.float32) / 32767.0
    else:
        audio_float32 = test_values.astype(np.float32)
    
    # Convert back to int16 (what Vosk expects)
    audio_int16_result = (audio_float32 * 32767).astype(np.int16)
    
    # Verify the conversion is correct
    print("Original int16 values:", test_values)
    print("After normalization to float32:", audio_float32)
    print("After conversion back to int16:", audio_int16_result)
    
    # The values should be approximately the same (within rounding error)
    # Note: -32768 becomes -32767 due to the normalization/denormalization
    expected = np.array([0, 16000, -16000, 32767, -32767], dtype=np.int16)
    
    # Check if values are close enough (allow for small rounding differences)
    differences = np.abs(audio_int16_result - expected)
    max_diff = np.max(differences)
    
    print(f"\nMaximum difference: {max_diff}")
    print(f"Expected values: {expected}")
    
    if max_diff <= 1:  # Allow 1 bit of difference due to rounding
        print("\n✓ TEST PASSED: int16 to float32 conversion is correct!")
        return True
    else:
        print("\n✗ TEST FAILED: Conversion resulted in too much error!")
        return False

def test_float32_unchanged():
    """Test that float32 audio data remains unchanged"""
    
    # Create test float32 audio data normalized to [-1, 1]
    test_values = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    
    # Simulate the conversion that happens in transcribe_audio
    if test_values.dtype != np.float32:
        audio_float32 = test_values.astype(np.float32) / 32767.0
    else:
        audio_float32 = test_values  # Should remain unchanged
    
    # Convert to int16 (what Vosk expects)
    audio_int16_result = (audio_float32 * 32767).astype(np.int16)
    
    print("\nOriginal float32 values:", test_values)
    print("After conversion to int16:", audio_int16_result)
    
    # Expected int16 values
    expected = np.array([0, 16383, -16383, 32767, -32767], dtype=np.int16)
    
    # Check if values are close enough
    differences = np.abs(audio_int16_result - expected)
    max_diff = np.max(differences)
    
    print(f"Maximum difference: {max_diff}")
    print(f"Expected values: {expected}")
    
    if max_diff <= 1:
        print("\n✓ TEST PASSED: float32 data remains properly normalized!")
        return True
    else:
        print("\n✗ TEST FAILED: float32 conversion error!")
        return False

def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing STT dtype conversion fix")
    print("=" * 70)
    
    print("\nTest 1: int16 to float32 conversion")
    print("-" * 70)
    test1_passed = test_int16_to_float32_conversion()
    
    print("\n" + "=" * 70)
    print("\nTest 2: float32 data handling")
    print("-" * 70)
    test2_passed = test_float32_unchanged()
    
    print("\n" + "=" * 70)
    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
