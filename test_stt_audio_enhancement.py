#!/usr/bin/env python3
"""
Integration test for STT audio enhancement.
Creates mock audio data and verifies enhancement is applied correctly by the STT module.
"""

import numpy as np
import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.stt import SpeechToText

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def test_stt_with_low_volume_audio():
    """Test STT module processes low volume audio with enhancement"""
    print("\n" + "=" * 70)
    print("Test: STT with Low Volume Audio Enhancement")
    print("=" * 70)
    
    # Create STT instance with enhancement enabled
    stt = SpeechToText(
        model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
        sample_rate=16000,
        enable_audio_enhancement=True,
        target_rms=0.1,
        target_peak=0.95
    )
    
    # Create low volume test audio
    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    frequency = 440.0
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    low_volume_audio = 0.05 * np.sin(2 * np.pi * frequency * t)
    
    original_rms = np.sqrt(np.mean(low_volume_audio ** 2))
    print(f"Original audio RMS: {original_rms:.4f}")
    
    # Apply enhancement through the private method (for testing)
    enhanced_audio = stt._normalize_audio_quality(low_volume_audio)
    
    enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
    print(f"Enhanced audio RMS: {enhanced_rms:.4f}")
    
    # Verify enhancement occurred
    if enhanced_rms < 0.09 or enhanced_rms > 0.11:  # Should be ~0.1
        print(f"✗ TEST FAILED: Enhanced RMS {enhanced_rms:.4f} not close to target 0.1")
        return False
    
    print("✓ TEST PASSED: Low volume audio enhanced correctly!")
    return True


def test_stt_with_disabled_enhancement():
    """Test STT module does not enhance when disabled"""
    print("\n" + "=" * 70)
    print("Test: STT with Enhancement Disabled")
    print("=" * 70)
    
    # Create STT instance with enhancement disabled
    stt = SpeechToText(
        model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
        sample_rate=16000,
        enable_audio_enhancement=False
    )
    
    # Create low volume test audio
    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    frequency = 440.0
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    low_volume_audio = 0.05 * np.sin(2 * np.pi * frequency * t)
    
    original_rms = np.sqrt(np.mean(low_volume_audio ** 2))
    print(f"Original audio RMS: {original_rms:.4f}")
    
    # Apply enhancement (should do nothing when disabled)
    enhanced_audio = stt._normalize_audio_quality(low_volume_audio)
    
    enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
    print(f"Enhanced audio RMS: {enhanced_rms:.4f}")
    
    # Verify no enhancement occurred (should be same as original)
    if abs(enhanced_rms - original_rms) > 0.001:
        print(f"✗ TEST FAILED: Audio was modified when enhancement disabled")
        return False
    
    print("✓ TEST PASSED: Enhancement correctly disabled!")
    return True


def test_stt_with_custom_parameters():
    """Test STT module with custom enhancement parameters"""
    print("\n" + "=" * 70)
    print("Test: STT with Custom Enhancement Parameters")
    print("=" * 70)
    
    # Create STT instance with custom parameters
    custom_target_rms = 0.15
    custom_target_peak = 0.90
    
    stt = SpeechToText(
        model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
        sample_rate=16000,
        enable_audio_enhancement=True,
        target_rms=custom_target_rms,
        target_peak=custom_target_peak
    )
    
    # Create test audio
    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    frequency = 440.0
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    test_audio = 0.05 * np.sin(2 * np.pi * frequency * t)
    
    print(f"Target RMS: {custom_target_rms}, Target Peak: {custom_target_peak}")
    
    # Apply enhancement
    enhanced_audio = stt._normalize_audio_quality(test_audio)
    
    enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
    enhanced_peak = np.max(np.abs(enhanced_audio))
    
    print(f"Enhanced audio - RMS: {enhanced_rms:.4f}, Peak: {enhanced_peak:.4f}")
    
    # Verify custom parameters were applied
    if abs(enhanced_rms - custom_target_rms) > 0.02:
        print(f"✗ TEST FAILED: RMS {enhanced_rms:.4f} not close to custom target {custom_target_rms}")
        return False
    
    if enhanced_peak > custom_target_peak + 0.01:
        print(f"✗ TEST FAILED: Peak {enhanced_peak:.4f} exceeds custom target {custom_target_peak}")
        return False
    
    print("✓ TEST PASSED: Custom parameters applied correctly!")
    return True


def test_int16_input_conversion():
    """Test that int16 input is properly converted and enhanced"""
    print("\n" + "=" * 70)
    print("Test: int16 Audio Input Conversion and Enhancement")
    print("=" * 70)
    
    # Create STT instance
    stt = SpeechToText(
        model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
        sample_rate=16000,
        enable_audio_enhancement=True
    )
    
    # Create int16 audio (low volume)
    duration = 1.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    frequency = 440.0
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    audio_float = 0.05 * np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio_float * 32767).astype(np.int16)
    
    print(f"Input: int16 audio with {num_samples} samples")
    
    # The transcribe_audio method should handle int16 input
    # For this test, we'll simulate the conversion it does
    if audio_int16.dtype == np.int16:
        audio_converted = audio_int16.astype(np.float32) / 32767.0
    
    original_rms = np.sqrt(np.mean(audio_converted ** 2))
    print(f"After conversion to float32, RMS: {original_rms:.4f}")
    
    # Apply enhancement
    enhanced_audio = stt._normalize_audio_quality(audio_converted)
    enhanced_rms = np.sqrt(np.mean(enhanced_audio ** 2))
    
    print(f"After enhancement, RMS: {enhanced_rms:.4f}")
    
    # Verify the int16 conversion and enhancement worked
    if abs(enhanced_rms - 0.1) > 0.02:
        print(f"✗ TEST FAILED: Enhanced RMS {enhanced_rms:.4f} not close to target 0.1")
        return False
    
    print("✓ TEST PASSED: int16 input converted and enhanced correctly!")
    return True


def main():
    """Run all integration tests"""
    print("=" * 70)
    print("STT Audio Enhancement Integration Tests")
    print("=" * 70)
    
    tests = [
        test_stt_with_low_volume_audio,
        test_stt_with_disabled_enhancement,
        test_stt_with_custom_parameters,
        test_int16_input_conversion,
    ]
    
    results = []
    for test_func in tests:
        try:
            passed = test_func()
            results.append(passed)
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    passed_count = sum(results)
    total_count = len(results)
    print(f"Passed: {passed_count}/{total_count}")
    
    if all(results):
        print("\n✓ ALL INTEGRATION TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME INTEGRATION TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
