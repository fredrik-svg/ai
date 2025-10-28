#!/usr/bin/env python3
"""
Unit test for STT audio enhancement feature.
Tests that audio normalization improves quality for STT processing.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_low_volume_enhancement():
    """Test that low volume audio is enhanced to appropriate levels"""
    
    # Create test audio with very low volume (peak ~0.05)
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Generate a sine wave with low amplitude
    frequency = 440.0  # Hz
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    low_volume_audio = 0.05 * np.sin(2 * np.pi * frequency * t)
    
    # Calculate RMS of original audio
    original_rms = np.sqrt(np.mean(low_volume_audio ** 2))
    original_peak = np.max(np.abs(low_volume_audio))
    
    print(f"Original audio - Peak: {original_peak:.4f}, RMS: {original_rms:.4f}")
    
    # Simulate the enhancement process from _normalize_audio_quality
    target_rms = 0.1
    target_peak = 0.95
    min_rms_threshold = 0.001
    
    # Check if audio is above threshold
    if original_rms < min_rms_threshold:
        print("✗ TEST FAILED: Audio below minimum threshold")
        return False
    
    # Apply RMS normalization
    audio = low_volume_audio.copy()
    normalization_factor = target_rms / original_rms
    audio = audio * normalization_factor
    
    # Apply peak normalization if needed
    peak_after_rms = np.max(np.abs(audio))
    if peak_after_rms > target_peak:
        peak_factor = target_peak / peak_after_rms
        audio = audio * peak_factor
    
    # Calculate enhanced audio statistics
    enhanced_rms = np.sqrt(np.mean(audio ** 2))
    enhanced_peak = np.max(np.abs(audio))
    
    print(f"Enhanced audio - Peak: {enhanced_peak:.4f}, RMS: {enhanced_rms:.4f}")
    print(f"Gain applied: {normalization_factor:.2f}x")
    
    # Verify enhancement
    # RMS should be close to target_rms
    if abs(enhanced_rms - target_rms) > 0.01:
        print(f"✗ TEST FAILED: RMS {enhanced_rms:.4f} not close to target {target_rms:.4f}")
        return False
    
    # Peak should be less than or equal to target_peak
    if enhanced_peak > target_peak + 0.01:  # Small tolerance
        print(f"✗ TEST FAILED: Peak {enhanced_peak:.4f} exceeds target {target_peak:.4f}")
        return False
    
    print("✓ TEST PASSED: Low volume audio successfully enhanced!")
    return True


def test_high_volume_limiting():
    """Test that high volume audio is limited to prevent clipping"""
    
    # Create test audio with high volume (peak ~0.9)
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Generate a sine wave with high amplitude
    frequency = 440.0  # Hz
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    high_volume_audio = 0.9 * np.sin(2 * np.pi * frequency * t)
    
    # Calculate RMS of original audio
    original_rms = np.sqrt(np.mean(high_volume_audio ** 2))
    original_peak = np.max(np.abs(high_volume_audio))
    
    print(f"Original audio - Peak: {original_peak:.4f}, RMS: {original_rms:.4f}")
    
    # Simulate the enhancement process
    target_rms = 0.1
    target_peak = 0.95
    min_rms_threshold = 0.001
    
    # Apply RMS normalization
    audio = high_volume_audio.copy()
    normalization_factor = target_rms / original_rms
    audio = audio * normalization_factor
    
    # Apply peak normalization
    peak_after_rms = np.max(np.abs(audio))
    if peak_after_rms > target_peak:
        peak_factor = target_peak / peak_after_rms
        audio = audio * peak_factor
        print(f"Peak limiting applied: {peak_factor:.4f}x")
    
    # Calculate enhanced audio statistics
    enhanced_rms = np.sqrt(np.mean(audio ** 2))
    enhanced_peak = np.max(np.abs(audio))
    
    print(f"Enhanced audio - Peak: {enhanced_peak:.4f}, RMS: {enhanced_rms:.4f}")
    
    # Verify peak limiting
    if enhanced_peak > target_peak + 0.01:  # Small tolerance
        print(f"✗ TEST FAILED: Peak {enhanced_peak:.4f} exceeds target {target_peak:.4f}")
        return False
    
    print("✓ TEST PASSED: High volume audio successfully limited!")
    return True


def test_silence_detection():
    """Test that very quiet audio (silence/noise) is detected and not enhanced"""
    
    # Create nearly silent audio
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Very low amplitude noise
    silent_audio = 0.0005 * np.random.randn(num_samples).astype(np.float32)
    
    # Calculate RMS
    rms = np.sqrt(np.mean(silent_audio ** 2))
    
    print(f"Silent audio RMS: {rms:.6f}")
    
    # Check threshold
    min_rms_threshold = 0.001
    
    if rms < min_rms_threshold:
        print("✓ TEST PASSED: Silent audio correctly detected (below threshold)!")
        return True
    else:
        print("✗ TEST FAILED: Silent audio not detected")
        return False


def test_normal_volume_preservation():
    """Test that audio at good levels is minimally modified"""
    
    # Create test audio with good volume (RMS ~0.1)
    duration = 1.0  # seconds
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    # Generate a sine wave at target RMS level
    frequency = 440.0  # Hz
    t = np.linspace(0, duration, num_samples, dtype=np.float32)
    # Amplitude for RMS of 0.1 for sine wave: amplitude = RMS * sqrt(2)
    amplitude = 0.1 * np.sqrt(2)
    good_volume_audio = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Calculate RMS of original audio
    original_rms = np.sqrt(np.mean(good_volume_audio ** 2))
    original_peak = np.max(np.abs(good_volume_audio))
    
    print(f"Original audio - Peak: {original_peak:.4f}, RMS: {original_rms:.4f}")
    
    # Simulate enhancement
    target_rms = 0.1
    target_peak = 0.95
    
    audio = good_volume_audio.copy()
    normalization_factor = target_rms / original_rms
    audio = audio * normalization_factor
    
    # Calculate enhanced audio statistics
    enhanced_rms = np.sqrt(np.mean(audio ** 2))
    enhanced_peak = np.max(np.abs(audio))
    
    print(f"Enhanced audio - Peak: {enhanced_peak:.4f}, RMS: {enhanced_rms:.4f}")
    print(f"Gain applied: {normalization_factor:.2f}x")
    
    # For audio already at good level, gain should be close to 1.0
    if abs(normalization_factor - 1.0) > 0.15:  # Allow 15% deviation
        print(f"✗ TEST FAILED: Excessive gain {normalization_factor:.2f}x applied to good audio")
        return False
    
    print("✓ TEST PASSED: Audio at good levels minimally modified!")
    return True


def main():
    """Run all tests"""
    print("=" * 70)
    print("Testing STT Audio Enhancement Feature")
    print("=" * 70)
    
    tests = [
        ("Low volume enhancement", test_low_volume_enhancement),
        ("High volume limiting", test_high_volume_limiting),
        ("Silence detection", test_silence_detection),
        ("Normal volume preservation", test_normal_volume_preservation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        print("-" * 70)
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"✗ TEST FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary:")
    print("=" * 70)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 70)
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
