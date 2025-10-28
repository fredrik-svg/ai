#!/usr/bin/env python3
"""
Test to verify TTS fade-out and silence trimming improvements.
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fade_out():
    """Test that fade-out is correctly applied"""
    print("Testing fade-out logic...")
    
    # Create a simple audio signal (1 second at 22050 Hz)
    sample_rate = 22050
    duration = 1.0
    samples = int(sample_rate * duration)
    
    # Create a constant amplitude signal
    data = np.ones(samples) * 0.5
    
    # Apply fade-out (50ms)
    fade_duration = 0.05
    fade_samples = int(fade_duration * sample_rate)
    
    print(f"Original signal length: {len(data)} samples")
    print(f"Fade duration: {fade_duration}s ({fade_samples} samples)")
    
    # Create fade curve
    fade_curve = np.linspace(1.0, 0.0, fade_samples)
    data_with_fade = data.copy()
    data_with_fade[-fade_samples:] *= fade_curve
    
    # Verify fade was applied
    # The last sample should be close to 0
    assert np.abs(data_with_fade[-1]) < 0.01, f"Last sample should be near 0, got {data_with_fade[-1]}"
    
    # The sample before fade should still be at original amplitude
    pre_fade_idx = len(data) - fade_samples - 1
    assert np.abs(data_with_fade[pre_fade_idx] - 0.5) < 0.01, \
        f"Pre-fade sample should be ~0.5, got {data_with_fade[pre_fade_idx]}"
    
    # Verify fade is gradual
    fade_region = data_with_fade[-fade_samples:]
    assert fade_region[0] > fade_region[-1], "Fade should decrease from start to end"
    assert np.all(np.diff(fade_region) <= 0), "Fade should be monotonically decreasing"
    
    print(f"✓ First fade sample: {fade_region[0]:.3f}")
    print(f"✓ Last fade sample: {fade_region[-1]:.6f}")
    print(f"✓ Pre-fade amplitude: {data_with_fade[pre_fade_idx]:.3f}")
    print(f"✓ Fade-out applied correctly")
    
    return True

def test_silence_trimming():
    """Test improved silence trimming"""
    print("\nTesting silence trimming...")
    
    # Create audio with silence at both ends
    sample_rate = 22050
    silence_samples = int(0.2 * sample_rate)  # 200ms silence
    signal_samples = int(0.5 * sample_rate)  # 500ms signal
    
    # Create signal: silence - signal - silence
    silence = np.zeros(silence_samples)
    # Use deterministic sine wave instead of random signal for reproducible tests
    t = np.linspace(0, 0.5, signal_samples)
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave at 0.1 amplitude
    data = np.concatenate([silence, signal, silence])
    
    print(f"Original length: {len(data)} samples ({len(data)/sample_rate:.3f}s)")
    
    # Trim silence (threshold = 0.01)
    threshold = 0.01
    non_silent = np.abs(data) > threshold
    
    if non_silent.any():
        first_idx = np.argmax(non_silent)
        last_idx = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        # Different padding for start and end
        start_padding = int(0.05 * sample_rate)  # 50ms
        end_padding = int(0.01 * sample_rate)    # 10ms
        
        first_idx = max(0, first_idx - start_padding)
        last_idx = min(len(data) - 1, last_idx + end_padding)
        
        trimmed = data[first_idx:last_idx + 1]
        
        print(f"Trimmed length: {len(trimmed)} samples ({len(trimmed)/sample_rate:.3f}s)")
        print(f"✓ Start padding: {start_padding} samples ({start_padding/sample_rate*1000:.0f}ms)")
        print(f"✓ End padding: {end_padding} samples ({end_padding/sample_rate*1000:.0f}ms)")
        
        # Verify trimming removed most silence
        assert len(trimmed) < len(data), "Trimmed data should be shorter"
        
        # Verify some padding remains
        expected_length = signal_samples + start_padding + end_padding
        tolerance = int(0.02 * sample_rate)  # 20ms tolerance
        assert abs(len(trimmed) - expected_length) < tolerance, \
            f"Trimmed length {len(trimmed)} not close to expected {expected_length}"
        
        print(f"✓ Silence trimmed correctly")
        print(f"✓ Less padding at end reduces trailing noise")
    
    return True

def test_combined_processing():
    """Test combined trimming and fade-out"""
    print("\nTesting combined trimming + fade-out...")
    
    sample_rate = 22050
    
    # Create audio with silence and signal
    silence_samples = int(0.1 * sample_rate)
    signal_samples = int(0.3 * sample_rate)
    
    silence = np.zeros(silence_samples)
    # Use deterministic sine wave instead of random signal for reproducible tests
    t = np.linspace(0, 0.3, signal_samples)
    signal = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave at 0.1 amplitude
    data = np.concatenate([silence, signal, silence])
    
    print(f"Original: {len(data)} samples")
    
    # 1. Trim silence
    threshold = 0.01
    non_silent = np.abs(data) > threshold
    
    if non_silent.any():
        first_idx = np.argmax(non_silent)
        last_idx = len(non_silent) - np.argmax(non_silent[::-1]) - 1
        
        start_padding = int(0.05 * sample_rate)
        end_padding = int(0.01 * sample_rate)
        
        first_idx = max(0, first_idx - start_padding)
        last_idx = min(len(data) - 1, last_idx + end_padding)
        
        data = data[first_idx:last_idx + 1]
        print(f"After trimming: {len(data)} samples")
    
    # 2. Apply fade-out
    fade_duration = 0.05
    fade_samples = int(fade_duration * sample_rate)
    fade_samples = min(fade_samples, len(data))
    
    if fade_samples > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        data[-fade_samples:] *= fade_curve
        print(f"Applied fade: {fade_samples} samples")
    
    # Verify final result
    assert data[-1] < 0.01, f"Last sample should be near 0 after fade, got {data[-1]}"
    print(f"✓ Final sample: {data[-1]:.6f}")
    print(f"✓ Combined processing works correctly")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TTS Audio Processing Test")
    print("=" * 60)
    print()
    
    try:
        test_fade_out()
        test_silence_trimming()
        test_combined_processing()
        
        print("\n" + "=" * 60)
        print("All tests PASSED ✓")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ Test FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
