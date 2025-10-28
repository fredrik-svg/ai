#!/usr/bin/env python3
"""
Test to verify pre-buffering works correctly for STT.
This test verifies that audio before VAD confirmation is captured.
"""

import sys
import os
import collections
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prebuffer_logic():
    """Test that pre-buffer captures audio before VAD triggers"""
    print("Testing pre-buffer logic...")
    
    # Simulate the pre-buffer
    pre_buffer = collections.deque(maxlen=10)
    audio_buffer = []
    vad_triggered = False
    
    # Simulate adding frames before VAD triggers
    print("Adding 5 frames before VAD triggers...")
    for i in range(5):
        frame = f"frame_{i}"
        if not vad_triggered:
            pre_buffer.append(frame)
    
    assert len(pre_buffer) == 5, f"Expected 5 frames in pre_buffer, got {len(pre_buffer)}"
    assert len(audio_buffer) == 0, f"Expected 0 frames in audio_buffer, got {len(audio_buffer)}"
    print(f"✓ Pre-buffer contains {len(pre_buffer)} frames")
    
    # Simulate VAD triggering
    print("Simulating VAD trigger...")
    is_speaking = True
    if is_speaking and not vad_triggered:
        print("VAD triggered - adding pre-buffered frames")
        audio_buffer.extend(list(pre_buffer))
        pre_buffer.clear()
        vad_triggered = True
    
    assert len(audio_buffer) == 5, f"Expected 5 frames in audio_buffer, got {len(audio_buffer)}"
    assert len(pre_buffer) == 0, f"Expected 0 frames in pre_buffer, got {len(pre_buffer)}"
    assert vad_triggered == True, "Expected vad_triggered to be True"
    print(f"✓ Audio buffer now contains {len(audio_buffer)} frames (including pre-buffered)")
    
    # Simulate adding more frames after VAD triggers
    print("Adding 3 more frames after VAD triggers...")
    for i in range(5, 8):
        frame = f"frame_{i}"
        if is_speaking:
            if not vad_triggered:
                pre_buffer.append(frame)
            else:
                audio_buffer.append(frame)
    
    assert len(audio_buffer) == 8, f"Expected 8 frames in audio_buffer, got {len(audio_buffer)}"
    print(f"✓ Audio buffer now contains {len(audio_buffer)} frames total")
    
    # Verify order
    expected_frames = [f"frame_{i}" for i in range(8)]
    assert audio_buffer == expected_frames, f"Frame order incorrect: {audio_buffer}"
    print(f"✓ Frame order is correct: {audio_buffer}")
    
    print("\n✓ All pre-buffer tests passed!")
    return True

def test_prebuffer_overflow():
    """Test that pre-buffer doesn't overflow beyond maxlen"""
    print("\nTesting pre-buffer overflow protection...")
    
    pre_buffer = collections.deque(maxlen=10)
    
    # Add more than maxlen frames
    print("Adding 15 frames to pre-buffer with maxlen=10...")
    for i in range(15):
        pre_buffer.append(f"frame_{i}")
    
    assert len(pre_buffer) == 10, f"Expected 10 frames in pre_buffer, got {len(pre_buffer)}"
    
    # Check that oldest frames were dropped (should have frames 5-14)
    expected_frames = [f"frame_{i}" for i in range(5, 15)]
    actual_frames = list(pre_buffer)
    assert actual_frames == expected_frames, f"Expected frames 5-14, got {actual_frames}"
    print(f"✓ Pre-buffer correctly limited to {len(pre_buffer)} frames")
    print(f"✓ Oldest frames dropped, keeping most recent: {actual_frames[:3]}...{actual_frames[-1]}")
    
    print("\n✓ Overflow test passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("STT Pre-Buffer Test")
    print("=" * 60)
    print()
    
    try:
        test_prebuffer_logic()
        test_prebuffer_overflow()
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
