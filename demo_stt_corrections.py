#!/usr/bin/env python3
"""
Demo script for STT corrections functionality.
Demonstrates how the correction system works without requiring Vosk model.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.stt_corrections import STTCorrections


def demo_basic_correction():
    """Demonstrate basic correction functionality."""
    print("=" * 70)
    print("STT Corrections Demo")
    print("=" * 70)
    print()
    
    # Create corrections engine
    corrections = STTCorrections()
    
    print("📋 Default correction rules loaded:")
    for wrong, correct in corrections.get_corrections().items():
        print(f"  '{wrong}' → '{correct}'")
    print()
    
    # Test cases
    test_cases = [
        "vem vann johannes och den i år",
        "jag såg johannes och dem på TV igår",
        "johannes och de spelas i New York",
        "hur är vädret idag",  # No correction needed
        "titta på johannes o den live",
    ]
    
    print("🧪 Testing corrections on sample phrases:")
    print()
    
    for text in test_cases:
        corrected, was_corrected = corrections.apply_corrections(text)
        status = "✓ CORRECTED" if was_corrected else "○ NO CHANGE"
        print(f"{status}")
        print(f"  Input:  '{text}'")
        if was_corrected:
            print(f"  Output: '{corrected}'")
        print()
    
    print("=" * 70)


def demo_custom_corrections():
    """Demonstrate adding custom corrections."""
    print("\n")
    print("=" * 70)
    print("Custom Corrections Demo")
    print("=" * 70)
    print()
    
    # Create corrections with custom rules
    custom = {
        "hej på dej": "hej på dig",
        "jag vill ha en köp": "jag vill ha en kopp"
    }
    
    corrections = STTCorrections(corrections=custom)
    
    print("📋 Custom correction rules added:")
    for wrong, correct in custom.items():
        print(f"  '{wrong}' → '{correct}'")
    print()
    
    test_cases = [
        "hej på dej, hur mår du",
        "jag vill ha en köp kaffe",
        "johannes och den",  # Should still have default corrections
    ]
    
    print("🧪 Testing with custom corrections:")
    print()
    
    for text in test_cases:
        corrected, was_corrected = corrections.apply_corrections(text)
        status = "✓ CORRECTED" if was_corrected else "○ NO CHANGE"
        print(f"{status}")
        print(f"  Input:  '{text}'")
        if was_corrected:
            print(f"  Output: '{corrected}'")
        print()
    
    print("=" * 70)


def demo_configuration():
    """Demonstrate how to configure corrections in config.yaml."""
    print("\n")
    print("=" * 70)
    print("Configuration Example")
    print("=" * 70)
    print()
    
    print("To use corrections in your config.yaml, add:")
    print()
    print("```yaml")
    print("stt:")
    print("  model_path: 'models/vosk/vosk-model-small-sv-rhasspy-0.15'")
    print("  language: 'sv'")
    print("  sample_rate: 16000")
    print("  ")
    print("  # Enable post-processing corrections")
    print("  enable_corrections: true")
    print("  ")
    print("  # Add custom corrections (optional)")
    print("  custom_corrections:")
    print("    'my wrong phrase': 'correct phrase'")
    print("    'another error': 'proper text'")
    print("```")
    print()
    print("=" * 70)


def main():
    """Run all demos."""
    demo_basic_correction()
    demo_custom_corrections()
    demo_configuration()
    
    print("\n")
    print("✅ Demo complete!")
    print()
    print("To test with actual STT, use:")
    print("  python test_stt_corrections.py      # Run unit tests")
    print("  python test_stt_microphone.py       # Test with microphone")
    print()


if __name__ == '__main__':
    main()
