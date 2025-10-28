#!/usr/bin/env python3
"""
Integration test for STT with corrections.
Tests the full STT pipeline with post-processing corrections enabled.
"""

import unittest
import logging
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add parent directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))

from modules.stt import SpeechToText


class TestSTTWithCorrections(unittest.TestCase):
    """Integration tests for STT with corrections enabled."""

    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def test_stt_initialization_with_corrections(self):
        """Test that STT initializes with corrections enabled."""
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            enable_corrections=True
        )
        
        # Check that corrections are initialized
        self.assertIsNotNone(stt.corrections)
        self.assertTrue(stt.enable_corrections)
        
        # Verify default corrections are present
        rules = stt.corrections.get_corrections()
        self.assertIn("johannes och den", rules)
        self.assertEqual(rules["johannes och den"], "US open")

    def test_stt_initialization_without_corrections(self):
        """Test that STT can be initialized without corrections."""
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            enable_corrections=False
        )
        
        # Check that corrections are disabled
        self.assertIsNone(stt.corrections)
        self.assertFalse(stt.enable_corrections)

    def test_stt_initialization_with_custom_corrections(self):
        """Test that STT initializes with custom corrections."""
        custom_corrections = {
            "test phrase": "corrected phrase"
        }
        
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            enable_corrections=True,
            custom_corrections=custom_corrections
        )
        
        # Check that both default and custom corrections are present
        rules = stt.corrections.get_corrections()
        self.assertIn("johannes och den", rules)  # Default
        self.assertIn("test phrase", rules)  # Custom
        self.assertEqual(rules["test phrase"], "corrected phrase")

    @patch.object(SpeechToText, 'load_model')
    @patch('modules.stt.KaldiRecognizer')
    def test_transcribe_with_correction_applied(self, mock_recognizer_class, mock_load_model):
        """Test transcription with correction applied."""
        # Mock the recognizer to return text that needs correction
        mock_recognizer = Mock()
        mock_recognizer.FinalResult.return_value = '{"text": "jag såg johannes och den igår"}'
        mock_recognizer_class.return_value = mock_recognizer
        
        # Create STT with corrections enabled
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            enable_corrections=True
        )
        stt.model = Mock()  # Mock the model
        stt.recognizer = mock_recognizer
        
        # Transcribe (mock audio data)
        import numpy as np
        audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = stt.transcribe_audio(audio_data)
        
        # Verify correction was applied
        self.assertEqual(result, "jag såg US open igår")

    @patch.object(SpeechToText, 'load_model')
    @patch('modules.stt.KaldiRecognizer')
    def test_transcribe_without_correction_needed(self, mock_recognizer_class, mock_load_model):
        """Test transcription when no correction is needed."""
        # Mock the recognizer to return text that doesn't need correction
        mock_recognizer = Mock()
        mock_recognizer.FinalResult.return_value = '{"text": "hur är vädret idag"}'
        mock_recognizer_class.return_value = mock_recognizer
        
        # Create STT with corrections enabled
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            enable_corrections=True
        )
        stt.model = Mock()  # Mock the model
        stt.recognizer = mock_recognizer
        
        # Transcribe (mock audio data)
        import numpy as np
        audio_data = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe_audio(audio_data)
        
        # Verify no correction was applied, text remains the same
        self.assertEqual(result, "hur är vädret idag")

    @patch.object(SpeechToText, 'load_model')
    @patch('modules.stt.KaldiRecognizer')
    def test_transcribe_with_corrections_disabled(self, mock_recognizer_class, mock_load_model):
        """Test transcription with corrections disabled."""
        # Mock the recognizer to return text that would need correction
        mock_recognizer = Mock()
        mock_recognizer.FinalResult.return_value = '{"text": "jag såg johannes och den igår"}'
        mock_recognizer_class.return_value = mock_recognizer
        
        # Create STT with corrections disabled
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            enable_corrections=False
        )
        stt.model = Mock()  # Mock the model
        stt.recognizer = mock_recognizer
        
        # Transcribe (mock audio data)
        import numpy as np
        audio_data = np.zeros(16000, dtype=np.float32)
        result = stt.transcribe_audio(audio_data)
        
        # Verify NO correction was applied (corrections disabled)
        self.assertEqual(result, "jag såg johannes och den igår")

    def test_corrections_backward_compatibility(self):
        """Test that existing code without corrections parameters still works."""
        # This tests backward compatibility - old code that doesn't specify
        # the new parameters should still work with defaults
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            language="sv",
            sample_rate=16000
            # Not specifying enable_corrections or custom_corrections
        )
        
        # Should default to corrections enabled
        self.assertTrue(stt.enable_corrections)
        self.assertIsNotNone(stt.corrections)


class TestSTTCorrectionsEndToEnd(unittest.TestCase):
    """End-to-end tests simulating real usage scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO)

    @patch.object(SpeechToText, 'load_model')
    @patch('modules.stt.KaldiRecognizer')
    def test_multiple_corrections_in_sequence(self, mock_recognizer_class, mock_load_model):
        """Test multiple transcriptions with corrections in sequence."""
        # Create STT instance
        stt = SpeechToText(
            model_path="models/vosk/vosk-model-small-sv-rhasspy-0.15",
            enable_corrections=True
        )
        stt.model = Mock()
        
        test_cases = [
            ('{"text": "vem vann johannes och den"}', "vem vann US open"),
            ('{"text": "hur är vädret"}', "hur är vädret"),
            ('{"text": "johannes och dem spelas i new york"}', "US open spelas i new york"),
        ]
        
        import numpy as np
        audio_data = np.zeros(16000, dtype=np.float32)
        
        for mock_result, expected_output in test_cases:
            # Update mock for each test case
            mock_recognizer = Mock()
            mock_recognizer.FinalResult.return_value = mock_result
            mock_recognizer_class.return_value = mock_recognizer
            stt.recognizer = mock_recognizer
            
            result = stt.transcribe_audio(audio_data)
            self.assertEqual(result, expected_output)


if __name__ == '__main__':
    unittest.main()
