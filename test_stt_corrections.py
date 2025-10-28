#!/usr/bin/env python3
"""
Unit tests for STT Corrections module.
Tests the post-processing correction functionality.
"""

import unittest
import logging
from modules.stt_corrections import STTCorrections


class TestSTTCorrections(unittest.TestCase):
    """Test cases for STTCorrections class."""

    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.DEBUG)
        self.corrections = STTCorrections()

    def test_initialization_with_defaults(self):
        """Test that corrections initializes with default rules."""
        corrections = STTCorrections()
        rules = corrections.get_corrections()
        self.assertGreater(len(rules), 0)
        self.assertIn("johannes och den", rules)
        self.assertEqual(rules["johannes och den"], "US open")

    def test_initialization_with_custom_corrections(self):
        """Test initialization with custom corrections."""
        custom = {"hej på dig": "hello"}
        corrections = STTCorrections(corrections=custom)
        rules = corrections.get_corrections()
        # Should have both default and custom corrections
        self.assertIn("johannes och den", rules)
        self.assertIn("hej på dig", rules)
        self.assertEqual(rules["hej på dig"], "hello")

    def test_apply_correction_simple(self):
        """Test applying a simple correction."""
        text = "jag såg johannes och den igår"
        corrected, was_corrected = self.corrections.apply_corrections(text)
        self.assertTrue(was_corrected)
        self.assertEqual(corrected, "jag såg US open igår")

    def test_apply_correction_case_insensitive(self):
        """Test case-insensitive correction."""
        text = "Jag såg Johannes Och Den igår"
        corrected, was_corrected = self.corrections.apply_corrections(text, case_insensitive=True)
        self.assertTrue(was_corrected)
        # Should replace while maintaining sentence structure
        self.assertIn("US open", corrected)

    def test_apply_correction_no_match(self):
        """Test when no correction is needed."""
        text = "detta är en vanlig mening"
        corrected, was_corrected = self.corrections.apply_corrections(text)
        self.assertFalse(was_corrected)
        self.assertEqual(corrected, text)

    def test_apply_correction_multiple_occurrences(self):
        """Test correcting multiple occurrences of the same phrase."""
        text = "johannes och den är stort och johannes och den pågår nu"
        corrected, was_corrected = self.corrections.apply_corrections(text)
        self.assertTrue(was_corrected)
        # Both occurrences should be corrected
        self.assertEqual(corrected, "US open är stort och US open pågår nu")

    def test_apply_correction_word_boundaries(self):
        """Test that corrections respect word boundaries."""
        # Add a custom correction for testing
        self.corrections.add_correction("den", "the")
        text = "johannes och den"
        corrected, was_corrected = self.corrections.apply_corrections(text)
        # Should correct "johannes och den" to "US open", not also replace standalone "den"
        # The longer phrase should be matched first due to sorting
        self.assertEqual(corrected, "US open")

    def test_apply_correction_empty_text(self):
        """Test with empty text."""
        text = ""
        corrected, was_corrected = self.corrections.apply_corrections(text)
        self.assertFalse(was_corrected)
        self.assertEqual(corrected, "")

    def test_apply_correction_none_text(self):
        """Test with None text."""
        text = None
        corrected, was_corrected = self.corrections.apply_corrections(text)
        self.assertFalse(was_corrected)
        self.assertIsNone(corrected)

    def test_add_correction(self):
        """Test adding a new correction rule."""
        self.corrections.add_correction("fel ord", "rätt ord")
        rules = self.corrections.get_corrections()
        self.assertIn("fel ord", rules)
        self.assertEqual(rules["fel ord"], "rätt ord")
        
        # Test applying the new correction
        text = "detta är fel ord här"
        corrected, was_corrected = self.corrections.apply_corrections(text)
        self.assertTrue(was_corrected)
        self.assertEqual(corrected, "detta är rätt ord här")

    def test_remove_correction(self):
        """Test removing a correction rule."""
        # First verify the rule exists
        self.assertIn("johannes och den", self.corrections.get_corrections())
        
        # Remove it
        result = self.corrections.remove_correction("johannes och den")
        self.assertTrue(result)
        
        # Verify it's gone
        self.assertNotIn("johannes och den", self.corrections.get_corrections())
        
        # Try to remove non-existent rule
        result = self.corrections.remove_correction("non existent")
        self.assertFalse(result)

    def test_clear_corrections(self):
        """Test clearing all correction rules."""
        self.corrections.clear_corrections()
        rules = self.corrections.get_corrections()
        self.assertEqual(len(rules), 0)

    def test_reset_to_defaults(self):
        """Test resetting to default corrections."""
        # Add a custom rule
        self.corrections.add_correction("custom", "rule")
        self.assertIn("custom", self.corrections.get_corrections())
        
        # Reset to defaults
        self.corrections.reset_to_defaults()
        rules = self.corrections.get_corrections()
        
        # Should have default rules but not custom
        self.assertIn("johannes och den", rules)
        self.assertNotIn("custom", rules)

    def test_multiple_variants_of_same_error(self):
        """Test that multiple variants of the same error are corrected."""
        variants = [
            "johannes och den",
            "johannes och dem",
            "johannes och de",
            "johannes o den",
            "johannes o dem",
        ]
        
        for variant in variants:
            text = f"jag såg {variant} på TV"
            corrected, was_corrected = self.corrections.apply_corrections(text)
            self.assertTrue(was_corrected, f"Failed to correct variant: {variant}")
            self.assertEqual(corrected, "jag såg US open på TV")

    def test_case_preservation_attempt(self):
        """Test that corrections are applied regardless of case."""
        text = "JAG SÅG JOHANNES OCH DEN IGÅR"
        corrected, was_corrected = self.corrections.apply_corrections(text, case_insensitive=True)
        self.assertTrue(was_corrected)
        self.assertIn("US open", corrected)

    def test_partial_match_not_corrected(self):
        """Test that partial matches within words are not corrected."""
        # This should not match because "johannes och den" is not a complete phrase
        text = "johannesochden"
        corrected, was_corrected = self.corrections.apply_corrections(text)
        self.assertFalse(was_corrected)
        self.assertEqual(corrected, text)

    def test_corrections_disabled(self):
        """Test behavior when corrections are cleared."""
        corrections = STTCorrections()
        corrections.clear_corrections()  # Clear all corrections
        text = "johannes och den"
        corrected, was_corrected = corrections.apply_corrections(text)
        # No corrections defined after clearing, so nothing should change
        self.assertFalse(was_corrected)
        self.assertEqual(corrected, text)


class TestSTTCorrectionsIntegration(unittest.TestCase):
    """Integration tests for STT corrections in realistic scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        logging.basicConfig(level=logging.INFO)
        self.corrections = STTCorrections()

    def test_realistic_sentence_us_open(self):
        """Test realistic sentence about US Open tennis."""
        sentences = [
            ("vem vann johannes och den i år", "vem vann US open i år"),
            ("johannes och den är en av de fyra grand slam turneringarna", 
             "US open är en av de fyra grand slam turneringarna"),
            ("titta på johannes och den live", "titta på US open live"),
        ]
        
        for input_text, expected in sentences:
            corrected, was_corrected = self.corrections.apply_corrections(input_text)
            self.assertTrue(was_corrected)
            self.assertEqual(corrected, expected)

    def test_sentence_without_error(self):
        """Test that correct sentences are not modified."""
        correct_sentences = [
            "hur är vädret idag",
            "vad är klockan",
            "spela musik",
            "slå på lampan",
        ]
        
        for text in correct_sentences:
            corrected, was_corrected = self.corrections.apply_corrections(text)
            self.assertFalse(was_corrected)
            self.assertEqual(corrected, text)


if __name__ == '__main__':
    unittest.main()
