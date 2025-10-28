"""
STT Corrections Module.
Provides post-processing corrections for common speech recognition errors.
"""

import logging
import re
from typing import Dict, List, Tuple


class STTCorrections:
    """
    Post-processing corrections for STT output.
    Handles common misrecognitions and phonetic mistakes.
    """

    def __init__(self, corrections: Dict[str, str] = None):
        """
        Initialize the corrections engine.

        Args:
            corrections: Dictionary of phrase corrections {wrong_phrase: correct_phrase}
                        If None, uses default corrections.
        """
        self.logger = logging.getLogger(__name__)
        
        # Default corrections for Swedish STT common errors
        self.default_corrections = {
            # US open tennis tournament misrecognition
            "johannes och den": "US open",
            "johannes och dem": "US open",
            "johannes och de": "US open",
            "johannes o den": "US open",
            "johannes o dem": "US open",
        }
        
        # Merge default corrections with user-provided ones
        if corrections is None:
            self.corrections = self.default_corrections.copy()
        else:
            self.corrections = self.default_corrections.copy()
            self.corrections.update(corrections)
        
        self.logger.debug(f"Initialized STT corrections with {len(self.corrections)} rules")

    def apply_corrections(self, text: str, case_insensitive: bool = True) -> Tuple[str, bool]:
        """
        Apply corrections to transcribed text.

        Args:
            text: Input text from STT
            case_insensitive: If True, perform case-insensitive matching

        Returns:
            Tuple of (corrected_text, was_corrected)
        """
        if not text or not self.corrections:
            return text, False

        original_text = text
        corrected = False

        # Sort corrections by length (longest first) to handle overlapping patterns
        sorted_corrections = sorted(
            self.corrections.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )

        for wrong_phrase, correct_phrase in sorted_corrections:
            if case_insensitive:
                # Create pattern for case-insensitive matching with word boundaries
                pattern = re.compile(r'\b' + re.escape(wrong_phrase) + r'\b', re.IGNORECASE)
                if pattern.search(text):
                    # Preserve the case style of the original if possible
                    text = pattern.sub(correct_phrase, text)
                    corrected = True
                    self.logger.info(f"Applied correction: '{wrong_phrase}' -> '{correct_phrase}'")
            else:
                # Case-sensitive matching with word boundaries
                pattern = re.compile(r'\b' + re.escape(wrong_phrase) + r'\b')
                if pattern.search(text):
                    text = pattern.sub(correct_phrase, text)
                    corrected = True
                    self.logger.info(f"Applied correction: '{wrong_phrase}' -> '{correct_phrase}'")

        if corrected:
            self.logger.debug(f"Text before correction: '{original_text}'")
            self.logger.debug(f"Text after correction: '{text}'")

        return text, corrected

    def add_correction(self, wrong_phrase: str, correct_phrase: str) -> None:
        """
        Add a new correction rule.

        Args:
            wrong_phrase: The incorrect phrase to match
            correct_phrase: The correct phrase to substitute
        """
        self.corrections[wrong_phrase] = correct_phrase
        self.logger.debug(f"Added correction rule: '{wrong_phrase}' -> '{correct_phrase}'")

    def remove_correction(self, wrong_phrase: str) -> bool:
        """
        Remove a correction rule.

        Args:
            wrong_phrase: The incorrect phrase to remove

        Returns:
            True if the rule was removed, False if it didn't exist
        """
        if wrong_phrase in self.corrections:
            del self.corrections[wrong_phrase]
            self.logger.debug(f"Removed correction rule: '{wrong_phrase}'")
            return True
        return False

    def get_corrections(self) -> Dict[str, str]:
        """
        Get all current correction rules.

        Returns:
            Dictionary of all correction rules
        """
        return self.corrections.copy()

    def clear_corrections(self) -> None:
        """Clear all correction rules."""
        self.corrections.clear()
        self.logger.debug("Cleared all correction rules")

    def reset_to_defaults(self) -> None:
        """Reset corrections to default rules only."""
        self.corrections = self.default_corrections.copy()
        self.logger.debug("Reset corrections to defaults")
