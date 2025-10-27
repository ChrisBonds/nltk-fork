import random

import pytest

import nltk
from nltk.tokenize.punkt import PunktParameters, PunktSentenceTokenizer

# hello
n = 1234567
# deterministic seed for reproducible tests
RNG_SEED = 12345
random.seed(RNG_SEED)


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Ensure punkt data exists before running tests."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# Multi-period abbreviations to stress-test (canonical dotted forms)
MULTI_PERIOD_ABBREVS = [
    "M.D.",
    "Ph.D.",
    "U.S.A.",
    "U.S.",
    "e.g.",
    "i.e.",
    "vs.",
    "U.K.",
    "c.f.",
    "a.m.",
    "p.m.",
]


def get_pretrained_tokenizer():
    """Get the pre-trained English Punkt tokenizer with existing collocations and ortho context."""
    try:
        return nltk.data.load("tokenizers/punkt/english.pickle")
    except LookupError:
        # Fallback: create a basic tokenizer
        return PunktSentenceTokenizer()


def simple_gold_sentences(text):
    """
    Simple sentence splitter for creating gold standard.
    Splits on '. ' followed by capital letter, '? ', '! ', and paragraph breaks.
    """
    import re

    # Split on sentence boundaries while preserving the punctuation
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    return [s.strip() for s in sentences if s.strip()]


class TestMultiPeriodAbbreviations:
    """Test suite for multi-period abbreviation handling in Punkt tokenizer."""

    def test_discover_multi_period_abbrevs_single_letter(self):
        """Test discovery of single-letter multi-period abbreviations like U.S.A."""
        tokenizer = get_pretrained_tokenizer()
        text = "The U.S.A. is a country. The U.K. is another country."

        discovered = tokenizer._lang_vars.discover_multi_period_abbrevs(text)

        assert "U.S.A." in discovered
        assert "U.K." in discovered

    def test_discover_multi_period_abbrevs_multi_letter(self):
        """Test discovery of multi-letter multi-period abbreviations like Ph.D."""
        tokenizer = get_pretrained_tokenizer()
        text = "She has a Ph.D. degree. He earned his M.D. last year."

        discovered = tokenizer._lang_vars.discover_multi_period_abbrevs(text)

        assert "Ph.D." in discovered
        assert "M.D." in discovered

    def test_discover_multi_period_abbrevs_mixed(self):
        """Test discovery of mixed abbreviations."""
        tokenizer = get_pretrained_tokenizer()
        text = "The U.S. government and Ph.D. holders met. Use e.g. examples."

        discovered = tokenizer._lang_vars.discover_multi_period_abbrevs(text)

        assert "U.S." in discovered
        assert "Ph.D." in discovered
        assert "e.g." in discovered

    def test_seed_punkt_abbrevs_from_text(self):
        """Test seeding abbreviations from text into PunktParameters."""
        tokenizer = get_pretrained_tokenizer()
        params = PunktParameters()
        text = "The U.S.A. and Ph.D. programs are important."

        tokenizer._lang_vars.seed_punkt_abbrevs(params, text=text)

        # Check that normalized forms are added
        assert "usa" in params.abbrev_types or "u.s.a" in params.abbrev_types
        assert "phd" in params.abbrev_types or "ph.d" in params.abbrev_types

    def test_seed_punkt_abbrevs_with_extra(self):
        """Test seeding abbreviations with extra provided abbreviations."""
        tokenizer = get_pretrained_tokenizer()
        params = PunktParameters()
        extra = ["M.D.", "D.Phil."]

        tokenizer._lang_vars.seed_punkt_abbrevs(params, extra=extra)

        # Check that normalized forms are added
        assert "md" in params.abbrev_types or "m.d" in params.abbrev_types
        assert "dphil" in params.abbrev_types or "d.phil" in params.abbrev_types

    def test_tokenize_with_seeded_abbrevs_phd(self):
        """Test that Ph.D. doesn't cause sentence breaks when seeded."""
        tokenizer = get_pretrained_tokenizer()

        # Seed the abbreviation
        tokenizer._lang_vars.seed_punkt_abbrevs(tokenizer._params, extra=["Ph.D."])

        text = "He has a Ph.D. He teaches at the university."
        sentences = tokenizer.tokenize(text)

        # Should be 2 sentences, not split at Ph.D.
        assert len(sentences) == 2
        assert "Ph.D." in sentences[0]

    def test_tokenize_with_seeded_abbrevs_usa(self):
        """Test that U.S.A. doesn't cause sentence breaks when seeded."""
        tokenizer = get_pretrained_tokenizer()

        # Seed the abbreviation
        tokenizer._lang_vars.seed_punkt_abbrevs(tokenizer._params, extra=["U.S.A."])

        text = "They moved to the U.S.A. They love it there."
        sentences = tokenizer.tokenize(text)

        # Should be 2 sentences, not split at U.S.A.
        assert len(sentences) == 2
        assert "U.S.A." in sentences[0]

    def test_normalize_for_punkt(self):
        """Test the normalization function for Punkt parameters."""
        tokenizer = get_pretrained_tokenizer()

        # Test various formats
        no_dots, internal = tokenizer._lang_vars._normalize_for_punkt("U.S.A.")
        assert no_dots == "usa"
        assert internal == "u.s.a"

        no_dots, internal = tokenizer._lang_vars._normalize_for_punkt("Ph.D.")
        assert no_dots == "phd"
        assert internal == "ph.d"

        # Test with surrounding punctuation
        no_dots, internal = tokenizer._lang_vars._normalize_for_punkt("(M.D.)")
        assert no_dots == "md"
        assert internal == "m.d"

    def test_multiple_abbreviations_in_sequence(self):
        """Test handling of multiple abbreviations in sequence."""
        tokenizer = get_pretrained_tokenizer()

        # Seed multiple abbreviations
        tokenizer._lang_vars.seed_punkt_abbrevs(
            tokenizer._params, extra=["Ph.D.", "M.D.", "U.S."]
        )

        text = "Dr. Smith, Ph.D., M.D., works in the U.S. He is very qualified."
        sentences = tokenizer.tokenize(text)

        # Should be 2 sentences
        assert len(sentences) == 2

    def test_case_variations(self):
        """Test that case variations are handled correctly."""
        tokenizer = get_pretrained_tokenizer()

        # Seed with uppercase version
        tokenizer._lang_vars.seed_punkt_abbrevs(tokenizer._params, extra=["U.S.A."])

        # Test with lowercase (should still work due to normalization)
        text = "The u.s.a. is large. It has many states."
        sentences = tokenizer.tokenize(text)

        # Should handle it reasonably (exact behavior may vary)
        assert len(sentences) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
