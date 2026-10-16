"""Comprehensive unit test suite for card_vision validation, Luhn, and BIN lookups."""

import pytest

from card_vision.models import CardClass, NetworkScheme
from card_vision.validator import (
    clean_cardholder_name,
    clean_pan_digits,
    format_pan,
    mask_pan,
    parse_expiry_date,
    resolve_bin_metadata,
    validate_luhn,
)


class TestLuhnAlgorithm:
    """Test ISO/IEC 7812 Luhn Mod-10 checksum validation."""

    def test_valid_luhn_numbers(self) -> None:
        # Standard test vectors with known valid check digits
        assert validate_luhn("49927398716") is True
        assert validate_luhn("49927398717") is False
        assert validate_luhn("1234567812345670") is True
        assert validate_luhn("79927398713") is True
        assert validate_luhn("79927398710") is False
        assert validate_luhn("79927398711") is False
        assert validate_luhn("79927398712") is False

    def test_luhn_with_spaces_and_hyphens(self) -> None:
        assert validate_luhn("4992-7398-716") is True
        assert validate_luhn("1234 5678 1234 5670") is True

    def test_invalid_lengths_and_empty(self) -> None:
        assert validate_luhn("") is False
        assert validate_luhn("123") is False
        assert validate_luhn("12345678901") is False  # Under 12 digits
        assert validate_luhn("12345678901234567890") is False  # 20 digits (over max 19)
        assert validate_luhn("abcdefghijklm") is False


class TestBinResolution:
    """Test BIN/IIN offline resolution for Scheme and Debit/Credit class."""

    @pytest.mark.parametrize(
        ("pan", "expected_network", "expected_class"),
        [
            # Visa
            ("4111111111111111", NetworkScheme.VISA, CardClass.DEBIT),
            ("4242424242424242", NetworkScheme.VISA, CardClass.CREDIT),
            ("4532015678901234", NetworkScheme.VISA, CardClass.DEBIT),
            ("4890123456789012", NetworkScheme.VISA, CardClass.CREDIT),
            # Mastercard
            ("5105105105105100", NetworkScheme.MASTERCARD, CardClass.CREDIT),
            ("5350123456789012", NetworkScheme.MASTERCARD, CardClass.DEBIT),
            ("2221001234567890", NetworkScheme.MASTERCARD, CardClass.CREDIT),
            ("2720991234567890", NetworkScheme.MASTERCARD, CardClass.CREDIT),
            # American Express (15 digits)
            ("341234567890123", NetworkScheme.AMEX, CardClass.CREDIT),
            ("378282246310005", NetworkScheme.AMEX, CardClass.CREDIT),
            # RuPay
            ("6521501234567890", NetworkScheme.RUPAY, CardClass.CREDIT),
            ("6071231234567890", NetworkScheme.RUPAY, CardClass.DEBIT),
            ("5085001234567890", NetworkScheme.RUPAY, CardClass.DEBIT),
            # Discover
            ("6011000990139424", NetworkScheme.DISCOVER, CardClass.CREDIT),
            ("6500001234567890", NetworkScheme.DISCOVER, CardClass.CREDIT),
            # Diners Club (14 digits)
            ("30012345678901", NetworkScheme.DINERS_CLUB, CardClass.CREDIT),
            ("36123456789012", NetworkScheme.DINERS_CLUB, CardClass.CREDIT),
            # JCB
            ("3530123456789012", NetworkScheme.JCB, CardClass.CREDIT),
        ],
    )
    def test_bin_lookup(
        self, pan: str, expected_network: NetworkScheme, expected_class: CardClass
    ) -> None:
        network, card_class = resolve_bin_metadata(pan)
        assert network == expected_network
        assert card_class == expected_class


class TestExpiryParsing:
    """Test date extraction and normalization."""

    @pytest.mark.parametrize(
        ("input_text", "expected"),
        [
            ("12/28", "12/28"),
            ("05/26", "05/26"),
            ("VALID THRU 08/29", "08/29"),
            ("EXPIRES 03/27", "03/27"),
            ("GOOD THRU 11-25", "11/25"),
            ("EXP 07/2026", "07/26"),
            ("12/2030", "12/30"),
            ("09 27", "09/27"),
            ("13/26", None),  # Invalid month
            ("00/26", None),  # Invalid month
            ("NOT A DATE", None),
            ("", None),
        ],
    )
    def test_parse_expiry_date(self, input_text: str, expected: str | None) -> None:
        assert parse_expiry_date(input_text) == expected


class TestPrivacyAndFormatting:
    """Test PAN masking and formatting functions."""

    def test_mask_pan_16_digits(self) -> None:
        pan = "4111222233334444"
        masked = mask_pan(pan)
        assert masked == "**** **** **** 4444"
        assert len(masked.replace(" ", "")) == 16

    def test_mask_pan_15_digits_amex(self) -> None:
        pan = "378282246310005"
        masked = mask_pan(pan)
        assert masked == "**** ****** *0005"
        assert masked.endswith("0005")

    def test_format_pan(self) -> None:
        assert format_pan("4111222233334444") == "4111 2222 3333 4444"
        assert format_pan("378282246310005") == "3782 822463 10005"

    def test_clean_pan_digits_ocr_replacements(self) -> None:
        assert clean_pan_digits("41ll-2222-3333-444O") == "4111222233334440"
        assert clean_pan_digits("4111 2222 3333 4444") == "4111222233334444"


class TestCardholderNameSanitization:
    """Test name parsing and filtering."""

    def test_valid_name(self) -> None:
        assert clean_cardholder_name("JOHN DOE") == "JOHN DOE"
        assert clean_cardholder_name("ALICE M SMITH") == "ALICE M SMITH"

    def test_filter_labels(self) -> None:
        # Should filter out VALID THRU and keep name
        assert clean_cardholder_name("VALID THRU JOHN DOE") == "JOHN DOE"
        assert clean_cardholder_name("CARDHOLDER VISA DEBIT") is None
