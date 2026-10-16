"""Validation, Luhn check, regex parsing, and offline BIN lookup resolution."""

import re
from typing import NamedTuple

from card_vision.models import CardClass, NetworkScheme


class BinRule(NamedTuple):
    """Specification for matching a BIN/IIN prefix range."""

    start: int
    end: int
    prefix_length: int
    network: NetworkScheme
    card_class: CardClass
    valid_lengths: tuple[int, ...]


# Comprehensive offline ISO/IEC 7812 BIN / IIN lookup table
# Prefixes are evaluated in order of specificity (longest prefix match first)
OFFLINE_BIN_RULES: tuple[BinRule, ...] = (
    # --- RuPay (India / NPCI) ---
    # RuPay Credit: 3535, 3560, 508, 652150-652200, 652500-653149
    BinRule(652150, 652200, 6, NetworkScheme.RUPAY, CardClass.CREDIT, (16,)),
    BinRule(652500, 653149, 6, NetworkScheme.RUPAY, CardClass.CREDIT, (16,)),
    BinRule(606985, 607999, 6, NetworkScheme.RUPAY, CardClass.DEBIT, (16,)),
    BinRule(608001, 608500, 6, NetworkScheme.RUPAY, CardClass.DEBIT, (16,)),
    BinRule(652150, 653149, 6, NetworkScheme.RUPAY, CardClass.DEBIT, (16,)),
    BinRule(508500, 508999, 6, NetworkScheme.RUPAY, CardClass.DEBIT, (16,)),
    BinRule(3535, 3535, 4, NetworkScheme.RUPAY, CardClass.CREDIT, (16,)),
    BinRule(3560, 3560, 4, NetworkScheme.RUPAY, CardClass.CREDIT, (16,)),
    BinRule(81, 82, 2, NetworkScheme.RUPAY, CardClass.PREPAID, (16,)),
    # --- American Express ---
    # 34, 37 (15 digits, Credit or Charge/Credit)
    BinRule(34, 34, 2, NetworkScheme.AMEX, CardClass.CREDIT, (15,)),
    BinRule(37, 37, 2, NetworkScheme.AMEX, CardClass.CREDIT, (15,)),
    # --- Diners Club ---
    # 300-305 (14 digits)
    BinRule(300, 305, 3, NetworkScheme.DINERS_CLUB, CardClass.CREDIT, (14,)),
    # 3095 (14 digits)
    BinRule(3095, 3095, 4, NetworkScheme.DINERS_CLUB, CardClass.CREDIT, (14,)),
    # 36, 38-39 (14 digits)
    BinRule(36, 36, 2, NetworkScheme.DINERS_CLUB, CardClass.CREDIT, (14,)),
    BinRule(38, 39, 2, NetworkScheme.DINERS_CLUB, CardClass.CREDIT, (14,)),
    # --- JCB ---
    # 3528-3589 (16 to 19 digits)
    BinRule(3528, 3589, 4, NetworkScheme.JCB, CardClass.CREDIT, (16, 17, 18, 19)),
    # --- Discover ---
    # 6011 (16, 19 digits)
    BinRule(6011, 6011, 4, NetworkScheme.DISCOVER, CardClass.CREDIT, (16, 19)),
    # 622126 - 622925 (China UnionPay co-branded / Discover)
    BinRule(622126, 622925, 6, NetworkScheme.DISCOVER, CardClass.CREDIT, (16, 19)),
    # 644 - 649
    BinRule(644, 649, 3, NetworkScheme.DISCOVER, CardClass.CREDIT, (16, 19)),
    # 65
    BinRule(65, 65, 2, NetworkScheme.DISCOVER, CardClass.CREDIT, (16, 19)),
    # --- Maestro ---
    # 5018, 5020, 5038, 5893, 6304, 6759, 6761, 6762, 6763
    BinRule(5018, 5018, 4, NetworkScheme.MAESTRO, CardClass.DEBIT, (12, 13, 14, 15, 16, 18, 19)),
    BinRule(5020, 5020, 4, NetworkScheme.MAESTRO, CardClass.DEBIT, (12, 13, 14, 15, 16, 18, 19)),
    BinRule(5038, 5038, 4, NetworkScheme.MAESTRO, CardClass.DEBIT, (12, 13, 14, 15, 16, 18, 19)),
    BinRule(5893, 5893, 4, NetworkScheme.MAESTRO, CardClass.DEBIT, (12, 13, 14, 15, 16, 18, 19)),
    BinRule(6304, 6304, 4, NetworkScheme.MAESTRO, CardClass.DEBIT, (12, 13, 14, 15, 16, 18, 19)),
    BinRule(6759, 6763, 4, NetworkScheme.MAESTRO, CardClass.DEBIT, (12, 13, 14, 15, 16, 18, 19)),
    # --- Mastercard ---
    # 2221 - 2720 (2-series ranges, Credit/Debit)
    BinRule(222100, 272099, 6, NetworkScheme.MASTERCARD, CardClass.CREDIT, (16,)),
    # 51 - 55 (Traditional Mastercard ranges)
    # Common Debit BINs: 5120-5150, 5200-5250, 5300-5350, 5400-5450, 5500-5550
    BinRule(510000, 529999, 6, NetworkScheme.MASTERCARD, CardClass.CREDIT, (16,)),
    BinRule(530000, 539999, 6, NetworkScheme.MASTERCARD, CardClass.DEBIT, (16,)),
    BinRule(540000, 559999, 6, NetworkScheme.MASTERCARD, CardClass.CREDIT, (16,)),
    # --- Visa ---
    # Visa traditional prefix: 4
    # Specific Debit BIN sub-allocations
    BinRule(400000, 419999, 6, NetworkScheme.VISA, CardClass.DEBIT, (13, 16, 19)),
    BinRule(420000, 439999, 6, NetworkScheme.VISA, CardClass.CREDIT, (13, 16, 19)),
    BinRule(440000, 459999, 6, NetworkScheme.VISA, CardClass.DEBIT, (13, 16, 19)),
    BinRule(460000, 499999, 6, NetworkScheme.VISA, CardClass.CREDIT, (13, 16, 19)),
    BinRule(4, 4, 1, NetworkScheme.VISA, CardClass.CREDIT, (13, 16, 19)),
)


def validate_luhn(pan: str) -> bool:
    """Verify PAN validity using the standard Luhn (Mod 10) algorithm.

    Args:
        pan: Card number string (digits only).

    Returns:
        True if valid Luhn checksum and format, False otherwise.
    """
    clean_pan = "".join(ch for ch in pan if ch.isdigit())
    if not clean_pan or len(clean_pan) < 11 or len(clean_pan) > 19:
        return False

    digits = [int(d) for d in clean_pan]
    checksum = 0
    # Process from right to left, starting with the check digit at index -1
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit

    return checksum % 10 == 0


def clean_pan_digits(text: str) -> str:
    """Extract digits and normalize common OCR misidentifications.

    Args:
        text: Raw OCR extracted string.

    Returns:
        Digits-only PAN candidate.
    """
    # Replace common character confusions in digit fields
    mapping = {
        "O": "0",
        "o": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "i": "1",
        "Z": "2",
        "z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "s": "5",
        "b": "6",
        "G": "6",
        "T": "7",
        "B": "8",
        "q": "9",
        "g": "9",
    }
    clean_chars = []
    for ch in text.strip():
        if ch.isdigit():
            clean_chars.append(ch)
        elif ch in mapping:
            clean_chars.append(mapping[ch])
    return "".join(clean_chars)


def mask_pan(pan: str, mask_char: str = "*") -> str:
    """Format and mask PAN according to PCI-DSS rules (retain first 6 and/or last 4).

    Example output for 16-digit card:
        '**** **** **** 1234' or '4111 11** **** 1234'
    Standard masked format shows only the last 4 digits grouped in blocks of 4.

    Args:
        pan: Card digits.
        mask_char: Character to use for obfuscation.

    Returns:
        Masked and formatted string.
    """
    digits = clean_pan_digits(pan)
    if not digits:
        return ""
    length = len(digits)
    if length <= 4:
        return digits

    # Mask everything except the last 4 digits
    masked = (mask_char * (length - 4)) + digits[-4:]

    # Format into blocks of 4 (or 4-6-5 for AMEX 15-digit)
    if length == 15:
        # Amex 4-6-5 format
        return f"{masked[:4]} {masked[4:10]} {masked[10:]}"
    # Standard 4-4-4-4
    chunks = [masked[i : i + 4] for i in range(0, len(masked), 4)]
    return " ".join(chunks)


def format_pan(pan: str) -> str:
    """Group PAN digits into standard spacing (e.g. 4-4-4-4 or 4-6-5)."""
    digits = clean_pan_digits(pan)
    if len(digits) == 15:
        return f"{digits[:4]} {digits[4:10]} {digits[10:]}"
    chunks = [digits[i : i + 4] for i in range(0, len(digits), 4)]
    return " ".join(chunks)


def parse_expiry_date(text: str) -> str | None:
    """Parse and validate card expiration date from OCR text.

    Matches formats like MM/YY, MM/YYYY, MM-YY, MM.YY, MM YY.
    Ensures Month is 01-12.

    Args:
        text: Raw OCR candidate string.

    Returns:
        Formatted 'MM/YY' string if valid, None otherwise.
    """
    if not text:
        return None

    # Replace potential OCR mistakes
    normalized = text.strip()
    normalized = normalized.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")

    # Regex matching MM/YY or MM/YYYY with various separators
    pattern = r"(?:(?:VALID\s+THRU|EXP\s+DATE|EXPIRES|EXP|GOOD\s+THRU|THRU)\s*[:.]?\s*)?([0-1]?[0-9])[\s\/\-\.\\]+([0-9]{2,4})"
    matches = re.findall(pattern, normalized, re.IGNORECASE)

    for month_str, year_str in matches:
        month = int(month_str)
        if 1 <= month <= 12:
            # Normalize 2-digit year
            if len(year_str) == 4:
                year_str = year_str[2:]
            return f"{month:02d}/{year_str}"

    # Also attempt strict search for isolated 4-digit groups (e.g. '0828' -> 08/28)
    digits_match = re.search(r"\b(0[1-9]|1[0-2])([2-3][0-9])\b", normalized)
    if digits_match:
        return f"{digits_match.group(1)}/{digits_match.group(2)}"

    return None


def resolve_bin_metadata(pan: str) -> tuple[NetworkScheme, CardClass]:
    """Resolve Card Network and Debit/Credit class offline using IIN/BIN prefix tables.

    Sorted to prioritize highest prefix length for maximum precision.

    Args:
        pan: Card number (digits).

    Returns:
        Tuple of (NetworkScheme, CardClass).
    """
    digits = clean_pan_digits(pan)
    if not digits:
        return NetworkScheme.UNKNOWN, CardClass.UNKNOWN

    length = len(digits)

    # Sort rules descending by prefix_length to prioritize specific sub-ranges
    sorted_rules = sorted(OFFLINE_BIN_RULES, key=lambda r: r.prefix_length, reverse=True)

    for rule in sorted_rules:
        if len(digits) < rule.prefix_length:
            continue

        prefix = int(digits[: rule.prefix_length])
        if rule.start <= prefix <= rule.end:
            if length in rule.valid_lengths:
                return rule.network, rule.card_class

    # Fallback to high-level first digit resolution if no exact sub-range matched
    first_digit = digits[0]
    if first_digit == "4":
        return NetworkScheme.VISA, CardClass.CREDIT
    if first_digit == "5":
        return NetworkScheme.MASTERCARD, CardClass.CREDIT
    if first_digit == "3" and len(digits) >= 2 and digits[:2] in ("34", "37"):
        return NetworkScheme.AMEX, CardClass.CREDIT

    return NetworkScheme.UNKNOWN, CardClass.UNKNOWN


def clean_cardholder_name(text: str) -> str | None:
    """Sanitize and validate extracted cardholder name.

    Names on financial cards consist of Latin uppercase letters, spaces, hyphens,
    and periods. Excludes common card labels like 'VALID THRU', 'CARDHOLDER', etc.

    Args:
        text: Raw OCR string.

    Returns:
        Cleaned uppercase name or None if invalid.
    """
    if not text:
        return None

    # Filter out known keywords
    blacklist = {
        "VALID",
        "THRU",
        "EXPIRES",
        "MEMBER",
        "SINCE",
        "DEBIT",
        "CREDIT",
        "ELECTRONIC",
        "USE",
        "ONLY",
        "AUTHORIZED",
        "SIGNATURE",
        "CARD",
        "CARDHOLDER",
        "GLOBAL",
        "RUPAY",
        "VISA",
        "MASTERCARD",
        "DISCOVER",
        "PLATINUM",
        "GOLD",
        "CLASSIC",
        "TITANIUM",
        "PREMIER",
        "WORLD",
        "BANK",
    }

    cleaned = re.sub(r"[^A-Za-z\s\.\-]", " ", text).strip().upper()
    words = [w for w in cleaned.split() if len(w) >= 1 and w not in blacklist]

    if not words or len(words) < 2 or not any(len(w) > 1 for w in words):
        return None

    candidate = " ".join(words)
    # Most full names are between 4 and 30 chars
    if 4 <= len(candidate) <= 30:
        return candidate

    return None
