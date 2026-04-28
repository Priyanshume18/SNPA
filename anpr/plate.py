"""
Plate validation and normalization for Indian number plates.

Supported formats
-----------------
  Standard    : KA 01 AB 1234  → KA01AB1234
  Old/short   : MH 1 AB 1234   → MH1AB1234
  BH (Bharat) : 22BH1234AA     → 22BH1234AA
"""

from __future__ import annotations

import re
from functools import lru_cache

# --------------------------------------------------------------------- #
# OCR confusion maps — applied SELECTIVELY per position type.
#
# KEY INSIGHT: corrections must only apply to DIGIT positions.
# Applying B→8 globally corrupts "AB" → "A8" in plate series letters.
_DIGIT_FIXES: dict[str, str] = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",
    "T": "7",
}

# Each tuple: (loose-match pattern, set of known digit index positions in match)
# Loose patterns accept both alpha and digits at numeric positions so we can
# match before correcting — prevents B→8 happening to letter positions.
_PLATE_SPECS: list[tuple[re.Pattern, set[int]]] = [
    # BH-series (2021+): 22BH1234AB  — positions 0,1 + last 4 are digits
    (re.compile(r"[0-9A-Z]{2}BH[0-9A-Z]{4}[A-Z]{1,2}"), {0, 1}),
    # Standard new: KA05MJ7777  — positions 2,3 + last 4 are digits
    (re.compile(r"[A-Z]{2}[0-9A-Z]{2}[A-Z]{1,3}[0-9A-Z]{4}"), {2, 3}),
    # Old short district: MH1AB1234  — position 2 + last 4 are digits
    (re.compile(r"[A-Z]{2}[0-9A-Z][A-Z]{1,3}[0-9A-Z]{4}"), {2}),
]

# Strict validation — only well-formed plates pass
_STRICT_PLATE = re.compile(
    r"([0-9]{2}BH[0-9]{4}[A-Z]{1,2})"
    r"|([A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4})"
    r"|([A-Z]{2}[0-9][A-Z]{1,3}[0-9]{4})"
)


def _clean(text: str) -> str:
    """Uppercase and strip non-alphanumeric — NO corrections yet."""
    text = text.upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def _fix_digits(s: str, explicit_digit_positions: set[int]) -> str:
    """Apply DIGIT_FIXES to known digit positions + always the trailing 4."""
    chars = list(s)
    length = len(chars)
    positions: set[int] = set()
    for p in explicit_digit_positions:
        if 0 <= p < length:
            positions.add(p)
    # Serial number (last 4) is always digits
    for p in range(max(0, length - 4), length):
        positions.add(p)
    for p in positions:
        chars[p] = _DIGIT_FIXES.get(chars[p], chars[p])
    return "".join(chars)


def extract_plate(raw: str) -> str:
    """
    Extract and return the first valid plate string found in *raw*.

    Uses overlapping position scan so prefix noise (e.g. "PLATE: MH12AB1234")
    does not consume the valid token before we can match it.
    Returns empty string if none found.
    """
    cleaned = _clean(raw)
    for pattern, digit_positions in _PLATE_SPECS:
        pos = 0
        while pos < len(cleaned):
            match = pattern.search(cleaned, pos)
            if not match:
                break
            candidate = _fix_digits(match.group(), digit_positions)
            if _STRICT_PLATE.fullmatch(candidate):
                return candidate
            pos = match.start() + 1  # advance by 1 for overlapping scan
    return ""


def is_valid(text: str) -> bool:
    return bool(extract_plate(text))


@lru_cache(maxsize=1024)
def normalize(raw: str) -> str:
    """Normalize and cache — safe to call frequently."""
    return extract_plate(raw)


def load_plates_from_text(text: str) -> set[str]:
    """
    Parse a multi-line string of plates (supports comments with #).
    Returns a set of normalized plate strings.
    """
    plates: set[str] = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        plate = normalize(line)
        if plate:
            plates.add(plate)
    return plates
