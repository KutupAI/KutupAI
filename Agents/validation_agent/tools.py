"""
Deterministic validation helpers for the Validation Agent.

Pure functions only: no state access, no I/O, no external clients.
"""

from __future__ import annotations

import re
from datetime import datetime


DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y")


def parse_date(value: str) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def validate_date_format(value: str) -> bool:
    return parse_date(value) is not None


def validate_email_format(value: str, pattern: str) -> bool:
    if not value or not isinstance(value, str):
        return False
    return bool(re.fullmatch(pattern, value.strip()))


def validate_phone_format(value: str, expected_digit_length: int) -> bool:
    if not value or not isinstance(value, str):
        return False
    digits = re.sub(r"\D", "", value)
    if digits.startswith("90") and len(digits) == expected_digit_length + 2:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == expected_digit_length + 1:
        digits = digits[1:]
    return len(digits) == expected_digit_length and digits.isdigit()