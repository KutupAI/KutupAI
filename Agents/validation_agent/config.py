"""
Configuration for the Validation Agent.

No document-type-specific required-field matrix exists in the real
Extraction contract, so none is invented here.
"""

MIN_CLASSIFICATION_CONFIDENCE = 0.5

EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

PHONE_DIGIT_LENGTH = 10