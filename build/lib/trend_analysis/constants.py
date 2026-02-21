"""Constants for the trend analysis package.

This module centralizes all hardcoded default values to improve
maintainability and make them easier to modify.
"""

# Default export configuration constants
DEFAULT_OUTPUT_DIRECTORY = "outputs"
DEFAULT_OUTPUT_FORMATS = ["excel"]

# Numerical tolerance constants for precision comparisons
NUMERICAL_TOLERANCE_HIGH = 1e-12  # High precision numerical tolerance
NUMERICAL_TOLERANCE_MEDIUM = 1e-9  # Medium precision numerical tolerance
NUMERICAL_TOLERANCE_LOW = 1e-6  # Lower precision numerical tolerance

__all__ = [
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_OUTPUT_FORMATS",
    "NUMERICAL_TOLERANCE_HIGH",
    "NUMERICAL_TOLERANCE_MEDIUM",
    "NUMERICAL_TOLERANCE_LOW",
]
