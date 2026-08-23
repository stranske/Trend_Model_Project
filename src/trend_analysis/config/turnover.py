"""Shared validation limits for portfolio turnover configuration."""

# A full liquidation followed by a full rebuild can trade 200% of portfolio value.
MAX_TURNOVER_CEILING = 2.0
