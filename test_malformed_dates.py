#!/usr/bin/env python3
"""Test script to demonstrate the malformed date validation issue."""

import pandas as pd
import sys
import os

# Add src to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_malformed_date_issue():
    """Demonstrate the current behavior with malformed dates."""
    
    # Create test data with malformed dates
    test_data = pd.DataFrame({
        'Date': ['2023-01-31', 'invalid-date', '2023-03-31', 'another-bad-date'],
        'Fund1': [0.01, 0.02, 0.03, 0.04],
        'Fund2': [0.05, 0.06, 0.07, 0.08]
    })
    
    print("Original data:")
    print(test_data)
    print()
    
    # Test current behavior with errors='coerce'
    print("Testing pd.to_datetime with errors='coerce':")
    coerced_dates = pd.to_datetime(test_data['Date'], errors='coerce')
    print("Coerced dates:")
    print(coerced_dates)
    print()
    
    # Check for NaT values (which indicate malformed dates)
    nat_mask = coerced_dates.isna()
    malformed_count = nat_mask.sum()
    print(f"Number of malformed dates (NaT values): {malformed_count}")
    
    if malformed_count > 0:
        malformed_indices = nat_mask[nat_mask].index.tolist()
        malformed_values = test_data.loc[malformed_indices, 'Date'].tolist()
        print(f"Malformed date values: {malformed_values}")
        print("These should be treated as validation errors, not added to expired lists!")
    
    return malformed_count > 0, malformed_values if malformed_count > 0 else []

if __name__ == "__main__":
    has_malformed, malformed_vals = test_malformed_date_issue()
    if has_malformed:
        print(f"\n✓ Test confirms the issue: malformed dates {malformed_vals} are converted to NaT")
        print("  These NaT values could be mistakenly added to expired lists instead of being validation errors")
    else:
        print("\n✗ No malformed dates detected in test")