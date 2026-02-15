#!/usr/bin/env python3
"""
Validate that example execution produced expected results.
"""
import sys
import os
import json
from pathlib import Path

def validate_results(example_path):
    """
    Validate results for a given example.
    
    Checks:
    1. Results table exists
    2. Metrics are populated (not NaN or empty)
    3. Accuracy is within expected range
    """
    workspace = Path(example_path) / "workspace"
    
    # Check results table exists
    results_file = workspace / "benchmarkingjob" / "rank" / "selected_rank.csv"
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return False
    
    # Parse results
    import pandas as pd
    try:
        df = pd.read_csv(results_file)
    except Exception as e:
        print(f"❌ Error reading results file: {e}")
        return False
    
    if df.empty:
        print("❌ Results file is empty")
        return False

    # Validate metrics are populated
    required_cols = ['accuracy', 'task_avg_acc']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Missing column: {col}")
            return False
        
        if df[col].isna().any():
            print(f"❌ Column {col} contains NaN values")
            return False
    
    # Validate accuracy range (0-1)
    if not (0 <= df['accuracy'].iloc[0] <= 1):
        print(f"❌ Accuracy out of range: {df['accuracy'].iloc[0]}")
        return False
    
    print(f"✅ Results validation passed for {example_path}")
    print(f"   Accuracy: {df['accuracy'].iloc[0]:.4f}")
    print(f"   Task Avg Acc: {df['task_avg_acc'].iloc[0]:.4f}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: validate_results.py <example_path>")
        sys.exit(1)
    
    example_path = sys.argv[1]
    success = validate_results(example_path)
    sys.exit(0 if success else 1)
