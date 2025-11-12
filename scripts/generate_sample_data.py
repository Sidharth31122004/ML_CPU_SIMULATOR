"""
Script to generate sample process data CSV file
"""
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.dataset import create_sample_csv

if __name__ == "__main__":
    output_file = create_sample_csv("sample_processes.csv", n_samples=100)
    print(f"Sample dataset created: {output_file}")
