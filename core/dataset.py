"""
Dataset loading and synthetic data generation module
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def generate_synthetic_dataset(n_processes: int = 1000) -> pd.DataFrame:
    """
    Generate a synthetic dataset of processes for CPU scheduling.
    
    Args:
        n_processes: Number of processes to generate
        
    Returns:
        DataFrame with process parameters
    """
    np.random.seed(42)
    
    data = {
        'Arrival_Time': np.sort(np.random.randint(0, 100, n_processes)),
        'Burst_Time': np.random.randint(5, 50, n_processes),
        'Priority': np.random.randint(1, 6, n_processes),
        'Prev_Burst': np.random.randint(0, 30, n_processes),
        'Time_Quantum': np.random.randint(5, 20, n_processes)
    }
    
    return pd.DataFrame(data)

def load_csv_dataset(file_path: str) -> Tuple[pd.DataFrame, Optional[list]]:
    """
    Load and validate a CSV dataset.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Tuple of (DataFrame, error_list)
    """
    errors = []
    required_columns = {'Arrival_Time', 'Burst_Time', 'Priority', 'Prev_Burst', 'Time_Quantum'}
    
    try:
        df = pd.read_csv(file_path)
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Check for required columns
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            return None, errors
        
        # Select only required columns
        df = df[list(required_columns)]
        
        # Validate data types and values
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                errors.append(f"Column '{col}' contains non-numeric values")
                return None, errors
        
        # Validate ranges
        if (df['Arrival_Time'] < 0).any():
            errors.append("Arrival_Time must be non-negative")
        if (df['Burst_Time'] <= 0).any():
            errors.append("Burst_Time must be positive")
        if (df['Priority'] < 1).any() or (df['Priority'] > 5).any():
            errors.append("Priority must be between 1 and 5")
        
        if errors:
            return None, errors
        
        return df, None
        
    except FileNotFoundError:
        errors.append(f"File not found: {file_path}")
    except Exception as e:
        errors.append(f"Error loading CSV: {str(e)}")
    
    return None, errors

def create_sample_csv(output_path: str = "sample_processes.csv", n_samples: int = 50):
    """Create a sample CSV file for reference."""
    df = generate_synthetic_dataset(n_samples)
    df.to_csv(output_path, index=False)
    return output_path
