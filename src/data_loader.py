import os
import glob
import pandas as pd
from typing import List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(ROOT, 'Datasets') # Directory containing the datasets

def list_lap_time_files() -> List[str]:
    # Search recursively for lap_time files in the datasets directory
    pattern = os.path.join(DATASETS_DIR, '**', '*lap_time*.*')
    files = glob.glob(pattern, recursive=True)

    # Return list of absolute paths
    return [f for f in files if f.lower().endswith('.csv')]

def load_lap_time(path: str) -> pd.DataFrame:
    # Load the data from the lap_time files
    df = pd.read_csv(path, dtype=str)
    
    # Convert 'timestamp' and 'meta_time' columns to datetime for time-based operations
    for c in ['timestamp', 'meta_time']:
        if c in df.columns:
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
            except Exception:
                pass
    
    # Convert 'value' column (lap time in milliseconds) to numeric
    if 'value' in df.columns:
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

def vehicle_ids_from_lap_time(df: pd.DataFrame) -> List[str]:
    # Extract and sort vehicle IDs from the lap_time files
    if 'vehicle_id' in df.columns:
        return sorted(df['vehicle_id'].dropna().unique().tolist())
    return [] # Return an empty list if column doesn't exist

def filter_vehicle_laps(df: pd.DataFrame, vehicle_id: str) -> pd.DataFrame:
    # Filter lap_time file data for specific vehicle ID
    if 'vehicle_id' in df.columns:
        d = df[df['vehicle_id'] == vehicle_id].copy()
    else:
        d = df.copy()
    # Sort by timestamp chronologically if available
    if 'timestamp' in d.columns:
        d = d.sort_values('timestamp')
    return d.reset_index(drop=True)