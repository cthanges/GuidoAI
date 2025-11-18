import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(ROOT, 'Datasets') # Directory containing the datasets

INVALID_LAP_NUMBER = 32768 # Known issue about lap count in the datasets

class TelemetryParameter(Enum):    
    # Speed & Drivetrain
    SPEED = ("Speed", "km/h", "Vehicle speed")
    GEAR = ("Gear", "gear", "Current gear selection")
    NMOT = ("nmot", "rpm", "Engine RPM")
    
    # Throttle & Braking
    ATH = ("ath", "%", "Throttle blade position (0% = Fully closed, 100% = Wide open)")
    APS = ("aps", "%", "Accelerator pedal position (0% = No acceleration, 100% = Fully pressed)")
    PBRAKE_F = ("pbrake_f", "bar", "Front brake pressure")
    PBRAKE_R = ("pbrake_r", "bar", "Rear brake pressure")
    
    # Acceleration & Steering
    ACCX_CAN = ("accx_can", "G", "Forward/backward acceleration (positive = accelerating, negative = braking)")
    ACCY_CAN = ("accy_can", "G", "Lateral acceleration (positive = left turn, negative = right turn)")
    STEERING_ANGLE = ("Steering_Angle", "degrees", "Steering wheel angle (0 = straight, negative = counterclockwise, positive = clockwise)")
    
    # Position & Lap Data
    VBOX_LONG = ("VBOX_Long_Minutes", "degrees", "GPS longitude")
    VBOX_LAT = ("VBOX_Lat_Min", "degrees", "GPS latitude")
    LAP_DIST = ("Laptrigger_lapdist_dls", "meters", "Distance from start/finish line")
    
    def __init__(self, param_name: str, unit: str, description: str):
        self.param_name = param_name
        self.unit = unit
        self.description = description

@dataclass
class VehicleID:
    raw: str # Original string (Ex. "GR86-004-78")
    chassis_number: str # Ex. 004
    car_number: str # Ex. 78
    
    @property
    def is_car_number_assigned(self) -> bool:
        # Check if car number is assigned
        return self.car_number != "000" # 000 means that it has not been assigned to ECU
    
    @property
    def unique_id(self) -> str:
        # Return unique identifier (chassis preferred if car number is unassigned)
        return f"chassis-{self.chassis_number}" if not self.is_car_number_assigned else self.raw
    
    def __str__(self):
        # Output one of the following (depending on whether car number is assigned)
        if self.is_car_number_assigned:
            return f"Car #{self.car_number} (Chassis {self.chassis_number})"

        return f"Chassis {self.chassis_number} (Car # not assigned)"

def parse_vehicle_id(vehicle_id: str) -> Optional[VehicleID]:
    # Validate format of the Vehicle ID
    if not vehicle_id or not isinstance(vehicle_id, str):
        return None
    
    # Split on the hyphens
    parts = vehicle_id.split('-')

    # Validate that there are 3 parts to the ID (series-chassis number-car number)
    if len(parts) != 3 or parts[0] != "GR86": # The race data ONLY contains Toyota GR86s
        return None
    
    # Create VehicleID object
    return VehicleID(
        raw=vehicle_id,
        chassis_number=parts[1],
        car_number=parts[2]
    )

def is_valid_lap(lap: int) -> bool:
    return lap != INVALID_LAP_NUMBER # Only use valid lap numbers (not using 32768)

def infer_lap_from_timestamp(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.Series:
    # Check if the 'timestamp' column exists
    if timestamp_col not in df.columns:
        return pd.Series([1] * len(df), index=df.index, dtype=int)
    
    # Prepare a working copy and convert the timestamps to datetime format
    df_work = df.copy()
    df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col], errors='coerce')
    
    # Sort by vehicle ID and timestamp
    if 'vehicle_id' in df_work.columns:
        df_work = df_work.sort_values(['vehicle_id', timestamp_col])
    else:
        df_work = df_work.sort_values(timestamp_col)
    
    # Compute time deltas between consecutive rows
    if 'vehicle_id' in df_work.columns:
        df_work['_time_delta'] = df_work.groupby('vehicle_id')[timestamp_col].diff().dt.total_seconds()
    else:
        df_work['_time_delta'] = df_work[timestamp_col].diff().dt.total_seconds()
    
    # Detect new laps (based on time gaps)
    LAP_GAP_THRESHOLD = 60.0 # when gap > 60 seconds, consider it a new lap
    df_work['_new_lap'] = (df_work['_time_delta'] > LAP_GAP_THRESHOLD) | (df_work['_time_delta'].isna())
    
    # Generate the lap numbers using the cumulative sum
    if 'vehicle_id' in df_work.columns:
        inferred_laps = df_work.groupby('vehicle_id')['_new_lap'].cumsum() + 1
    else:
        inferred_laps = df_work['_new_lap'].cumsum() + 1
    
    # Restore the original index order
    result = pd.Series(index=df.index, dtype=int)
    result.loc[inferred_laps.index] = inferred_laps.values
    result = result.fillna(1).astype(int)
    
    return result

def clean_lap_numbers(df: pd.DataFrame, lap_col: str = 'lap', timestamp_col: str = 'timestamp') -> pd.DataFrame:
    # Copy the dataframe (We don't want to modify the original)
    df = df.copy()
    
    # Check if the 'lap' column exists (exit early if not)
    if lap_col not in df.columns:
        return df
    
    # Convert 'lap' column to numeric
    df[lap_col] = pd.to_numeric(df[lap_col], errors='coerce')
    
    # Detect any invalid laps
    invalid_mask = (df[lap_col] == INVALID_LAP_NUMBER) | df[lap_col].isna()
    
    # Fix the invalid laps
    if invalid_mask.sum() > 0:
        print(f"WARNING: Found {invalid_mask.sum()} invalid lap numbers (lap #{INVALID_LAP_NUMBER} or NaN). Attempting to infer from timestamps.")
        
        # Reconstruct laps from timestamps
        inferred = infer_lap_from_timestamp(df, timestamp_col)
        
        # Update invalid laps with inferred values (use nullable int for NaNs)
        df[lap_col] = df[lap_col].astype('Int64')
        df.loc[invalid_mask, lap_col] = inferred[invalid_mask].astype('Int64')
    
    return df

def list_telemetry_files() -> List[str]:
    # Search recursively for telemetry CSV files in the datasets directory
    pattern = os.path.join(DATASETS_DIR, '**', '*telemetry*.*')
    files = glob.glob(pattern, recursive=True)
    
    # Return list of absolute paths (excluding __MACOSX folders)
    return [f for f in files if f.lower().endswith('.csv') and '__MACOSX' not in f]

def load_telemetry(path: str, clean_data: bool = True) -> pd.DataFrame:
    # Load the telemetry CSV file
    df = pd.read_csv(path, dtype={'lap': str, 'vehicle_number': str, 'car_number': str})

    # Exit early if raw data is requested (i.e. no cleaning)
    if not clean_data:
        return df
    
    # Parse timestamps
    for time_col in ['meta_time', 'timestamp']:
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Call the lap cleaning function ('meta_time' preferred if available)
    df = clean_lap_numbers(df, lap_col='lap', timestamp_col='meta_time' if 'meta_time' in df.columns else 'timestamp')
    
    # Sort by reliable timestamp ('meta_time' preferred if available as time on ECU may not be accurate)
    sort_col = 'meta_time' if 'meta_time' in df.columns else 'timestamp'
    if sort_col in df.columns:
        df = df.sort_values(sort_col)
    
    return df.reset_index(drop=True)

def get_vehicle_telemetry(df: pd.DataFrame, vehicle_id: str, parameter: Optional[str] = None, lap_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    result = df.copy() # Create a working copy
    
    # Filter by vehicle
    if 'vehicle_id' in result.columns:
        # Support partial matching (Ex. chassis only)
        result = result[result['vehicle_id'].str.contains(vehicle_id, na=False)] # NaN values are non-matches, they don't count
    
    # Filter by parameter
    if parameter and 'telemetry_name' in result.columns:
        result = result[result['telemetry_name'] == parameter]
    
    # Filter by lap range
    if lap_range and 'lap' in result.columns:
        min_lap, max_lap = lap_range
        result = result[(result['lap'] >= min_lap) & (result['lap'] <= max_lap)]
    
    return result.reset_index(drop=True)

def get_available_parameters(df: pd.DataFrame) -> List[str]:
    # List all available telemetry parameters in dataset
    if 'telemetry_name' not in df.columns:
        return []

    return sorted(df['telemetry_name'].dropna().unique().tolist())


def get_vehicle_ids(df: pd.DataFrame) -> List[VehicleID]:
    # Extract and parse all unique vehicle IDs from the dataset
    if 'vehicle_id' not in df.columns:
        return []
    
    raw_ids = df['vehicle_id'].dropna().unique().tolist()
    parsed = [parse_vehicle_id(vid) for vid in raw_ids] # Call parser on each ID

    return [v for v in parsed if v is not None]


def telemetry_to_wide_format(df: pd.DataFrame, index_cols: List[str] = ['meta_time', 'vehicle_id', 'lap']) -> pd.DataFrame:
    # Make sure that 'telemetry_name' and 'telemetry_value' columns exist
    if 'telemetry_name' not in df.columns or 'telemetry_value' not in df.columns:
        return df
    
    # Select relevant columns only
    cols = index_cols + ['telemetry_name', 'telemetry_value']
    available_cols = [c for c in cols if c in df.columns]
    df_subset = df[available_cols].copy()
    
    # Create pivot table (this makes it wide format)
    pivot_df = df_subset.pivot_table(
        index=[c for c in index_cols if c in df_subset.columns],
        columns='telemetry_name',
        values='telemetry_value',
        aggfunc='first' # If a duplicate exists, take the first value
    ).reset_index()
    
    return pivot_df

def validate_telemetry_quality(df: pd.DataFrame) -> Dict[str, any]:
    # Run data quality checks on telemetry data (we need to account for all the known issues in the files)
    report = {
        'total_rows': len(df),
        'invalid_laps': 0,
        'missing_timestamps': 0,
        'vehicles_without_car_numbers': [],
        'available_parameters': [],
        'warnings': []
    }
    
    # Check lap quality
    if 'lap' in df.columns:
        invalid = (df['lap'] == INVALID_LAP_NUMBER) | df['lap'].isna()
        report['invalid_laps'] = invalid.sum()
        if report['invalid_laps'] > 0:
            report['warnings'].append(f"{report['invalid_laps']} rows with invalid lap numbers")
    
    # Check timestamps
    for col in ['meta_time', 'timestamp']:
        if col in df.columns:
            missing = df[col].isna().sum()
            report['missing_timestamps'] += missing
    
    # Check vehicle IDs
    vehicles = get_vehicle_ids(df)
    unassigned = [v for v in vehicles if not v.is_car_number_assigned]
    if unassigned:
        report['vehicles_without_car_numbers'] = [v.chassis_number for v in unassigned]
        report['warnings'].append(f"{len(unassigned)} vehicles without assigned car numbers (using chassis only)")
    
    # Get available parameters
    report['available_parameters'] = get_available_parameters(df)
    
    return report