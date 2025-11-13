"""
Telemetry data loader with robust handling of known data quality issues.

Handles:
- meta_time vs timestamp distinction (meta_time = message received, timestamp = ECU time)
- Vehicle ID parsing (GR86-chassis-carnumber format)
- Lap #32768 detection and correction
- Telemetry parameter validation and units
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Root directories
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(ROOT, 'Datasets')

# Known bad lap number (common ECU error)
INVALID_LAP_NUMBER = 32768


class TelemetryParameter(Enum):
    """Telemetry parameters with units and descriptions."""
    
    # Speed & Drivetrain
    SPEED = ("Speed", "km/h", "Vehicle speed")
    GEAR = ("Gear", "gear", "Current gear selection")
    NMOT = ("nmot", "rpm", "Engine RPM")
    
    # Throttle & Braking
    ATH = ("ath", "%", "Throttle blade position (0=closed, 100=wide open)")
    APS = ("aps", "%", "Accelerator pedal position (0=none, 100=full)")
    PBRAKE_F = ("pbrake_f", "bar", "Front brake pressure")
    PBRAKE_R = ("pbrake_r", "bar", "Rear brake pressure")
    
    # Acceleration & Steering
    ACCX_CAN = ("accx_can", "G", "Forward/backward acceleration (+forward, -braking)")
    ACCY_CAN = ("accy_can", "G", "Lateral acceleration (+left, -right)")
    STEERING_ANGLE = ("Steering_Angle", "degrees", "Steering wheel angle (0=straight, +CW, -CCW)")
    
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
    """Parsed vehicle identifier."""
    raw: str  # Original string (e.g., "GR86-004-78")
    chassis_number: str  # e.g., "004"
    car_number: str  # e.g., "78" (may be "000" if not assigned)
    
    @property
    def is_car_number_assigned(self) -> bool:
        """True if car number is assigned (not 000)."""
        return self.car_number != "000"
    
    @property
    def unique_id(self) -> str:
        """Unique identifier (prefer chassis if car number unassigned)."""
        return f"chassis-{self.chassis_number}" if not self.is_car_number_assigned else self.raw
    
    def __str__(self):
        if self.is_car_number_assigned:
            return f"Car #{self.car_number} (Chassis {self.chassis_number})"
        return f"Chassis {self.chassis_number} (Car # not assigned)"


def parse_vehicle_id(vehicle_id: str) -> Optional[VehicleID]:
    """Parse vehicle ID from format GR86-XXX-YYY.
    
    Args:
        vehicle_id: String like "GR86-004-78"
        
    Returns:
        VehicleID object or None if parse fails
        
    Example:
        >>> v = parse_vehicle_id("GR86-004-78")
        >>> v.chassis_number
        '004'
        >>> v.car_number
        '78'
    """
    if not vehicle_id or not isinstance(vehicle_id, str):
        return None
    
    parts = vehicle_id.split('-')
    if len(parts) != 3 or parts[0] != "GR86":
        return None
    
    return VehicleID(
        raw=vehicle_id,
        chassis_number=parts[1],
        car_number=parts[2]
    )


def is_valid_lap(lap: int) -> bool:
    """Check if lap number is valid (not the common ECU error value).
    
    Args:
        lap: Lap number to validate
        
    Returns:
        False if lap is the known error value (32768), True otherwise
    """
    return lap != INVALID_LAP_NUMBER


def infer_lap_from_timestamp(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.Series:
    """Infer lap numbers from timestamps when lap counter is unreliable.
    
    Uses timestamp ordering to reconstruct lap sequence. Groups by vehicle
    and uses time gaps to detect new laps.
    
    Args:
        df: DataFrame with timestamp column
        timestamp_col: Name of timestamp column
        
    Returns:
        Series with inferred lap numbers (int)
    """
    if timestamp_col not in df.columns:
        return pd.Series([1] * len(df), index=df.index, dtype=int)
    
    # Ensure timestamps are datetime
    df_work = df.copy()
    df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col], errors='coerce')
    
    # Sort by vehicle and timestamp
    if 'vehicle_id' in df_work.columns:
        df_work = df_work.sort_values(['vehicle_id', timestamp_col])
    else:
        df_work = df_work.sort_values(timestamp_col)
    
    # Compute time deltas
    if 'vehicle_id' in df_work.columns:
        df_work['_time_delta'] = df_work.groupby('vehicle_id')[timestamp_col].diff().dt.total_seconds()
    else:
        df_work['_time_delta'] = df_work[timestamp_col].diff().dt.total_seconds()
    
    # New lap when time gap > threshold (e.g., 60 seconds suggests new lap start)
    LAP_GAP_THRESHOLD = 60.0
    df_work['_new_lap'] = (df_work['_time_delta'] > LAP_GAP_THRESHOLD) | (df_work['_time_delta'].isna())
    
    # Cumulative sum gives lap number
    if 'vehicle_id' in df_work.columns:
        inferred_laps = df_work.groupby('vehicle_id')['_new_lap'].cumsum() + 1
    else:
        inferred_laps = df_work['_new_lap'].cumsum() + 1
    
    # Restore original index order and return as int series
    result = pd.Series(index=df.index, dtype=int)
    result.loc[inferred_laps.index] = inferred_laps.values
    result = result.fillna(1).astype(int)
    
    return result


def clean_lap_numbers(df: pd.DataFrame, lap_col: str = 'lap', timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """Clean lap numbers by detecting and replacing invalid values.
    
    Args:
        df: DataFrame with lap and timestamp columns
        lap_col: Name of lap column
        timestamp_col: Name of timestamp column
        
    Returns:
        DataFrame with cleaned lap numbers
    """
    df = df.copy()
    
    if lap_col not in df.columns:
        return df
    
    # Convert lap to numeric, coercing errors
    df[lap_col] = pd.to_numeric(df[lap_col], errors='coerce')
    
    # Detect invalid laps
    invalid_mask = (df[lap_col] == INVALID_LAP_NUMBER) | df[lap_col].isna()
    
    if invalid_mask.sum() > 0:
        print(f"Warning: Found {invalid_mask.sum()} invalid lap numbers (lap #{INVALID_LAP_NUMBER} or NaN). Attempting to infer from timestamps.")
        
        # Infer laps from timestamps
        inferred = infer_lap_from_timestamp(df, timestamp_col)
        
        # Update invalid laps with inferred values
        df[lap_col] = df[lap_col].astype('Int64')  # Use nullable int
        df.loc[invalid_mask, lap_col] = inferred[invalid_mask].astype('Int64')
    
    return df


def list_telemetry_files() -> List[str]:
    """Find all telemetry CSV files in Datasets/.
    
    Returns:
        List of absolute paths to telemetry files
    """
    pattern = os.path.join(DATASETS_DIR, '**', '*telemetry*.*')
    files = glob.glob(pattern, recursive=True)
    return [f for f in files if f.lower().endswith('.csv') and '__MACOSX' not in f]


def load_telemetry(path: str, clean_data: bool = True) -> pd.DataFrame:
    """Load telemetry CSV with robust error handling.
    
    Args:
        path: Path to telemetry CSV
        clean_data: If True, apply data quality fixes (lap cleaning, timestamp parsing)
        
    Returns:
        DataFrame with telemetry data
        
    Notes:
        - meta_time: When message was received (reliable)
        - timestamp: ECU time (may be inaccurate)
        - Prefer meta_time for ordering when available
    """
    df = pd.read_csv(path, dtype={'lap': str, 'vehicle_number': str, 'car_number': str})
    
    if not clean_data:
        return df
    
    # Parse timestamps
    for time_col in ['meta_time', 'timestamp']:
        if time_col in df.columns:
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    
    # Clean lap numbers
    df = clean_lap_numbers(df, lap_col='lap', timestamp_col='meta_time' if 'meta_time' in df.columns else 'timestamp')
    
    # Sort by reliable timestamp (prefer meta_time)
    sort_col = 'meta_time' if 'meta_time' in df.columns else 'timestamp'
    if sort_col in df.columns:
        df = df.sort_values(sort_col)
    
    return df.reset_index(drop=True)


def get_vehicle_telemetry(df: pd.DataFrame, vehicle_id: str, 
                          parameter: Optional[str] = None,
                          lap_range: Optional[Tuple[int, int]] = None) -> pd.DataFrame:
    """Extract telemetry for a specific vehicle and optional parameter/lap range.
    
    Args:
        df: Telemetry DataFrame
        vehicle_id: Vehicle identifier (e.g., "GR86-004-78" or just chassis "004")
        parameter: Optional telemetry parameter name to filter
        lap_range: Optional (min_lap, max_lap) tuple to filter laps
        
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    
    # Filter by vehicle
    if 'vehicle_id' in result.columns:
        # Support partial matching (e.g., chassis only)
        result = result[result['vehicle_id'].str.contains(vehicle_id, na=False)]
    
    # Filter by parameter
    if parameter and 'telemetry_name' in result.columns:
        result = result[result['telemetry_name'] == parameter]
    
    # Filter by lap range
    if lap_range and 'lap' in result.columns:
        min_lap, max_lap = lap_range
        result = result[(result['lap'] >= min_lap) & (result['lap'] <= max_lap)]
    
    return result.reset_index(drop=True)


def get_available_parameters(df: pd.DataFrame) -> List[str]:
    """Get list of available telemetry parameters in dataset.
    
    Args:
        df: Telemetry DataFrame
        
    Returns:
        Sorted list of unique parameter names
    """
    if 'telemetry_name' not in df.columns:
        return []
    return sorted(df['telemetry_name'].dropna().unique().tolist())


def get_vehicle_ids(df: pd.DataFrame) -> List[VehicleID]:
    """Extract and parse all vehicle IDs from dataset.
    
    Args:
        df: DataFrame with vehicle_id column
        
    Returns:
        List of parsed VehicleID objects
    """
    if 'vehicle_id' not in df.columns:
        return []
    
    raw_ids = df['vehicle_id'].dropna().unique().tolist()
    parsed = [parse_vehicle_id(vid) for vid in raw_ids]
    return [v for v in parsed if v is not None]


def telemetry_to_wide_format(df: pd.DataFrame, 
                             index_cols: List[str] = ['meta_time', 'vehicle_id', 'lap']) -> pd.DataFrame:
    """Convert long-format telemetry (one row per parameter) to wide format (one row per timestamp).
    
    Args:
        df: Long-format telemetry DataFrame
        index_cols: Columns to use as index (timestamp, vehicle, lap)
        
    Returns:
        Wide-format DataFrame with parameters as columns
        
    Example:
        Input:
            meta_time | vehicle_id | lap | telemetry_name | telemetry_value
            10:00:00  | GR86-004   | 1   | Speed          | 150
            10:00:00  | GR86-004   | 1   | aps            | 85
            
        Output:
            meta_time | vehicle_id | lap | Speed | aps
            10:00:00  | GR86-004   | 1   | 150   | 85
    """
    if 'telemetry_name' not in df.columns or 'telemetry_value' not in df.columns:
        return df
    
    # Keep only relevant columns
    cols = index_cols + ['telemetry_name', 'telemetry_value']
    available_cols = [c for c in cols if c in df.columns]
    df_subset = df[available_cols].copy()
    
    # Pivot
    pivot_df = df_subset.pivot_table(
        index=[c for c in index_cols if c in df_subset.columns],
        columns='telemetry_name',
        values='telemetry_value',
        aggfunc='first'  # Take first value if duplicates
    ).reset_index()
    
    return pivot_df


def validate_telemetry_quality(df: pd.DataFrame) -> Dict[str, any]:
    """Run data quality checks on telemetry data.
    
    Returns:
        Dictionary with quality metrics and warnings
    """
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
