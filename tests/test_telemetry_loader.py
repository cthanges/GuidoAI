"""Tests for telemetry data loader and data quality functions."""

import pytest
import pandas as pd
import numpy as np
from src.telemetry_loader import (
    parse_vehicle_id, is_valid_lap, clean_lap_numbers,
    get_vehicle_telemetry, telemetry_to_wide_format,
    validate_telemetry_quality, TelemetryParameter,
    INVALID_LAP_NUMBER
)


def test_parse_vehicle_id_valid():
    """Test parsing valid vehicle ID."""
    vid = parse_vehicle_id("GR86-004-78")
    assert vid is not None
    assert vid.chassis_number == "004"
    assert vid.car_number == "78"
    assert vid.is_car_number_assigned
    assert "Car #78" in str(vid)


def test_parse_vehicle_id_unassigned_car_number():
    """Test parsing vehicle ID with unassigned car number."""
    vid = parse_vehicle_id("GR86-002-000")
    assert vid is not None
    assert vid.chassis_number == "002"
    assert vid.car_number == "000"
    assert not vid.is_car_number_assigned
    assert "Chassis 002" in str(vid)
    assert vid.unique_id == "chassis-002"


def test_parse_vehicle_id_invalid():
    """Test parsing invalid vehicle IDs."""
    assert parse_vehicle_id(None) is None
    assert parse_vehicle_id("") is None
    assert parse_vehicle_id("INVALID") is None
    assert parse_vehicle_id("GR86-004") is None  # Missing part


def test_is_valid_lap():
    """Test lap number validation."""
    assert is_valid_lap(1)
    assert is_valid_lap(50)
    assert not is_valid_lap(INVALID_LAP_NUMBER)  # 32768
    assert not is_valid_lap(32768)


def test_clean_lap_numbers_with_invalid():
    """Test cleaning lap numbers with invalid value."""
    df = pd.DataFrame({
        'lap': [1, 2, 32768, 4, 5],  # lap 3 is invalid
        'meta_time': pd.date_range('2025-01-01 10:00:00', periods=5, freq='2min'),
        'vehicle_id': ['GR86-004-78'] * 5
    })
    
    cleaned = clean_lap_numbers(df)
    
    # Invalid lap should be inferred from timestamp
    assert cleaned['lap'].iloc[2] != 32768
    assert cleaned['lap'].iloc[2] > 0  # Should be a valid lap number


def test_clean_lap_numbers_no_lap_column():
    """Test cleaning when lap column doesn't exist."""
    df = pd.DataFrame({
        'timestamp': pd.date_range('2025-01-01', periods=3, freq='1min'),
        'value': [1, 2, 3]
    })
    
    cleaned = clean_lap_numbers(df)
    assert 'lap' not in cleaned.columns


def test_get_vehicle_telemetry_filter_by_vehicle():
    """Test filtering telemetry by vehicle ID."""
    df = pd.DataFrame({
        'vehicle_id': ['GR86-004-78', 'GR86-004-78', 'GR86-005-99', 'GR86-005-99'],
        'telemetry_name': ['Speed', 'aps', 'Speed', 'aps'],
        'telemetry_value': [150, 85, 140, 90],
        'lap': [1, 1, 1, 1]
    })
    
    result = get_vehicle_telemetry(df, 'GR86-004-78')
    assert len(result) == 2
    assert all(result['vehicle_id'] == 'GR86-004-78')


def test_get_vehicle_telemetry_filter_by_parameter():
    """Test filtering telemetry by parameter."""
    df = pd.DataFrame({
        'vehicle_id': ['GR86-004-78'] * 4,
        'telemetry_name': ['Speed', 'aps', 'Speed', 'aps'],
        'telemetry_value': [150, 85, 155, 88],
        'lap': [1, 1, 2, 2]
    })
    
    result = get_vehicle_telemetry(df, 'GR86-004-78', parameter='Speed')
    assert len(result) == 2
    assert all(result['telemetry_name'] == 'Speed')


def test_get_vehicle_telemetry_filter_by_lap_range():
    """Test filtering telemetry by lap range."""
    df = pd.DataFrame({
        'vehicle_id': ['GR86-004-78'] * 5,
        'telemetry_name': ['Speed'] * 5,
        'telemetry_value': [150, 155, 160, 165, 170],
        'lap': [1, 2, 3, 4, 5]
    })
    
    result = get_vehicle_telemetry(df, 'GR86-004-78', lap_range=(2, 4))
    assert len(result) == 3
    assert result['lap'].min() == 2
    assert result['lap'].max() == 4


def test_telemetry_to_wide_format():
    """Test conversion from long to wide format."""
    df = pd.DataFrame({
        'meta_time': ['2025-01-01 10:00:00'] * 3,
        'vehicle_id': ['GR86-004-78'] * 3,
        'lap': [1, 1, 1],
        'telemetry_name': ['Speed', 'aps', 'nmot'],
        'telemetry_value': [150, 85, 5500]
    })
    df['meta_time'] = pd.to_datetime(df['meta_time'])
    
    wide = telemetry_to_wide_format(df)
    
    # Should have one row with parameters as columns
    assert len(wide) == 1
    assert 'Speed' in wide.columns
    assert 'aps' in wide.columns
    assert 'nmot' in wide.columns
    assert wide['Speed'].iloc[0] == 150


def test_telemetry_to_wide_format_no_telemetry_columns():
    """Test wide format conversion when telemetry columns missing."""
    df = pd.DataFrame({
        'timestamp': [1, 2, 3],
        'value': [100, 200, 300]
    })
    
    wide = telemetry_to_wide_format(df)
    assert wide.equals(df)  # Should return unchanged


def test_validate_telemetry_quality():
    """Test data quality validation."""
    df = pd.DataFrame({
        'lap': [1, 2, 32768, 4],  # One invalid lap
        'meta_time': pd.date_range('2025-01-01', periods=4, freq='1min'),
        'timestamp': pd.date_range('2025-01-01', periods=4, freq='1min'),
        'vehicle_id': ['GR86-004-78', 'GR86-004-78', 'GR86-005-000', 'GR86-005-000'],
        'telemetry_name': ['Speed', 'aps', 'Speed', 'aps'],
        'telemetry_value': [150, 85, 140, 90]
    })
    
    report = validate_telemetry_quality(df)
    
    assert report['total_rows'] == 4
    assert report['invalid_laps'] == 1  # One lap #32768
    assert len(report['vehicles_without_car_numbers']) == 1  # One vehicle with 000
    assert 'Speed' in report['available_parameters']
    assert 'aps' in report['available_parameters']
    assert len(report['warnings']) > 0


def test_telemetry_parameter_enum():
    """Test telemetry parameter enum has correct attributes."""
    speed = TelemetryParameter.SPEED
    assert speed.param_name == "Speed"
    assert speed.unit == "km/h"
    assert "speed" in speed.description.lower()
    
    accx = TelemetryParameter.ACCX_CAN
    assert accx.param_name == "accx_can"
    assert accx.unit == "G"
    assert "acceleration" in accx.description.lower()


def test_telemetry_parameter_enum_coverage():
    """Test that all major parameter groups are covered."""
    params = [p.param_name for p in TelemetryParameter]
    
    # Speed & Drivetrain
    assert "Speed" in params
    assert "Gear" in params
    assert "nmot" in params
    
    # Throttle & Braking
    assert "ath" in params
    assert "aps" in params
    assert "pbrake_f" in params
    assert "pbrake_r" in params
    
    # Acceleration & Steering
    assert "accx_can" in params
    assert "accy_can" in params
    assert "Steering_Angle" in params
    
    # Position
    assert "VBOX_Long_Minutes" in params
    assert "VBOX_Lat_Min" in params
    assert "Laptrigger_lapdist_dls" in params
