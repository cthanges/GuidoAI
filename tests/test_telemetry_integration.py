"""Tests for TelemetrySimulator and telemetry integration."""

import pytest
import pandas as pd
import numpy as np
from src.simulator import TelemetrySimulator
from src.analytics.pit_strategy import estimate_degradation_from_telemetry
from src.analytics.anomaly_detection import (
    detect_rpm_drop, detect_brake_lockup, detect_speed_anomaly,
    detect_all_anomalies, Severity, AnomalyType
)


def create_sample_telemetry(n_rows=100, n_laps=5):
    """Create sample telemetry data for testing."""
    return pd.DataFrame({
        'meta_time': pd.date_range('2025-01-01 10:00:00', periods=n_rows, freq='100ms'),
        'vehicle_id': ['GR86-004-78'] * n_rows,
        'lap': np.repeat(range(1, n_laps + 1), n_rows // n_laps),
        'Speed': np.random.uniform(100, 180, n_rows),
        'nmot': np.random.uniform(4000, 6000, n_rows),
        'aps': np.random.uniform(50, 100, n_rows),
        'pbrake_f': np.random.uniform(0, 50, n_rows),
        'accx_can': np.random.uniform(-1.0, 1.0, n_rows),
        'accy_can': np.random.uniform(-1.2, 1.2, n_rows)
    })


def test_telemetry_simulator_initialization():
    """Test TelemetrySimulator basic initialization."""
    df = create_sample_telemetry()
    sim = TelemetrySimulator(df, speed=1.0, aggregate_by_lap=True)
    
    assert sim.has_next()
    assert sim.speed == 1.0
    assert sim.aggregate_by_lap


def test_telemetry_simulator_lap_aggregation():
    """Test that simulator correctly aggregates telemetry by lap."""
    df = create_sample_telemetry(n_rows=100, n_laps=5)
    sim = TelemetrySimulator(df, speed=1.0, aggregate_by_lap=True)
    
    assert hasattr(sim, 'lap_data')
    # Should have aggregated data for 5 laps
    assert len(sim.lap_data) == 5


def test_telemetry_simulator_replay():
    """Test telemetry replay yields correct number of rows."""
    df = create_sample_telemetry(n_rows=50, n_laps=5)
    sim = TelemetrySimulator(df, speed=10.0, aggregate_by_lap=True)
    
    rows = list(sim.replay(delay_callback=lambda x: None))  # No delay for testing
    
    # Should yield 5 lap aggregates
    assert len(rows) == 5


def test_telemetry_simulator_get_lap_summary():
    """Test getting summary for specific lap."""
    df = create_sample_telemetry(n_rows=100, n_laps=5)
    sim = TelemetrySimulator(df, speed=1.0, aggregate_by_lap=True)
    
    lap_3 = sim.get_lap_summary(3, vehicle_id='GR86-004-78')
    
    assert len(lap_3) == 1
    assert lap_3['lap'].iloc[0] == 3


def test_estimate_degradation_from_telemetry():
    """Test degradation estimation from lateral G forces."""
    # Create degrading telemetry (lower accy in later laps)
    df = pd.DataFrame({
        'vehicle_id': ['GR86-004-78'] * 200,
        'lap': list(range(1, 6)) * 40,
        'telemetry_name': ['accy_can'] * 200,
        'telemetry_value': [1.2] * 80 + [1.0] * 80 + [0.9] * 40  # Degrading
    })
    
    deg_rate = estimate_degradation_from_telemetry(
        df, 'GR86-004-78', 
        early_laps=(1, 2), 
        late_laps=(4, 5)
    )
    
    # Should detect some degradation (>= because clamping might hit lower bound)
    assert deg_rate >= 0.05
    assert deg_rate <= 0.5  # Within reasonable range


def test_estimate_degradation_insufficient_data():
    """Test degradation estimation fallback with insufficient data."""
    df = pd.DataFrame({
        'vehicle_id': ['GR86-004-78'] * 5,
        'lap': [1, 1, 2, 2, 3],
        'telemetry_name': ['accy_can'] * 5,
        'telemetry_value': [1.0, 1.1, 1.0, 0.9, 1.0]
    })
    
    deg_rate = estimate_degradation_from_telemetry(df, 'GR86-004-78')
    
    # Should return fallback value
    assert deg_rate == 0.15


def test_detect_rpm_drop():
    """Test RPM drop detection."""
    df = pd.DataFrame({
        'meta_time': pd.date_range('2025-01-01', periods=10, freq='1s'),
        'lap': [1] * 10,
        'nmot': [5000, 5100, 5000, 4900, 3500, 3400, 5000, 5100, 5000, 4900]  # Drop at index 4
    })
    
    anomalies = detect_rpm_drop(df, threshold_drop=-1000)
    
    assert len(anomalies) > 0
    assert anomalies[0].type == AnomalyType.ENGINE_ISSUE
    assert anomalies[0].severity == Severity.CRITICAL


def test_detect_brake_lockup():
    """Test brake lockup detection."""
    df = pd.DataFrame({
        'meta_time': pd.date_range('2025-01-01', periods=10, freq='1s'),
        'lap': [1] * 10,
        'pbrake_f': [10, 20, 30, 85, 90, 85, 30, 20, 10, 5],  # High pressure
        'accx_can': [0.5, 0.3, 0.0, -1.6, -1.8, -1.5, -0.5, 0.0, 0.5, 0.3]  # Extreme decel
    })
    
    anomalies = detect_brake_lockup(df, threshold_pressure=80, threshold_decel=-1.5)
    
    assert len(anomalies) > 0
    assert anomalies[0].type == AnomalyType.BRAKE_ISSUE


def test_detect_speed_anomaly():
    """Test speed sensor error detection."""
    df = pd.DataFrame({
        'meta_time': pd.date_range('2025-01-01', periods=10, freq='1s'),
        'lap': [1] * 10,
        'Speed': [150, 160, 300, 155, 150, 160, 155, 150, -10, 160]  # Unreasonable values
    })
    
    anomalies = detect_speed_anomaly(df, max_reasonable_speed=250)
    
    assert len(anomalies) >= 2  # Should catch both 300 and -10
    assert all(a.type == AnomalyType.SENSOR_ERROR for a in anomalies)


def test_detect_all_anomalies():
    """Test running all anomaly checks together."""
    df = pd.DataFrame({
        'meta_time': pd.date_range('2025-01-01', periods=20, freq='1s'),
        'vehicle_id': ['GR86-004-78'] * 20,
        'lap': [1] * 20,
        'Speed': [150] * 18 + [300, 150],  # One speed anomaly
        'nmot': [5000] * 15 + [3000, 3000, 5000, 5000, 5000],  # One RPM drop
        'pbrake_f': [10] * 17 + [90, 90, 10],  # High brake
        'accx_can': [0.5] * 17 + [-1.8, -1.8, 0.5]  # Extreme decel
    })
    
    results = detect_all_anomalies(df, vehicle_id='GR86-004-78')
    
    # Should detect anomalies in multiple categories
    assert 'rpm_drop' in results
    assert 'brake_lockup' in results
    assert 'speed_anomaly' in results


def test_telemetry_simulator_parameter_history():
    """Test extracting parameter history."""
    df = create_sample_telemetry(n_rows=100, n_laps=5)
    sim = TelemetrySimulator(df, speed=1.0, aggregate_by_lap=False)
    
    speed_history = sim.get_parameter_history('Speed', vehicle_id='GR86-004-78')
    
    assert len(speed_history) > 0
    assert speed_history.dtype in [float, 'float64']
