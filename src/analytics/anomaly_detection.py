import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class AnomalyType(Enum):
    # Detected anomaly types
    MECHANICAL_FAILURE = "Mechanical failure"
    ENGINE_ISSUE = "Engine issue"
    BRAKE_ISSUE = "Brake issue"
    DRIVER_ERROR = "Driver error"
    PERFORMANCE_DROP = "Performance drop"
    SENSOR_ERROR = "Sensor error"

class Severity(Enum):
    # Severity levels for listed anomalies
    CRITICAL = "Critical"
    WARNING = "Warning"
    INFO = "Info" # Informational; this could be normal

@dataclass
class Anomaly:
    # Detected anomaly details
    type: AnomalyType
    severity: Severity
    message: str
    value: float
    threshold: float
    lap: Optional[int] = None
    timestamp: Optional[pd.Timestamp] = None
    parameter: Optional[str] = None
    
    def __str__(self):
        severity_icon = {"Critical": "🚨", "Warning": "⚠️", "Info": "ℹ️"}
        icon = severity_icon.get(self.severity.value, "")
        return f"{icon} [{self.severity.value.upper()}] {self.message}"


def detect_rpm_drop(telemetry_df: pd.DataFrame, 
                   threshold_drop: float = -1000.0,
                   window_size: int = 5) -> List[Anomaly]:
    """Detect sudden RPM drops (potential engine failure).
    
    Args:
        telemetry_df: Wide-format telemetry DataFrame
        threshold_drop: RPM drop threshold (negative value)
        window_size: Number of samples to compute diff over
        
    Returns:
        List of detected anomalies
    """
    anomalies = []
    
    if 'nmot' not in telemetry_df.columns:
        return anomalies
    
    # Compute RPM changes
    rpm_diff = telemetry_df['nmot'].diff(window_size)
    
    # Find significant drops
    drops = rpm_diff < threshold_drop
    
    for idx in telemetry_df[drops].index:
        row = telemetry_df.loc[idx]
        anomalies.append(Anomaly(
            type=AnomalyType.ENGINE_ISSUE,
            severity=Severity.CRITICAL,
            message=f"Sudden RPM drop: {rpm_diff[idx]:.0f} RPM",
            value=rpm_diff[idx],
            threshold=threshold_drop,
            lap=int(row['lap']) if 'lap' in row.index and pd.notna(row['lap']) else None,
            timestamp=row.get('meta_time') or row.get('timestamp'),
            parameter='nmot'
        ))
    
    return anomalies


def detect_brake_lockup(telemetry_df: pd.DataFrame,
                       threshold_pressure: float = 80.0,
                       threshold_decel: float = -1.5) -> List[Anomaly]:
    """Detect potential brake lockups (high pressure + extreme deceleration).
    
    Args:
        telemetry_df: Wide-format telemetry DataFrame
        threshold_pressure: Brake pressure threshold (bar)
        threshold_decel: Deceleration threshold (G's, negative)
        
    Returns:
        List of detected anomalies
    """
    anomalies = []
    
    has_brake = 'pbrake_f' in telemetry_df.columns or 'pbrake_r' in telemetry_df.columns
    has_accel = 'accx_can' in telemetry_df.columns
    
    if not (has_brake and has_accel):
        return anomalies
    
    # Get brake pressure (use front if available, else rear)
    brake_pressure = telemetry_df.get('pbrake_f', telemetry_df.get('pbrake_r'))
    accel_x = telemetry_df['accx_can']
    
    # Detect lockup: high brake pressure + extreme decel
    lockups = (brake_pressure > threshold_pressure) & (accel_x < threshold_decel)
    
    for idx in telemetry_df[lockups].index:
        row = telemetry_df.loc[idx]
        anomalies.append(Anomaly(
            type=AnomalyType.BRAKE_ISSUE,
            severity=Severity.WARNING,
            message=f"Possible brake lockup: {brake_pressure[idx]:.1f} bar, {accel_x[idx]:.2f}G",
            value=accel_x[idx],
            threshold=threshold_decel,
            lap=int(row['lap']) if 'lap' in row.index and pd.notna(row['lap']) else None,
            timestamp=row.get('meta_time') or row.get('timestamp'),
            parameter='pbrake_f'
        ))
    
    return anomalies


def detect_speed_anomaly(telemetry_df: pd.DataFrame,
                        max_reasonable_speed: float = 250.0,
                        min_reasonable_speed: float = -5.0) -> List[Anomaly]:
    """Detect unreasonable speed values (sensor errors).
    
    Args:
        telemetry_df: Wide-format telemetry DataFrame
        max_reasonable_speed: Maximum plausible speed (km/h)
        min_reasonable_speed: Minimum plausible speed (km/h, negative for reverse)
        
    Returns:
        List of detected anomalies
    """
    anomalies = []
    
    if 'Speed' not in telemetry_df.columns:
        return anomalies
    
    speed = pd.to_numeric(telemetry_df['Speed'], errors='coerce')
    
    # Detect unreasonable values
    too_fast = speed > max_reasonable_speed
    too_slow = speed < min_reasonable_speed
    
    for idx in telemetry_df[too_fast | too_slow].index:
        row = telemetry_df.loc[idx]
        anomalies.append(Anomaly(
            type=AnomalyType.SENSOR_ERROR,
            severity=Severity.INFO,
            message=f"Unreasonable speed reading: {speed[idx]:.1f} km/h",
            value=speed[idx],
            threshold=max_reasonable_speed if too_fast[idx] else min_reasonable_speed,
            lap=int(row['lap']) if 'lap' in row.index and pd.notna(row['lap']) else None,
            timestamp=row.get('meta_time') or row.get('timestamp'),
            parameter='Speed'
        ))
    
    return anomalies


def detect_performance_drop(telemetry_df: pd.DataFrame,
                           vehicle_id: str,
                           baseline_laps: tuple = (1, 5),
                           check_laps: tuple = (15, 20),
                           threshold_percent: float = 10.0) -> List[Anomaly]:
    """Detect significant performance drop (damage, tire issue, etc.).
    
    Compares lap times or max speeds between baseline and later laps.
    
    Args:
        telemetry_df: Telemetry DataFrame
        vehicle_id: Vehicle identifier
        baseline_laps: (min, max) reference lap range
        check_laps: (min, max) laps to check for drop
        threshold_percent: Performance drop threshold (%)
        
    Returns:
        List of detected anomalies
    """
    anomalies = []
    
    try:
        from src.telemetry_loader import get_vehicle_telemetry
        
        # Get speed data
        baseline = get_vehicle_telemetry(telemetry_df, vehicle_id,
                                        parameter='Speed', lap_range=baseline_laps)
        check = get_vehicle_telemetry(telemetry_df, vehicle_id,
                                     parameter='Speed', lap_range=check_laps)
        
        if len(baseline) < 10 or len(check) < 10:
            return anomalies
        
        baseline_speed = pd.to_numeric(baseline['telemetry_value'], errors='coerce').quantile(0.95)
        check_speed = pd.to_numeric(check['telemetry_value'], errors='coerce').quantile(0.95)
        
        drop_percent = ((baseline_speed - check_speed) / baseline_speed) * 100
        
        if drop_percent > threshold_percent:
            anomalies.append(Anomaly(
                type=AnomalyType.PERFORMANCE_DROP,
                severity=Severity.WARNING,
                message=f"Performance drop: {drop_percent:.1f}% max speed reduction (laps {check_laps[0]}-{check_laps[1]})",
                value=drop_percent,
                threshold=threshold_percent,
                lap=check_laps[0],
                parameter='Speed'
            ))
    
    except Exception:
        pass
    
    return anomalies


def detect_all_anomalies(telemetry_df: pd.DataFrame,
                        vehicle_id: Optional[str] = None,
                        enable_checks: Optional[List[str]] = None) -> Dict[str, List[Anomaly]]:
    """Run all anomaly detection checks.
    
    Args:
        telemetry_df: Telemetry DataFrame (wide format preferred)
        vehicle_id: Optional vehicle filter
        enable_checks: Optional list of check names to run (default: all)
        
    Returns:
        Dictionary mapping check name to list of anomalies
    """
    if enable_checks is None:
        enable_checks = ['rpm_drop', 'brake_lockup', 'speed_anomaly', 'performance_drop']
    
    # Filter by vehicle if specified
    if vehicle_id and 'vehicle_id' in telemetry_df.columns:
        telemetry_df = telemetry_df[telemetry_df['vehicle_id'].str.contains(vehicle_id, na=False)]
    
    results = {}
    
    if 'rpm_drop' in enable_checks:
        results['rpm_drop'] = detect_rpm_drop(telemetry_df)
    
    if 'brake_lockup' in enable_checks:
        results['brake_lockup'] = detect_brake_lockup(telemetry_df)
    
    if 'speed_anomaly' in enable_checks:
        results['speed_anomaly'] = detect_speed_anomaly(telemetry_df)
    
    if 'performance_drop' in enable_checks and vehicle_id:
        results['performance_drop'] = detect_performance_drop(telemetry_df, vehicle_id)
    
    return results


def get_anomaly_summary(anomaly_results: Dict[str, List[Anomaly]]) -> Dict:
    """Generate summary statistics for detected anomalies.
    
    Args:
        anomaly_results: Dictionary from detect_all_anomalies
        
    Returns:
        Summary dictionary with counts and most severe anomalies
    """
    total = sum(len(anomalies) for anomalies in anomaly_results.values())
    
    all_anomalies = []
    for anomalies in anomaly_results.values():
        all_anomalies.extend(anomalies)
    
    by_severity = {
        'critical': len([a for a in all_anomalies if a.severity == Severity.CRITICAL]),
        'warning': len([a for a in all_anomalies if a.severity == Severity.WARNING]),
        'info': len([a for a in all_anomalies if a.severity == Severity.INFO])
    }
    
    # Get most severe (critical first, then warning)
    critical = [a for a in all_anomalies if a.severity == Severity.CRITICAL]
    warnings = [a for a in all_anomalies if a.severity == Severity.WARNING]
    most_severe = critical[:3] if critical else warnings[:3]
    
    return {
        'total_anomalies': total,
        'by_severity': by_severity,
        'most_severe': most_severe,
        'all_anomalies': all_anomalies
    }