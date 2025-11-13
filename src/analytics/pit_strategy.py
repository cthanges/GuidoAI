from typing import List, Dict, Optional
import numpy as np
import pandas as pd


def estimate_degradation_from_telemetry(telemetry_df: pd.DataFrame, 
                                        vehicle_id: str,
                                        early_laps: tuple = (1, 5),
                                        late_laps: tuple = (15, 20)) -> float:
    """Estimate tire degradation from actual lateral acceleration telemetry.
    
    Measures loss of lateral grip over stint by comparing max lateral G in early vs late laps.
    
    Args:
        telemetry_df: Telemetry DataFrame (wide format preferred)
        vehicle_id: Vehicle identifier
        early_laps: (min, max) lap range for baseline performance
        late_laps: (min, max) lap range for degraded performance
        
    Returns:
        Estimated degradation rate (seconds/lap). Returns 0.15 as fallback if data insufficient.
    """
    try:
        from src.telemetry_loader import get_vehicle_telemetry
        
        # Get lateral accel for early laps
        early = get_vehicle_telemetry(telemetry_df, vehicle_id, 
                                      parameter='accy_can', lap_range=early_laps)
        # Get lateral accel for late laps
        late = get_vehicle_telemetry(telemetry_df, vehicle_id, 
                                     parameter='accy_can', lap_range=late_laps)
        
        if len(early) == 0 or len(late) == 0:
            return 0.15  # Fallback to default
        
        # Extract values
        early_values = pd.to_numeric(early['telemetry_value'], errors='coerce').dropna()
        late_values = pd.to_numeric(late['telemetry_value'], errors='coerce').dropna()
        
        if len(early_values) < 10 or len(late_values) < 10:
            return 0.15  # Insufficient data
        
        # Max lateral G (absolute value - can be left or right)
        early_max_g = early_values.abs().quantile(0.95)  # 95th percentile for robustness
        late_max_g = late_values.abs().quantile(0.95)
        
        # Loss of grip in G's
        grip_loss = early_max_g - late_max_g
        
        # Convert to lap time degradation (rough approximation: 0.1G loss ≈ 0.3s/lap)
        # Based on typical corner-limited behavior
        lap_time_delta = grip_loss * 3.0
        
        # Normalize by lap count difference
        lap_diff = (late_laps[0] + late_laps[1]) / 2 - (early_laps[0] + early_laps[1]) / 2
        if lap_diff > 0:
            degradation_rate = lap_time_delta / lap_diff
        else:
            degradation_rate = 0.15
        
        # Clamp to reasonable range (0.05 - 0.5 s/lap)
        return max(0.05, min(0.5, degradation_rate))
        
    except Exception as e:
        # Fallback on any error
        return 0.15


def recommend_pit(current_lap: int,
                  last_pit_lap: int,
                  last_laps_seconds: List[float],
                  target_stint: int = 20,
                  pit_time_cost: float = 20.0,
                  remaining_laps: Optional[int] = None,
                  degradation_per_lap: float = 0.15) -> Dict:
    """Multi-lap pit window optimizer that computes expected time-to-finish for candidate strategies.

    Strategy:
    1. Evaluate multiple candidate pit laps (current + 1 to current + window_size).
    2. For each candidate, compute expected total time using:
       - Tyre degradation model (lap time increases linearly with tyre age)
       - Fresh tyre benefit (reset degradation after pit)
       - Pit time cost
    3. Return the pit lap that minimizes total expected time.

    Args:
        current_lap: Current lap number
        last_pit_lap: Lap when last pit occurred
        last_laps_seconds: Recent lap times (used to estimate baseline)
        target_stint: Target stint length (used as max window if remaining_laps unknown)
        pit_time_cost: Time lost in pit (seconds)
        remaining_laps: Laps remaining in race (if known). If None, uses target_stint as proxy.
        degradation_per_lap: Seconds lost per lap due to tyre wear

    Returns:
        Dict with keys:
        - recommended_lap: Optimal pit lap (or None if no pit recommended)
        - reason: Explanation string
        - score: Expected time saved vs no-pit baseline (negative = time lost)
        - candidates: List of evaluated strategies with expected times
    """
    current_stint = current_lap - last_pit_lap
    
    # Estimate baseline lap time from recent laps
    if len(last_laps_seconds) > 0:
        baseline_lap_time = float(np.mean(last_laps_seconds[-3:]))  # Use last 3 laps
    else:
        baseline_lap_time = 90.0  # Fallback default
    
    # If remaining laps unknown, use target_stint as conservative estimate for window
    if remaining_laps is None:
        remaining_laps = max(target_stint - current_stint, 5)
    
    # Don't evaluate if very few laps remain (pit won't pay back)
    if remaining_laps < 3:
        return {
            "recommended_lap": None,
            "reason": "too_few_laps_remaining",
            "score": 0.0,
            "candidates": []
        }
    
    # Evaluate candidate pit laps: from next lap up to a reasonable window
    window_size = min(10, remaining_laps - 2)  # Don't pit in last 2 laps
    candidate_laps = range(current_lap + 1, current_lap + window_size + 1)
    
    candidates = []
    
    # Strategy 0: No pit (baseline)
    no_pit_time = _compute_stint_time(
        baseline_lap_time=baseline_lap_time,
        stint_start_age=current_stint,
        num_laps=remaining_laps,
        degradation_per_lap=degradation_per_lap
    )
    
    # Strategy 1+: Pit at each candidate lap
    for pit_lap in candidate_laps:
        laps_before_pit = pit_lap - current_lap
        laps_after_pit = remaining_laps - laps_before_pit
        
        # Time before pit (continue on worn tyres)
        time_before = _compute_stint_time(
            baseline_lap_time=baseline_lap_time,
            stint_start_age=current_stint,
            num_laps=laps_before_pit,
            degradation_per_lap=degradation_per_lap
        )
        
        # Time after pit (fresh tyres)
        time_after = _compute_stint_time(
            baseline_lap_time=baseline_lap_time,
            stint_start_age=0,  # Fresh tyres
            num_laps=laps_after_pit,
            degradation_per_lap=degradation_per_lap
        )
        
        total_time = time_before + pit_time_cost + time_after
        delta_vs_no_pit = total_time - no_pit_time
        
        candidates.append({
            "pit_lap": pit_lap,
            "expected_time": round(total_time, 2),
            "delta_vs_no_pit": round(delta_vs_no_pit, 2)
        })
    
    # Find optimal strategy
    if not candidates:
        return {
            "recommended_lap": None,
            "reason": "no_valid_candidates",
            "score": 0.0,
            "candidates": []
        }
    
    best = min(candidates, key=lambda c: c["expected_time"])
    time_saved = -best["delta_vs_no_pit"]  # Positive = time saved
    
    # Only recommend pit if it saves time
    if best["delta_vs_no_pit"] < -0.5:  # At least 0.5s benefit
        return {
            "recommended_lap": best["pit_lap"],
            "reason": "optimal_window",
            "score": round(time_saved, 2),
            "candidates": candidates,
            "no_pit_time": round(no_pit_time, 2)
        }
    else:
        return {
            "recommended_lap": None,
            "reason": "no_net_benefit",
            "score": round(time_saved, 2),
            "candidates": candidates,
            "no_pit_time": round(no_pit_time, 2)
        }


def _compute_stint_time(baseline_lap_time: float,
                        stint_start_age: int,
                        num_laps: int,
                        degradation_per_lap: float) -> float:
    """Compute total time for a stint with linear tyre degradation.
    
    Lap time = baseline + (tyre_age * degradation_per_lap)
    
    Args:
        baseline_lap_time: Lap time on fresh tyres
        stint_start_age: Tyre age at start of this stint (laps)
        num_laps: Number of laps in this stint
        degradation_per_lap: Seconds added per lap of tyre age
    
    Returns:
        Total time for stint (seconds)
    """
    total = 0.0
    for i in range(num_laps):
        tyre_age = stint_start_age + i
        lap_time = baseline_lap_time + (tyre_age * degradation_per_lap)
        total += lap_time
    return total
