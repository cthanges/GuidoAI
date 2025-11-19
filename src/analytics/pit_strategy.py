from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from src.analytics.traffic_model import TrafficModel, TrafficOpportunity
from src.analytics.caution_handler import analyze_caution_scenarios

def estimate_degradation_from_telemetry(telemetry_df: pd.DataFrame, vehicle_id: str, early_laps: tuple = (1, 5), late_laps: tuple = (15, 20)) -> float:
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
            return 0.15 # Insufficient data
        
        # Measure the max lateral G (absolute value is used since both turns matter)
        early_max_g = early_values.abs().quantile(0.95)  # 95th percentile to avoid noise
        late_max_g = late_values.abs().quantile(0.95)
        
        # Calculate the grip loss of the tires
        grip_loss = early_max_g - late_max_g
        
        # Convert to lap time degradation (rough approximation: 0.1G loss ≈ 0.3s/lap)
        # Based on typical corner-limited behavior
        lap_time_delta = grip_loss * 3.0
        
        # Normalize by lap count difference
        lap_diff = (late_laps[0] + late_laps[1]) / 2 - (early_laps[0] + early_laps[1]) / 2
        if lap_diff > 0: # We don't want to divide by zero or negative
            degradation_rate = lap_time_delta / lap_diff
        else:
            degradation_rate = 0.15
        
        # Clamp to reasonable range (0.05 - 0.5 s/lap)
        return max(0.05, min(0.5, degradation_rate))
        
    except Exception as e:
        return 0.15 # Fallback on any error

def recommend_pit(current_lap: int, last_pit_lap: int,
                  last_laps_seconds: List[float],
                  target_stint: int = 20,
                  pit_time_cost: float = 20.0,
                  remaining_laps: Optional[int] = None,
                  degradation_per_lap: float = 0.15,
                  traffic_model: Optional[TrafficModel] = None,
                  car_number: Optional[int] = None,
                  consider_traffic: bool = True,
                  consider_caution: bool = False,
                  total_laps: Optional[int] = None,
                  cautions_per_race: float = 2.0) -> Dict:
    current_stint = current_lap - last_pit_lap # Calculate current stint length
    
    # Traffic analysis (if available)
    traffic_info = {}
    undercut_opportunities = []
    
    # Debug logging
    import sys
    print(f"DEBUG recommend_pit: traffic_model={traffic_model is not None}, car_number={car_number}, consider_traffic={consider_traffic}, current_lap={current_lap}", file=sys.stderr)
    
    if traffic_model and car_number and consider_traffic:
        current_pos = traffic_model.get_field_position(car_number, current_lap)
        print(f"DEBUG: current_pos = {current_pos}", file=sys.stderr)
        if current_pos:
            traffic_info['field_position'] = current_pos.position
            traffic_info['gap_to_leader'] = round(current_pos.gap_to_leader, 2)
            traffic_info['gap_to_ahead'] = round(current_pos.gap_to_ahead, 2)
            
            # Detect undercut opportunities
            # Need to track other cars' stint lengths (simplified: assume similar to ours)
            laps_since_pit_ahead = {pos.car_number: current_stint for pos in traffic_model.get_running_order(current_lap)}
            
            undercut_opportunities = traffic_model.detect_undercut_opportunities(car_number=car_number, current_lap=current_lap, pit_time_loss=pit_time_cost, degradation_rate=degradation_per_lap, laps_since_pit_ahead=laps_since_pit_ahead)
            
            traffic_info['undercut_opportunities'] = [
                {
                    'target_car': opp.target_car_number,
                    'target_position': opp.target_position,
                    'advantage': round(opp.pit_now_advantage, 2),
                    'confidence': opp.confidence,
                    'description': opp.description
                }
                for opp in undercut_opportunities
            ]
    
    # Estimate baseline lap time from recent laps
    if len(last_laps_seconds) > 0:
        baseline_lap_time = float(np.mean(last_laps_seconds[-3:])) # Use last 3 laps for smoothing out anomalies
    else:
        baseline_lap_time = 90.0 # Fallback default
    
    # If remaining laps unknown, use the target stint as conservative estimate for window
    if remaining_laps is None:
        remaining_laps = max(target_stint - current_stint, 5)
    
    # Don't evaluate if this is true
    if remaining_laps < 3:
        return {
            "recommended_lap": None,
            "reason": "too_few_laps_remaining",
            "score": 0.0,
            "candidates": []
        }
    
    # Evaluate candidate pit laps: from next lap up to a reasonable window
    window_size = min(10, remaining_laps - 2) # Don't pit in last 2 laps
    candidate_laps = range(current_lap + 1, current_lap + window_size + 1)
    
    candidates = []
    
    # No pit (baseline)
    no_pit_time = _compute_stint_time(
        baseline_lap_time=baseline_lap_time,
        stint_start_age=current_stint,
        num_laps=remaining_laps,
        degradation_per_lap=degradation_per_lap
    )
    
    # Pit at each candidate lap
    for pit_lap in candidate_laps:
        laps_before_pit = pit_lap - current_lap
        laps_after_pit = remaining_laps - laps_before_pit
        
        # Time before pit (continue on worn tires)
        time_before = _compute_stint_time(
            baseline_lap_time=baseline_lap_time,
            stint_start_age=current_stint,
            num_laps=laps_before_pit,
            degradation_per_lap=degradation_per_lap
        )
        
        # Time after pit (fresh tires)
        time_after = _compute_stint_time(
            baseline_lap_time=baseline_lap_time,
            stint_start_age=0, # Fresh tires
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
    
    # Find the most optimal strategy
    if not candidates:
        return {
            "recommended_lap": None,
            "reason": "no_valid_candidates",
            "score": 0.0,
            "candidates": []
        }
    
    best = min(candidates, key=lambda c: c["expected_time"])
    time_saved = -best["delta_vs_no_pit"] # Positive = time saved
    
    # Decision logic
    if best["delta_vs_no_pit"] < -0.5: # Only recommend a pit if it saves atleast 0.5 seconds
        result = {
            "recommended_lap": best["pit_lap"],
            "reason": "optimal_window",
            "score": round(time_saved, 2),
            "candidates": candidates,
            "no_pit_time": round(no_pit_time, 2)
        }
        
        # Add traffic info
        result.update(traffic_info)
        
        # Enhance reason if undercut opportunity
        if undercut_opportunities:
            high_conf_undercuts = [o for o in undercut_opportunities if o.confidence == 'high']
            if high_conf_undercuts:
                result['reason'] = 'undercut_opportunity'
                result['undercut_target'] = high_conf_undercuts[0].target_car_number
        
        # Add projected position
        if traffic_model and car_number:
            est_pos, est_gap = traffic_model.estimate_position_after_pit(
                car_number, current_lap, pit_time_cost
            )
            if est_pos:
                result['position_after_pit'] = est_pos
                result['gap_after_pit'] = round(est_gap, 2)
        
        return result
    else:
        result = {
            "recommended_lap": None,
            "reason": "no_net_benefit",
            "score": round(time_saved, 2),
            "candidates": candidates,
            "no_pit_time": round(no_pit_time, 2)
        }
        
        # Add traffic info even when not recommending pit
        result.update(traffic_info)
        
        return result
    
    # Caution probability analysis (if enabled)
    if consider_caution and total_laps and result.get('recommended_lap'):
        try:
            caution_analysis = analyze_caution_scenarios(
                current_lap=current_lap,
                pit_recommendation=result,
                pit_time_cost=pit_time_cost,
                total_laps=total_laps,
                laps_since_pit=current_stint,
                baseline_lap_time=baseline_lap_time,
                degradation_per_lap=degradation_per_lap,
                cautions_per_race=cautions_per_race
            )
            result['caution_analysis'] = caution_analysis
            
            # Potentially override recommendation based on caution probability
            if caution_analysis['recommended_strategy'] == 'wait_for_caution':
                result['reason'] = 'wait_for_caution'
                result['original_recommendation'] = result['recommended_lap']
                result['recommended_lap'] = None # Wait instead
            elif caution_analysis['recommended_strategy'] == 'pit_now':
                result['reason'] = 'pit_now_caution_unlikely'
                result['recommended_lap'] = current_lap + 1 # Pit ASAP
        except Exception as e:
            # Don't fail entire recommendation if caution analysis fails
            result['caution_analysis'] = {'error': str(e)}
    
    return result

def _compute_stint_time(baseline_lap_time: float, stint_start_age: int, num_laps: int, degradation_per_lap: float) -> float:
    # Calculate the stint time given the parameters
    total = 0.0
    for i in range(num_laps):
        tire_age = stint_start_age + i
        lap_time = baseline_lap_time + (tire_age * degradation_per_lap)
        total += lap_time

    return total