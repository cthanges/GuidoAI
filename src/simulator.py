import time
import pandas as pd
from typing import Iterator, Optional, List

# Two simulator classes for real-time analysis
class BasicSimulator: # Replay lap-by-lap race data (one lap at a time)
    def __init__(self, laps_df, speed: float = 1.0):
        # Initialize the simulator
        self.laps = laps_df.reset_index(drop=True)
        self.pos = 0
        self.speed = float(speed) if speed > 0 else 1.0

    def has_next(self) -> bool:
        # Check if there are more laps are available
        return self.pos < len(self.laps)

    def next(self):
        # Stop if there are no more laps
        if not self.has_next():
            raise StopIteration
        # Get the current lap and advance to the next
        row = self.laps.iloc[self.pos]
        self.pos += 1

        return row

    def replay(self, delay_callback=None) -> Iterator[object]:
        # Yield the laps one-by-one with delays
        delay = delay_callback or time.sleep

        while self.has_next():
            row = self.next()
            yield row
            delay(max(0.01, 1.0 / self.speed)) # Basic pacing: 1.0 / speed seconds between laps

class TelemetrySimulator: # Replay high-frequency telemetry data
    def __init__(self, telemetry_df: pd.DataFrame, speed: float = 1.0, aggregate_by_lap: bool = True, sample_rate_hz: float = 10.0):
        # Initialize the simulator
        time_col = 'meta_time' if 'meta_time' in telemetry_df.columns else 'timestamp' # Sort data chronologically (meta_time preferred)

        self.data = telemetry_df.sort_values(time_col).reset_index(drop=True)
        self.pos = 0
        self.speed = float(speed) if speed > 0 else 1.0
        self.aggregate_by_lap = aggregate_by_lap
        self.sample_rate_hz = sample_rate_hz
        self.is_long_format = 'telemetry_name' in self.data.columns # Use long format if 'telemetry_name' column exists

        # Pre-aggregate by lap if requested and format allows
        if aggregate_by_lap and 'lap' in self.data.columns:
            self._prepare_lap_aggregates()
    
    def _prepare_lap_aggregates(self):
        # Pre-compute lap-level statistics for faster replay
        if self.is_long_format:
            from src.telemetry_loader import telemetry_to_wide_format
            self.data = telemetry_to_wide_format(self.data) # Convert data to wide format first (one column per parameter)
        
        # Group data depending on if there are multiple vehicles or single vehicle
        group_cols = ['vehicle_id', 'lap'] if 'vehicle_id' in self.data.columns else ['lap']

        # Find all numeric columns for aggregation (grouping columns not included)
        numeric_cols = self.data.select_dtypes(include=['number']).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in group_cols]
        
        if numeric_cols:
            self.lap_data = self.data.groupby(group_cols)[numeric_cols].agg(['mean', 'max', 'min', 'std']).reset_index()

            # Flatten column names (Instead of 'Speed', 'mean' use 'Speed_mean')
            self.lap_data.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in self.lap_data.columns.values]
        else:
            self.lap_data = self.data.groupby(group_cols).size().reset_index(name='count')
    
    def has_next(self) -> bool:
        # Check if there's more data available
        data_source = self.lap_data if self.aggregate_by_lap and hasattr(self, 'lap_data') else self.data

        return self.pos < len(data_source)
    
    def next(self):
        # Stop if there's no more data
        if not self.has_next():
            raise StopIteration
        
        # Get the next aggregate or telemetry row and advance to the next
        data_source = self.lap_data if self.aggregate_by_lap and hasattr(self, 'lap_data') else self.data
        row = data_source.iloc[self.pos]
        self.pos += 1

        return row
    
    def replay(self, delay_callback=None) -> Iterator[pd.Series]:
        # Yield data with appropriate delays
        delay = delay_callback or time.sleep
        
        if self.aggregate_by_lap:
            # Lap-by-lap replay (similar to BasicSimulator)
            while self.has_next():
                row = self.next()
                yield row
                delay(max(0.01, 1.0 / self.speed))
        else:
            # High-frequency replay
            while self.has_next():
                row = self.next()
                yield row
                delay(max(0.001, 1.0 / (self.sample_rate_hz * self.speed))) # Delay based on sample rate
    
    def get_lap_summary(self, lap_number: int, vehicle_id: Optional[str] = None) -> pd.DataFrame:
        # Extract statistics for a specific lap
        if not hasattr(self, 'lap_data'):
            self._prepare_lap_aggregates()
        
        # Filter by specific lap and vehicle (if the vehicle ID was provided)
        mask = self.lap_data['lap'] == lap_number
        if vehicle_id and 'vehicle_id' in self.lap_data.columns:
            mask &= self.lap_data['vehicle_id'] == vehicle_id
        
        return self.lap_data[mask]
    
    def get_parameter_history(self, parameter: str, vehicle_id: Optional[str] = None, lap_range: Optional[tuple] = None) -> pd.Series:
        # Extract history for a specific parameter
        df = self.data.copy()
        
        # Filter by vehicle
        if vehicle_id and 'vehicle_id' in df.columns:
            df = df[df['vehicle_id'] == vehicle_id]
        
        # Filter by lap range
        if lap_range and 'lap' in df.columns:
            min_lap, max_lap = lap_range
            df = df[(df['lap'] >= min_lap) & (df['lap'] <= max_lap)]
        
        # Extract the parameter (format-specific)
        if self.is_long_format:
            df = df[df['telemetry_name'] == parameter]
            return df['telemetry_value']
        else:
            # Wide format here
            if parameter in df.columns:
                return df[parameter]
        
        return pd.Series(dtype=float)