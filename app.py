import streamlit as st
import time
import pandas as pd
from src import data_loader
from src.simulator import SimpleSimulator, TelemetrySimulator
from src.analytics.pit_strategy import recommend_pit, estimate_degradation_from_telemetry
from src.analytics.caution_handler import recommend_under_caution
from src.analytics.anomaly_detection import detect_all_anomalies, get_anomaly_summary
from src import telemetry_loader


st.set_page_config(page_title="Race Engineer MVP", layout='wide')

st.title('Race Engineer — Streamlit MVP')

with st.sidebar:
    st.header('Dataset')
    
    # File type selector
    file_type = st.radio('Data Type', ['Lap Times', 'Telemetry'], index=0)
    
    if file_type == 'Lap Times':
        files = data_loader.list_lap_time_files()
        label = 'Choose lap_time file'
    else:
        files = telemetry_loader.list_telemetry_files()
        label = 'Choose telemetry file'
    
    if files:
        choice = st.selectbox(label, options=files)
    else:
        st.info(f'No {file_type.lower()} files auto-detected in Datasets/. You can upload a CSV.')
        choice = st.file_uploader(f'Upload {file_type.lower()} CSV', type=['csv'])

    speed = st.slider('Replay speed (laps/sec)', 0.1, 5.0, 1.0)
    target_stint = st.number_input('Target stint (laps)', min_value=1, max_value=100, value=20)
    pit_cost = st.number_input('Pit time cost (s)', min_value=1.0, max_value=120.0, value=20.0)
    degradation_rate = st.number_input('Tyre degradation (s/lap)', min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    total_race_laps = st.number_input('Total race laps (optional)', min_value=0, max_value=200, value=50)

if not choice:
    st.warning(f'Please select or upload a {file_type.lower()} CSV to begin.')
    st.stop()

# Load data based on file type
if file_type == 'Lap Times':
    if isinstance(choice, str):
        df = data_loader.load_lap_time(choice)
    else:
        df = pd.read_csv(choice)
    
    vehicle_ids = data_loader.vehicle_ids_from_lap_time(df)
    vehicle = None
    if vehicle_ids:
        vehicle = st.selectbox('Vehicle', vehicle_ids)
    else:
        st.warning('No vehicle_id column found in the selected file; the app will use entire file rows.')
    
    if vehicle:
        vdf = data_loader.filter_vehicle_laps(df, vehicle)
    else:
        vdf = df
    
    st.sidebar.markdown('---')
    st.sidebar.write('Rows for selected vehicle: %d' % len(vdf))
    
    sim = SimpleSimulator(vdf, speed=speed)
    use_telemetry = False
    telemetry_df = None
    
else:  # Telemetry mode
    if isinstance(choice, str):
        df = telemetry_loader.load_telemetry(choice, clean_data=True)
    else:
        df = pd.read_csv(choice)
    
    # Get vehicles from telemetry
    vehicle_objs = telemetry_loader.get_vehicle_ids(df)
    if vehicle_objs:
        vehicle_options = [v.raw for v in vehicle_objs]
        vehicle = st.selectbox('Vehicle', vehicle_options)
    else:
        st.warning('No vehicle_id column found in the selected file.')
        vehicle = None
    
    if vehicle:
        vdf = telemetry_loader.get_vehicle_telemetry(df, vehicle)
    else:
        vdf = df
    
    st.sidebar.markdown('---')
    st.sidebar.write('Telemetry rows for selected vehicle: %d' % len(vdf))
    
    # Create telemetry simulator (aggregate by lap for now)
    sim = TelemetrySimulator(vdf, speed=speed, aggregate_by_lap=True)
    use_telemetry = True
    telemetry_df = df  # Keep full telemetry for degradation estimation

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Replay')
    placeholder = st.empty()
    step_button = st.button('Step one lap')
    run_button = st.button('Run replay')

with col2:
    st.subheader('Pit Strategy')
    info_box = st.empty()

# Add anomaly detection box for telemetry mode
if use_telemetry:
    st.markdown('---')
    st.subheader('Anomaly Detection')
    anomaly_box = st.empty()
    run_anomaly_check = st.button('Run Anomaly Check')

last_pit_lap = 0
last_laps = []
current_lap = 0

if 'sim_pos' not in st.session_state:
    st.session_state['sim_pos'] = 0

# Compute telemetry-based degradation if available
if use_telemetry and vehicle and telemetry_df is not None:
    try:
        auto_degradation = estimate_degradation_from_telemetry(telemetry_df, vehicle)
        st.sidebar.info(f'Auto-detected degradation: {auto_degradation:.3f} s/lap (from telemetry)')
        # Override sidebar value
        degradation_rate = auto_degradation
    except Exception as e:
        st.sidebar.warning(f'Could not auto-detect degradation: {str(e)}')

def render_row(row):
    # display some useful fields
    timestamp = row.get('timestamp') if 'timestamp' in row.index else None
    
    # Handle different data formats
    if 'lap' in row.index:
        lap_val = row['lap']
        if pd.notna(lap_val) and str(lap_val).replace('.','').isdigit():
            lap = int(float(lap_val))
        else:
            lap = st.session_state['sim_pos']
    else:
        lap = st.session_state['sim_pos']
    
    # For lap time data
    val = row.get('value') if 'value' in row.index else None
    
    # For telemetry aggregated data, try to get speed or lap time proxy
    if val is None and use_telemetry:
        # Try common aggregate columns
        for col in ['Speed_mean', 'Speed', 'lap_time', 'value']:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                break
    
    return lap, timestamp, val


if step_button or run_button:
    try:
        if run_button:
            # run until end
            for row in sim.replay(delay_callback=time.sleep):
                lap, ts, val = render_row(row)
                st.session_state['sim_pos'] += 1
                # update last laps
                try:
                    last_laps.append(float(row.get('value', 0)))
                except Exception:
                    pass
                # limit history
                last_laps = last_laps[-5:]
                # compute recommendation with remaining laps
                remaining = total_race_laps - lap if total_race_laps > lap else None
                rec = recommend_pit(
                    lap, last_pit_lap, last_laps, 
                    target_stint=target_stint, 
                    pit_time_cost=pit_cost,
                    remaining_laps=remaining,
                    degradation_per_lap=degradation_rate
                )
                placeholder.metric('Lap', lap, delta=None)
                info_box.json(rec)
                # Attempt a safe rerun. Newer/older Streamlit builds sometimes
                # don't expose `st.experimental_rerun`. Use it when present,
                # otherwise raise the internal RerunException as a fallback.
                try:
                    # preferred public API (may not exist on some builds)
                    st.experimental_rerun()
                except Exception:
                    try:
                        # internal API fallback (works in many Streamlit versions)
                        from streamlit.runtime.scriptrunner.script_runner import RerunException

                        raise RerunException()
                    except Exception:
                        # Last-resort: set a flag so the UI can react without hard rerun.
                        st.session_state['_rerun_requested'] = True
        else:
            row = sim.next()
            lap, ts, val = render_row(row)
            st.session_state['sim_pos'] += 1
            try:
                last_laps.append(float(row.get('value', 0)))
            except Exception:
                pass
            last_laps = last_laps[-5:]
            remaining = total_race_laps - lap if total_race_laps > lap else None
            rec = recommend_pit(
                lap, last_pit_lap, last_laps, 
                target_stint=target_stint, 
                pit_time_cost=pit_cost,
                remaining_laps=remaining,
                degradation_per_lap=degradation_rate
            )
            placeholder.metric('Lap', lap, delta=None)
            info_box.json(rec)
    except StopIteration:
        st.info('Replay finished')

st.markdown('---')
st.subheader('Caution handling')
if st.button('Simulate Caution Now'):
    # get last recommendation from info_box by recomputing from state
    pos = st.session_state.get('sim_pos', 0)
    if pos > 0 and pos <= len(vdf):
        # recompute using last available lap
        row = vdf.iloc[min(pos - 1, len(vdf)-1)]
        if 'lap' in row.index:
            lap_val = row['lap']
            lap = int(float(lap_val)) if pd.notna(lap_val) and str(lap_val).replace('.','').isdigit() else pos
        else:
            lap = pos
        remaining = total_race_laps - lap if total_race_laps > lap else None
        rec = recommend_pit(
            lap, last_pit_lap, last_laps, 
            target_stint=target_stint, 
            pit_time_cost=pit_cost,
            remaining_laps=remaining,
            degradation_per_lap=degradation_rate
        )
        caution = recommend_under_caution(rec, pit_time_cost=pit_cost)
        st.json({'recommendation': rec, 'caution_decision': caution})
    else:
        st.warning('No lap in flight to base a caution decision on')

# Anomaly detection (telemetry mode only)
if use_telemetry and run_anomaly_check and vehicle and telemetry_df is not None:
    with st.spinner('Running anomaly detection...'):
        # Convert to wide format if needed
        check_df = telemetry_df.copy()
        if 'telemetry_name' in check_df.columns:
            check_df = telemetry_loader.telemetry_to_wide_format(check_df)
        
        # Run checks
        anomaly_results = detect_all_anomalies(check_df, vehicle_id=vehicle)
        summary = get_anomaly_summary(anomaly_results)
        
        anomaly_box.markdown(f"**Anomaly Detection Results**")
        anomaly_box.metric('Total Anomalies', summary['total_anomalies'])
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric('🚨 Critical', summary['by_severity']['critical'])
        col_b.metric('⚠️ Warnings', summary['by_severity']['warning'])
        col_c.metric('ℹ️ Info', summary['by_severity']['info'])
        
        if summary['most_severe']:
            st.markdown('**Most Severe Anomalies:**')
            for anomaly in summary['most_severe']:
                st.warning(str(anomaly))
