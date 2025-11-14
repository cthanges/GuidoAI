import streamlit as st
import time
import pandas as pd
from pathlib import Path
from src import data_loader
from src.simulator import SimpleSimulator, TelemetrySimulator
from src.analytics.pit_strategy import recommend_pit, estimate_degradation_from_telemetry
from src.analytics.caution_handler import recommend_under_caution
from src.analytics.anomaly_detection import detect_all_anomalies, get_anomaly_summary
from src.analytics.traffic_model import TrafficModel
from src import telemetry_loader


st.set_page_config(page_title="GuidoAI", layout='wide', page_icon="Logo.png")

# Custom CSS for cornflower blue theme
st.markdown("""
<style>
    /* Primary buttons */
    .stButton > button {
        background-color: #6495ed;
        color: white;
    }
    .stButton > button:hover {
        background-color: #4169e1;
        color: white;
    }
    
    /* Radio buttons (Data Type selector) */
    div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        background-color: #6495ed !important;
    }
    div[role="radiogroup"] label[data-baseweb="radio"] input:checked ~ div:first-child {
        background-color: #6495ed !important;
        border-color: #6495ed !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #6495ed !important;
    }
    div[data-baseweb="slider"] > div > div {
        background-color: #6495ed !important;
    }
    div[data-baseweb="slider"] [role="slider"] {
        background-color: #6495ed !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #6495ed !important;
    }
    div[role="progressbar"] > div {
        background-color: #6495ed !important;
    }
    
    /* Selectbox and other inputs focus */
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div:focus-within,
    .stNumberInput > div > div:focus-within {
        border-color: #6495ed !important;
    }
</style>
""", unsafe_allow_html=True)

# Display logo and title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("Logo.png", width=100)
with col2:
    st.title('GuidoAI — AI Race Strategist')
    st.caption('Pit stop perfection, powered by data 🏁')

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
    
    st.markdown('---')
    st.header('Traffic Model')
    enable_traffic = st.checkbox('Enable traffic analysis', value=True, 
                                   help='Use field position data to detect undercut/overcut opportunities')
    if enable_traffic:
        endurance_files = list(Path('Datasets').rglob('*AnalysisEndurance*.CSV'))
        if endurance_files:
            endurance_choice = st.selectbox('Endurance data file', 
                                            options=[str(f) for f in endurance_files],
                                            help='Select the endurance analysis CSV with lap-by-lap position data')
        else:
            st.info('No endurance analysis files found. Upload one to enable traffic analysis.')
            endurance_choice = st.file_uploader('Upload endurance CSV', type=['csv'])
            enable_traffic = False  # Disable if no file
    
    st.markdown('---')
    st.header('Caution Probability')
    enable_caution = st.checkbox('Enable caution analysis', value=False,
                                  help='Use probabilistic modeling to factor caution likelihood into pit strategy')
    if enable_caution:
        cautions_per_race = st.slider('Expected cautions per race', 0.0, 5.0, 2.0, 0.5,
                                       help='Average number of caution periods expected in this race')
        show_caution_details = st.checkbox('Show detailed scenarios', value=True,
                                           help='Display probability distribution and scenario breakdown')

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

# Add field position display (always create placeholders)
position_box = None
gap_leader_box = None
gap_ahead_box = None
traffic_impact_box = None

if enable_traffic:
    st.markdown('---')
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader('Field Position')
        position_box = st.empty()
    with col_b:
        st.subheader('Gap to Leader')
        gap_leader_box = st.empty()
    with col_c:
        st.subheader('Gap to Car Ahead')
        gap_ahead_box = st.empty()
    
    st.markdown('### Track Position Impact')
    traffic_impact_box = st.empty()

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

# Initialize traffic model if enabled
traffic_model = None
car_number = None

if enable_traffic and endurance_choice:
    try:
        # Load endurance data
        if isinstance(endurance_choice, str):
            endurance_df = pd.read_csv(endurance_choice, sep=';', encoding='utf-8')
        else:
            endurance_df = pd.read_csv(endurance_choice, sep=';', encoding='utf-8')
        
        # Initialize traffic model
        traffic_model = TrafficModel(endurance_df)
        st.sidebar.success(f'✓ Traffic model loaded ({len(endurance_df)} rows)')
        
        # Extract car number from vehicle ID
        if vehicle:
            # Parse vehicle ID to get car number (e.g., "GR86-004-78" -> 78)
            from src.telemetry_loader import parse_vehicle_id
            parsed = parse_vehicle_id(vehicle)
            if parsed and parsed.car_number:
                car_number = parsed.car_number
                st.sidebar.success(f'✓ Traffic model initialized for car #{car_number}')
            else:
                # Try extracting from vehicle string directly
                parts = str(vehicle).split('-')
                if len(parts) >= 2 and parts[-1].isdigit():
                    car_number = int(parts[-1])
                    st.sidebar.success(f'✓ Traffic model initialized for car #{car_number}')
                else:
                    st.sidebar.warning(f'⚠️ Could not extract car number from vehicle ID: {vehicle}')
        else:
            st.sidebar.warning('⚠️ No vehicle selected - traffic model needs a car number')
    except Exception as e:
        st.sidebar.error(f'Failed to load traffic model: {str(e)}')
        import traceback
        st.sidebar.code(traceback.format_exc())
        traffic_model = None
        car_number = None

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


def _update_traffic_display(rec, traffic_model, car_number, lap, 
                            position_box, gap_leader_box, gap_ahead_box, 
                            traffic_impact_box):
    """Update traffic-related display elements."""
    # Skip if boxes not created
    if not all([position_box, gap_leader_box, gap_ahead_box, traffic_impact_box]):
        return
    
    if 'field_position' not in rec:
        return
    
    if 'field_position' in rec:
        position_box.metric('Position', f"P{rec['field_position']}")
    
    if 'gap_to_leader' in rec:
        gap_leader_box.metric('Gap to Leader', f"{rec['gap_to_leader']}s")
    
    if 'gap_to_ahead' in rec:
        gap_ahead_box.metric('Gap Ahead', f"{rec['gap_to_ahead']}s")
    
    # Show position impact
    impact_text = []
    
    if 'position_after_pit' in rec:
        pos_change = rec.get('field_position', 0) - rec['position_after_pit']
        if pos_change > 0:
            impact_text.append(f"⬇️ Will drop to P{rec['position_after_pit']} (lose {pos_change} position{'s' if pos_change > 1 else ''})")
        elif pos_change < 0:
            impact_text.append(f"⬆️ Will gain to P{rec['position_after_pit']} (gain {-pos_change} position{'s' if -pos_change > 1 else ''})")
        else:
            impact_text.append(f"➡️ Will maintain P{rec['position_after_pit']}")
    
    # Show undercut opportunities
    if 'undercut_opportunities' in rec and rec['undercut_opportunities']:
        impact_text.append("\n**🎯 Undercut Opportunities:**")
        for opp in rec['undercut_opportunities']:
            impact_text.append(f"- Car #{opp['target_car']} (P{opp['target_position']}): +{opp['advantage']}s advantage ({opp['confidence']})")
    
    if impact_text:
        traffic_impact_box.markdown('\n'.join(impact_text))
    else:
        traffic_impact_box.info('No significant traffic impact detected')


def _update_caution_display(caution_analysis, show_details=False):
    """Update caution probability display elements."""
    st.markdown('---')
    st.subheader('🚩 Caution Analysis')
    
    # Show recommended strategy
    strategy_emoji = {
        'pit_now': '⛽',
        'wait_for_caution': '⏳',
        'optimal_timing': '🎯'
    }
    
    rec_strategy = caution_analysis.get('recommended_strategy', 'unknown')
    emoji = strategy_emoji.get(rec_strategy, '❓')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric('Strategy', f"{emoji} {rec_strategy.replace('_', ' ').title()}")
    
    with col2:
        confidence = caution_analysis.get('confidence', 'unknown')
        conf_emoji = {'high': '✅', 'medium': '⚠️', 'low': '❌'}.get(confidence, '❓')
        st.metric('Confidence', f"{conf_emoji} {confidence.title()}")
    
    with col3:
        time_saved = caution_analysis.get('expected_time_saved', 0)
        st.metric('Expected Savings', f"{time_saved:.1f}s", 
                 delta=f"{time_saved:.1f}s" if time_saved != 0 else None)
    
    # Show caution probability
    prob_next_10 = caution_analysis.get('caution_probability_next_10_laps', 0)
    st.progress(min(1.0, prob_next_10), text=f"Caution probability (next 10 laps): {prob_next_10*100:.1f}%")
    
    # Show detailed scenarios if enabled
    if show_details and 'scenarios' in caution_analysis:
        st.markdown('#### Scenario Breakdown')
        scenarios = caution_analysis['scenarios']
        
        if scenarios:
            scenario_df = pd.DataFrame([
                {
                    'Laps Until Caution': s['laps_until'],
                    'Probability': f"{s['probability']*100:.1f}%",
                    'Time Saved': f"{s['time_saved']:.1f}s",
                    'Confidence': s['confidence']
                }
                for s in scenarios
            ])
            st.dataframe(scenario_df, use_container_width=True)
        
        # Show strategy comparison
        if 'strategies' in caution_analysis:
            st.markdown('#### Strategy Comparison')
            strat_df = pd.DataFrame(caution_analysis['strategies'])
            strat_df['expected_time'] = strat_df['expected_time'].round(1)
            strat_df['variance'] = strat_df['variance'].round(1)
            st.dataframe(strat_df, use_container_width=True)



            confidence_emoji = {'high': '🔥', 'medium': '⚡', 'low': '💡'}
            emoji = confidence_emoji.get(opp['confidence'], '•')
            impact_text.append(
                f"{emoji} {opp['description']} "
                f"(advantage: {opp['advantage']:.1f}s)"
            )
    
    if impact_text:
        traffic_impact_box.markdown('\n\n'.join(impact_text))
    else:
        traffic_impact_box.info('No significant traffic impact detected')



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
                    degradation_per_lap=degradation_rate,
                    traffic_model=traffic_model,
                    car_number=car_number,
                    consider_traffic=enable_traffic,
                    consider_caution=enable_caution,
                    total_laps=total_race_laps if enable_caution else None,
                    cautions_per_race=cautions_per_race if enable_caution else 2.0
                )
                placeholder.metric('Lap', lap, delta=None)
                info_box.json(rec)
                
                # Display caution analysis if enabled
                if enable_caution and rec.get('caution_analysis'):
                    _update_caution_display(rec['caution_analysis'], 
                                           show_caution_details if enable_caution else False)
                
                # Update traffic displays
                if enable_traffic and traffic_model and car_number:
                    _update_traffic_display(rec, traffic_model, car_number, lap,
                                           position_box, gap_leader_box, gap_ahead_box, 
                                           traffic_impact_box)
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
                degradation_per_lap=degradation_rate,
                traffic_model=traffic_model,
                car_number=car_number,
                consider_traffic=enable_traffic
            )
            placeholder.metric('Lap', lap, delta=None)
            info_box.json(rec)
            
            # Update traffic displays
            if enable_traffic and traffic_model and car_number:
                _update_traffic_display(rec, traffic_model, car_number, lap,
                                       position_box, gap_leader_box, gap_ahead_box, 
                                       traffic_impact_box)
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
            degradation_per_lap=degradation_rate,
            traffic_model=traffic_model,
            car_number=car_number,
            consider_traffic=enable_traffic
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
