import streamlit as st
import time
import pandas as pd
from pathlib import Path
from src import data_loader
from src.simulator import BasicSimulator, TelemetrySimulator
from src.analytics.pit_strategy import recommend_pit, estimate_degradation_from_telemetry
from src.analytics.caution_handler import recommend_under_caution
from src.analytics.anomaly_detection import detect_all_anomalies, get_anomaly_summary
from src.analytics.traffic_model import TrafficModel
from src import telemetry_loader


st.set_page_config(page_title="GuidoPro", layout='wide', page_icon="Logo.png")

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
    st.title('GuidoPro — Professional Race Strategist')
    st.caption('Pit Stop! 🏁')

# Main workflow tabs
tab1, tab2, tab3 = st.tabs(["📁 Setup", "🏁 Race Analysis", "🔧 Advanced Settings"])

# ========== TAB 1: SETUP ==========
with tab1:
    st.header("Step 1: Select Your Data")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🗂️ Data Source")
        file_type = st.radio('What type of data do you have?', 
                            ['Lap Times', 'Telemetry'], 
                            index=0,
                            help="**Lap Times**: Race lap-by-lap data (simpler)\n**Telemetry**: High-frequency sensor data (advanced)")
        
        if file_type == 'Lap Times':
            files = data_loader.list_lap_time_files()
            label = 'Select your race data'
        else:
            files = telemetry_loader.list_telemetry_files()
            label = 'Select your telemetry data'
        
        if files:
            choice = st.selectbox(label, options=files, help="CSV files auto-detected from Datasets/ folder")
        else:
            st.info(f'💡 No files found. Upload your own CSV:')
            choice = st.file_uploader(f'Upload CSV', type=['csv'])
    
    with col_b:
        st.subheader("🏎️ Vehicle Selection")
        
        # Vehicle selection (will be populated after file choice)
        if choice:
            # Load data to get vehicle list
            if file_type == 'Lap Times':
                if isinstance(choice, str):
                    temp_df = data_loader.load_lap_time(choice)
                else:
                    temp_df = pd.read_csv(choice)
                vehicle_ids = data_loader.vehicle_ids_from_lap_time(temp_df)
            else:
                if isinstance(choice, str):
                    temp_df = telemetry_loader.load_telemetry(choice, clean_data=True)
                else:
                    temp_df = pd.read_csv(choice)
                vehicle_objs = telemetry_loader.get_vehicle_ids(temp_df)
                vehicle_ids = [v.raw for v in vehicle_objs] if vehicle_objs else []
            
            if vehicle_ids:
                vehicle = st.selectbox('Which car do you want to analyze?', 
                                      vehicle_ids,
                                      help="Select the vehicle/car number to analyze")
                st.success(f"✅ Ready to analyze **{vehicle}**")
            else:
                st.warning('⚠️ No vehicle IDs found in this file')
                vehicle = None
        else:
            st.info("👈 Select a data file first")
            vehicle = None
    
    if choice and vehicle:
        st.markdown("---")
        st.header("Step 2: Configure Race Parameters")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        
        with col_p1:
            st.subheader("🏁 Race Setup")
            total_race_laps = st.number_input('Total race laps', 
                                             min_value=10, 
                                             max_value=200, 
                                             value=50,
                                             help="How many laps in this race?")
            target_stint = st.number_input('Max stint length (laps)', 
                                          min_value=1, 
                                          max_value=100, 
                                          value=20,
                                          help="Maximum laps on a single set of tires")
        
        with col_p2:
            st.subheader("⚙️ Car Setup")
            pit_cost = st.number_input('Pit stop time (seconds)', 
                                      min_value=5.0, 
                                      max_value=120.0, 
                                      value=20.0,
                                      help="Time lost entering, stopping, and exiting pit lane")
            degradation_rate = st.number_input('Tire degradation (s/lap)', 
                                              min_value=0.0, 
                                              max_value=1.0, 
                                              value=0.15, 
                                              step=0.05,
                                              help="How much slower per lap on worn tires")
        
        with col_p3:
            st.subheader("⚡ Replay Speed")
            speed = st.slider('Laps per second', 
                            0.1, 5.0, 1.0,
                            help="How fast to replay the race simulation")
            st.info(f"💡 **{speed} laps/sec** = {60/speed:.1f} seconds per lap")
        
        st.success("✅ Configuration complete! Go to the **🏁 Race Analysis** tab to start →")
    else:
        st.info("👆 Complete Step 1 first")

# ========== TAB 2: RACE ANALYSIS (Main Features) ==========
with tab2:
    if not choice or not vehicle:
        st.warning("⚠️ Please complete the **Setup** tab first!")
        st.stop()
    
    # Quick settings reminder
    with st.expander("📋 Current Configuration", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Race Laps", total_race_laps)
        col2.metric("Pit Cost", f"{pit_cost}s")
        col3.metric("Degradation", f"{degradation_rate} s/lap")
        col4.metric("Max Stint", f"{target_stint} laps")

# Store in sidebar for compatibility
with st.sidebar:
    st.header('Quick Settings')
    
    # Store all settings in sidebar (hidden/collapsed)
    if 'choice' not in locals() or not choice:
        choice = None
        vehicle = None
        file_type = 'Lap Times'
        speed = 1.0
        target_stint = 20
        pit_cost = 20.0
        degradation_rate = 0.15
        total_race_laps = 50
    
    st.markdown('---')
    st.header('Advanced Features')
    enable_traffic = st.checkbox('Enable traffic analysis', 
                                 value=False, 
                                 help='Detect undercut/overcut opportunities')
    if enable_traffic:
        endurance_files = list(Path('Datasets').rglob('*AnalysisEndurance*.CSV'))
        if endurance_files:
            endurance_choice = st.selectbox('Traffic data file', 
                                          options=[str(f) for f in endurance_files])
        else:
            st.info('No endurance files found')
            endurance_choice = None
            enable_traffic = False
    
    enable_caution = st.checkbox('Enable caution analysis', 
                                 value=False,
                                 help='Predict caution flags probability')
    if enable_caution:
        cautions_per_race = st.slider('Expected cautions', 0.0, 5.0, 2.0, 0.5)
        show_caution_details = st.checkbox('Show details', value=False)

if not choice:
    st.info('👈 Complete the Setup tab to begin')
    st.stop()

# Load data based on file type (moved inside tab2)
with tab2:
    if file_type == 'Lap Times':
        if isinstance(choice, str):
            df = data_loader.load_lap_time(choice)
        else:
            df = pd.read_csv(choice)
        
        if vehicle:
            vdf = data_loader.filter_vehicle_laps(df, vehicle)
        else:
            vdf = df
        
        sim = BasicSimulator(vdf, speed=speed)
        use_telemetry = False
        telemetry_df = None
        
    else:  # Telemetry mode
        if isinstance(choice, str):
            df = telemetry_loader.load_telemetry(choice, clean_data=True)
        else:
            df = pd.read_csv(choice)
        
        if vehicle:
            vdf = telemetry_loader.get_vehicle_telemetry(df, vehicle)
        else:
            vdf = df
        
        sim = TelemetrySimulator(vdf, speed=speed, aggregate_by_lap=True)
        use_telemetry = True
        telemetry_df = df
    
    # Initialize traffic model if enabled
    traffic_model = None
    car_number = None
    
    if enable_traffic and 'endurance_choice' in locals() and endurance_choice:
        try:
            if isinstance(endurance_choice, str):
                endurance_df = pd.read_csv(endurance_choice, sep=';', encoding='utf-8')
            else:
                endurance_df = pd.read_csv(endurance_choice, sep=';', encoding='utf-8')
            
            traffic_model = TrafficModel(endurance_df)
            
            # Extract car number from vehicle ID
            if vehicle:
                from src.telemetry_loader import parse_vehicle_id
                parsed = parse_vehicle_id(vehicle)
                if parsed and parsed.car_number:
                    car_number = parsed.car_number
                else:
                    parts = str(vehicle).split('-')
                    if len(parts) >= 2 and parts[-1].isdigit():
                        car_number = int(parts[-1])
        except Exception as e:
            st.error(f'Failed to load traffic model: {str(e)}')
            traffic_model = None
            car_number = None
    
    # Compute telemetry-based degradation if available
    if use_telemetry and vehicle and telemetry_df is not None:
        try:
            auto_degradation = estimate_degradation_from_telemetry(telemetry_df, vehicle)
            st.info(f'📊 Auto-detected degradation from telemetry: **{auto_degradation:.3f} s/lap**')
            degradation_rate = auto_degradation
        except Exception:
            pass
    
    # Main analysis area
    st.header("Race Simulation")
    
    # Control buttons prominently displayed
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([2, 2, 1, 1])
    
    with col_btn1:
        run_button = st.button('▶️ Auto-Replay', type="primary", use_container_width=True)
    with col_btn2:
        step_button = st.button('➡️ Next Lap', use_container_width=True)
    with col_btn3:
        if st.button('🔄', help="Reset simulation"):
            st.session_state['sim_pos'] = 0
            st.rerun()
    with col_btn4:
        st.metric("Lap", st.session_state.get('sim_pos', 0))
    
    st.markdown("---")
    
    # Analysis results display
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader("📊 Current Lap Data")
        placeholder = st.empty()
    
    with col_right:
        st.subheader("🎯 Pit Strategy Recommendation")
        info_box = st.empty()
    
    # Traffic analysis (if enabled)
    if enable_traffic and traffic_model and car_number:
        st.markdown("---")
        st.subheader("🏎️ Track Position Analysis")
        
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            position_box = st.empty()
        with col_t2:
            gap_leader_box = st.empty()
        with col_t3:
            gap_ahead_box = st.empty()
        
        traffic_impact_box = st.empty()
    else:
        position_box = None
        gap_leader_box = None
        gap_ahead_box = None
        traffic_impact_box = None
    
    # Anomaly detection (telemetry mode)
    if use_telemetry:
        st.markdown("---")
        st.subheader("🔍 Anomaly Detection")
        col_ano1, col_ano2 = st.columns([3, 1])
        with col_ano1:
            anomaly_box = st.empty()
        with col_ano2:
            run_anomaly_check = st.button('🔍 Run Check', use_container_width=True)

# ========== TAB 3: ADVANCED ==========
with tab3:
    st.header("Advanced Settings & Features")
    
    st.subheader("🚩 Caution Flag Simulator")
    st.markdown("Test how your strategy changes under caution conditions")
    
    if st.button('🟡 Simulate Caution Now', type="secondary"):
        pos = st.session_state.get('sim_pos', 0)
        if pos > 0 and 'vdf' in locals() and pos <= len(vdf):
            row = vdf.iloc[min(pos - 1, len(vdf)-1)]
            if 'lap' in row.index:
                lap_val = row['lap']
                lap = int(float(lap_val)) if pd.notna(lap_val) and str(lap_val).replace('.','').isdigit() else pos
            else:
                lap = pos
            
            remaining = total_race_laps - lap if total_race_laps > lap else None
            rec = recommend_pit(
                lap, 0, [], 
                target_stint=target_stint, 
                pit_time_cost=pit_cost,
                remaining_laps=remaining,
                degradation_per_lap=degradation_rate,
                traffic_model=None,
                car_number=None,
                consider_traffic=False
            )
            caution = recommend_under_caution(rec, pit_time_cost=pit_cost)
            st.json({'recommendation': rec, 'caution_decision': caution})
        else:
            st.warning('⚠️ Start the race simulation first')
    
    st.markdown("---")
    st.subheader("📈 Data Export")
    st.markdown("*Coming soon: Export recommendations and analysis results*")
    
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.markdown("""
    **GuidoPro** uses advanced analytics to optimize race strategy:
    - Multi-lap pit window optimization
    - Probabilistic caution modeling
    - Traffic-aware position tracking
    - Real-time anomaly detection
    
    See **USER_GUIDE.md** for detailed documentation.
    """)


last_pit_lap = 0
last_laps = []
current_lap = 0

if 'sim_pos' not in st.session_state:
    st.session_state['sim_pos'] = 0

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

# Simulation logic (keep existing functionality)
if 'step_button' in locals() and 'run_button' in locals() and (step_button or run_button):
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
                    traffic_model=traffic_model if enable_traffic else None,
                    car_number=car_number if enable_traffic else None,
                    consider_traffic=enable_traffic,
                    consider_caution=enable_caution,
                    total_laps=total_race_laps if enable_caution else None,
                    cautions_per_race=cautions_per_race if enable_caution else 2.0
                )
                
                # Display lap and recommendation
                placeholder.write(f"**Lap {lap}** | Time: {val if val else 'N/A'}")
                
                # Format recommendation display
                if rec:
                    # Determine action from reason
                    reason = rec.get('reason', 'unknown')
                    action_map = {
                        'optimal_window': '⛽ Pit Now',
                        'undercut_opportunity': '🎯 Pit Now (Undercut)',
                        'no_net_benefit': '✅ Stay Out',
                        'too_few_laps_remaining': 'ℹ️ Too Late',
                        'no_valid_candidates': 'ℹ️ No Strategy',
                        'wait_for_caution': '⏳ Wait for Caution',
                        'pit_now_caution_unlikely': '⚠️ Pit Immediately'
                    }
                    action = action_map.get(reason, reason.replace('_', ' ').title())
                    
                    rec_display = f"""
                    **Action:** {action}
                    
                    **Reason:** {reason.replace('_', ' ').title()}
                    
                    **Optimal Pit Lap:** {rec.get('recommended_lap', 'N/A')}
                    
                    **Expected Savings:** {rec.get('score', 0):.1f}s
                    """
                    
                    # Color coding based on action urgency:
                    # - Green (success) = Good! Stay out, tires still good
                    # - Red (error) = Action needed, pit recommended
                    # - Yellow (warning) = Informational, no action yet
                    if rec.get('recommended_lap'):
                        # Pit is recommended
                        info_box.error(rec_display)  # Red = action needed
                    elif reason == 'no_net_benefit':
                        # Tires still good, stay out
                        info_box.success(rec_display)  # Green = all good
                    else:
                        # Informational
                        info_box.warning(rec_display)  # Yellow = just info
                else:
                    info_box.info("Calculating...")
                
                # Display caution analysis if enabled
                if enable_caution and rec.get('caution_analysis'):
                    _update_caution_display(rec['caution_analysis'], 
                                           show_caution_details if enable_caution else False)
                
                # Update traffic displays
                if enable_traffic and traffic_model and car_number:
                    _update_traffic_display(rec, traffic_model, car_number, lap,
                                           position_box, gap_leader_box, gap_ahead_box, 
                                           traffic_impact_box)
                
                # Rerun to update display
                try:
                    st.experimental_rerun()
                except Exception:
                    try:
                        from streamlit.runtime.scriptrunner.script_runner import RerunException
                        raise RerunException()
                    except Exception:
                        st.session_state['_rerun_requested'] = True
        else:
            # Single step
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
                traffic_model=traffic_model if enable_traffic else None,
                car_number=car_number if enable_traffic else None,
                consider_traffic=enable_traffic
            )
            
            # Display lap and recommendation
            placeholder.write(f"**Lap {lap}** | Time: {val if val else 'N/A'}")
            
            if rec:
                # Determine action from reason
                reason = rec.get('reason', 'unknown')
                action_map = {
                    'optimal_window': '⛽ Pit Now',
                    'undercut_opportunity': '🎯 Pit Now (Undercut)',
                    'no_net_benefit': '✅ Stay Out',
                    'too_few_laps_remaining': 'ℹ️ Too Late',
                    'no_valid_candidates': 'ℹ️ No Strategy',
                    'wait_for_caution': '⏳ Wait for Caution',
                    'pit_now_caution_unlikely': '⚠️ Pit Immediately'
                }
                action = action_map.get(reason, reason.replace('_', ' ').title())
                
                rec_display = f"""
                **Action:** {action}
                
                **Reason:** {reason.replace('_', ' ').title()}
                
                **Optimal Pit Lap:** {rec.get('recommended_lap', 'N/A')}
                
                **Expected Savings:** {rec.get('score', 0):.1f}s
                """
                
                # Color coding based on action urgency:
                # - Green (success) = Good! Stay out, tires still good
                # - Red (error) = Action needed, pit recommended
                # - Yellow (warning) = Informational, no action yet
                if rec.get('recommended_lap'):
                    # Pit is recommended
                    info_box.error(rec_display)  # Red = action needed
                elif reason == 'no_net_benefit':
                    # Tires still good, stay out
                    info_box.success(rec_display)  # Green = all good
                else:
                    # Informational
                    info_box.warning(rec_display)  # Yellow = just info
            
            # Update traffic displays
            if enable_traffic and traffic_model and car_number:
                _update_traffic_display(rec, traffic_model, car_number, lap,
                                       position_box, gap_leader_box, gap_ahead_box, 
                                       traffic_impact_box)
    except StopIteration:
        st.success('✅ Replay finished!')
        placeholder.info(f"Completed {st.session_state['sim_pos']} laps")
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
        st.success('✅ Replay finished!')
        placeholder.info(f"Completed {st.session_state['sim_pos']} laps")

# Anomaly detection handler (telemetry mode only)
if 'use_telemetry' in locals() and use_telemetry and 'run_anomaly_check' in locals() and run_anomaly_check and vehicle and telemetry_df is not None:
    with st.spinner('🔍 Analyzing telemetry data...'):
        # Convert to wide format if needed
        check_df = telemetry_df.copy()
        if 'telemetry_name' in check_df.columns:
            check_df = telemetry_loader.telemetry_to_wide_format(check_df)
        
        # Run checks
        anomaly_results = detect_all_anomalies(check_df, vehicle_id=vehicle)
        summary = get_anomaly_summary(anomaly_results)
        
        anomaly_box.markdown(f"**Anomaly Detection Results**")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric('Total Anomalies', summary['total_anomalies'])
        col_b.metric('🚨 Critical', summary['by_severity']['critical'])
        col_c.metric('⚠️ Warnings', summary['by_severity']['warning'])
        col_d.metric('ℹ️ Info', summary['by_severity']['info'])
        
        if summary['most_severe']:
            st.markdown('**Most Severe Issues:**')
            for anomaly in summary['most_severe'][:5]:  # Show top 5
                st.warning(str(anomaly))
