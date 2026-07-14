"""
NeoAI — Live Dynamic Enterprise Dashboard v6.1
FIXED: Bulletproof Error Handling Restored
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
import os
import time
from datetime import datetime

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeoAI — Enterprise EMS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded" 
)

DB_PATH = "neoai_live.db"

# ── Premium CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #05090f; }
  .block-container { padding: 2rem 2.5rem; }
  
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #04080f 100%);
    border-right: 1px solid #1a2744;
  }

  .main .block-container { animation: fadeIn 0.4s ease-out; }
  @keyframes fadeIn { 0% { opacity: 0; transform: translateY(5px); } 100% { opacity: 1; transform: translateY(0); } }

  .section-header { color: #8ba3c4; font-size: 14px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 10px; margin-bottom: 15px; border-bottom: 1px solid #162238; padding-bottom: 5px; }

  .kpi { background: linear-gradient(135deg, #0d1f3c, #07111d); border: 1px solid #1e3a5f; border-radius: 12px; padding: 16px 18px; text-align: center; position: relative; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
  .kpi::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--ac, #00e6b8); }
  .kpi-label { font-size: 11px; color: #6e8fb3; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; font-weight: 600;}
  .kpi-value { font-family: 'JetBrains Mono', monospace; font-size: 26px; font-weight: 600; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
  .kpi-unit  { font-size: 13px; color: #6e8fb3; margin-left: 4px; font-weight: 400;}
  .kpi-sub   { font-size: 11px; color: #4b6a8b; margin-top: 6px; }

  .green { color: #00e6b8; } .red { color: #ff3355; } .amber { color: #ffb833; } .blue { color: #3399ff; } .purple { color: #b388ff; } .white { color: #f0f4f8; }

  .alarm-c { background:#1a0510; border-left:4px solid #ff3355; padding:12px 16px; border-radius:6px; margin:6px 0; font-size:13px; color:#ff3355; font-family:'JetBrains Mono',monospace; animation: pulseRed 2s infinite; }
  .alarm-w { background:#1a1000; border-left:4px solid #ffb833; padding:12px 16px; border-radius:6px; margin:6px 0; font-size:13px; color:#ffb833; font-family:'JetBrains Mono',monospace; }
  .alarm-n { background:#001a12; border-left:4px solid #00e6b8; padding:12px 16px; border-radius:6px; margin:6px 0; font-size:13px; color:#00e6b8; font-family:'JetBrains Mono',monospace; }
  
  .live-dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#ff3355; margin-right:8px; box-shadow: 0 0 8px #ff3355; animation:blink 1.2s infinite; }
  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

  #MainMenu, footer { visibility:hidden; }
  header { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8ba3c4", family="Inter", size=11),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor="#162238", tickfont=dict(size=9, color="#4b6a8b"), showline=False, zeroline=False),
    yaxis=dict(gridcolor="#162238", tickfont=dict(size=9, color="#4b6a8b"), showline=False, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#8ba3c4"), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ── BULLETPROOF Data Acquisition Engine ──────────────────────────────────────
def load_table(table_name, location, limit=80):
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f'SELECT * FROM "{table_name}" WHERE location="{location}" ORDER BY rowid DESC LIMIT {limit}'
        df = pd.read_sql(query, conn)
        conn.close()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.iloc[::-1].reset_index(drop=True) 
    except Exception:
        return pd.DataFrame()

def safe_val(df, col, default=0.0):
    try:
        if col in df.columns and len(df) > 0:
            val = df[col].iloc[-1]
            return float(val) if pd.notna(val) else default
        return default
    except Exception:
        return default

def safe_str(df, col, default="—"):
    try:
        if col in df.columns and len(df) > 0:
            val = df[col].iloc[-1]
            return str(val) if pd.notna(val) else default
        return default
    except Exception:
        return default

# ── UI Presentation Components ───────────────────────────────────────────────
def kpi_card(label, value, unit, color_class, subtext, accent_hex="#00e6b8"):
    return f"""<div class="kpi" style="--ac:{accent_hex}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {color_class}">{value}<span class="kpi-unit">{unit}</span></div>
      <div class="kpi-sub">{subtext}</div>
    </div>"""

def hex_to_rgba(hex_color, alpha=0.1):
    try:
        hex_color = hex_color.lstrip('#')
        if len(hex_color) >= 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"
    except Exception:
        pass
    return "rgba(0, 230, 184, 0.1)"

def area_chart(df, cols, colors, title, height=250, smooth=True):
    fig = go.Figure()
    for col, color in zip(cols, colors):
        try:
            if col in df.columns and len(df) > 0:
                y_data = df[col].rolling(window=5, min_periods=1).mean() if smooth else df[col]
                clean_label = col.replace("_", " ").title()
                safe_fill = hex_to_rgba(color, 0.1) if "#" in color else color.replace('rgb', 'rgba').replace(')', ', 0.1)')
                
                fig.add_trace(go.Scatter(
                    x=df["timestamp"], y=y_data,
                    mode='lines', line=dict(color=color, width=2.5, shape='spline', smoothing=1.2),
                    name=clean_label, fill='tozeroy', fillcolor=safe_fill
                ))
        except Exception:
            continue # Skip charting this specific line if data is corrupted
            
    fig.update_layout(**PLOT_THEME, title=dict(text=title, font=dict(size=13, color="#e2e8f0")), height=height)
    return fig

# ── Sidebar Navigation & Location Switcher ───────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='margin-bottom:0;'>NeoAI EMS</h2>", unsafe_allow_html=True)
    st.markdown("<span style='color:#00e6b8; font-size:11px; font-weight:600; letter-spacing:1px;'>BESS MONITORING MATRIX</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<span style='color:#4b6a8b; font-size:10px; font-weight:700; text-transform:uppercase;'>📍 Select Plant Location</span>", unsafe_allow_html=True)
    selected_location = st.selectbox("", ["Prakasha_Nandurbar", "Bhandu_Rajasthan"], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<span style='color:#4b6a8b; font-size:10px; font-weight:700; text-transform:uppercase;'>Select Component View</span>", unsafe_allow_html=True)
    page = st.radio("", [
        "🏠 Master Overview",
        "🔋 Battery Storage (LFP)",
        "⚡ Power Conversion (PCS)",
        "🔌 Distribution Transformer",
        "🔧 Main Switchgear Panel",
        "📡 Transmission & Grid Line",
        "🚨 Alarms & Faults"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    refresh_rate = st.slider("Telemetry Refresh (sec)", 2, 10, 5)
    st.markdown("---")
    st.markdown(f'<div style="background:#1a0510; padding:12px; border-radius:8px; border: 1px solid #3d1424;"><span class="live-dot"></span><span style="color:#ff3355; font-size:12px; font-weight:600;">STREAM ACTIVE</span><br><span style="color:#6e8fb3; font-size:10px; font-family:monospace;">Tick: {datetime.now().strftime("%H:%M:%S")}</span></div>', unsafe_allow_html=True)

# ── Verification & DB Load ───────────────────────────────────────────────────
if not os.path.exists(DB_PATH):
    st.error("⚠️ Database not found. Please run your simulator.py script in a separate terminal first.")
    st.stop()

batt  = load_table("battery", selected_location)
pcs   = load_table("pcs", selected_location)
xfmr  = load_table("transformer", selected_location)
swgr  = load_table("switchgear", selected_location)
tline = load_table("tline", selected_location)

if len(batt) == 0:
    st.warning(f"⏳ Waiting for simulator to push data for {selected_location}...")
    st.stop()

# ── Dynamic Header ───────────────────────────────────────────────────────────
header_meta = {
    "🏠 Master Overview":            ("System Infrastructure Overview", "#00e6b8"),
    "🔋 Battery Storage (LFP)":       ("BESS Core Module Metrics",      "#00e6b8"),
    "⚡ Power Conversion (PCS)":      ("PCS Inverter Telemetry",        "#3399ff"),
    "🔌 Distribution Transformer":    ("Step-Up Transformer Status",     "#ffb833"),
    "🔧 Main Switchgear Panel":       ("Switchgear & Breaker Topology",  "#b388ff"),
    "📡 Transmission & Grid Line":     ("Substation Transmission Feed",   "#ff6699"),
    "🚨 Alarms & Faults":             ("Active Operational Faults",      "#ff3355"),
}
title, theme_color = header_meta[page]
last_sync = safe_str(batt, "timestamp", "—")
clean_loc = selected_location.replace('_', ', ')

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between; margin-bottom:24px;border-bottom:1px solid #162238;padding-bottom:14px">
  <div>
    <h2 style="margin:0; font-size:22px; font-weight:700; color:{theme_color};">{title}</h2>
    <span style="font-size:13px; color:#4b6a8b; font-weight:600;">📍 Site: {clean_loc} • Live Data Feed</span>
  </div>
  <div style="background:#0a1628; padding:8px 16px; border-radius:20px; border:1px solid #1a2744;">
    <span style="font-size:12px; color:#e2e8f0; font-family:'JetBrains Mono',monospace">LAST SYNC: {last_sync}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 1. MASTER OVERVIEW 
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Master Overview":
    st.markdown('<div class="section-header">🔋 BESS Storage Core</div>', unsafe_allow_html=True)
    soc = safe_val(batt, "state_of_charge_soc_pct")
    soh = safe_val(batt, "state_of_health_soh_pct")
    pwr = safe_val(batt, "battery_power_kw")
    temp = safe_val(batt, "average_cell_temperature_c")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("System SOC", f"{soc:.1f}", "%", "green" if soc > 20 else "red", "State of Charge", "#00e6b8"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("System SOH", f"{soh:.1f}", "%", "green" if soh > 80 else "amber", "Battery Health", "#00e6b8"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Active Power Out", f"{pwr:.1f}", "kW", "blue", "Discharging (-) / Charging (+)", "#3399ff"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Avg Cell Temp", f"{temp:.1f}", "°C", "amber" if temp > 35 else "green", "Thermal Management", "#ffb833"), unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:20px;">⚡ Grid & Conversion</div>', unsafe_allow_html=True)
    eff = safe_val(pcs, "conversion_efficiency_pct")
    freq = safe_val(pcs, "grid_frequency_hz", 50.0)
    xfmr_load = safe_val(xfmr, "transformer_loading_pct")
    brk = safe_str(swgr, "main_breaker_position", "CLOSED")

    c5, c6, c7, c8 = st.columns(4)
    with c5: st.markdown(kpi_card("PCS Efficiency", f"{eff:.2f}", "%", "purple", "Inverter Conversion", "#b388ff"), unsafe_allow_html=True)
    with c6: st.markdown(kpi_card("Grid Frequency", f"{freq:.3f}", "Hz", "green" if abs(freq-50) < 0.2 else "red", "Target: 50Hz", "#00e6b8"), unsafe_allow_html=True)
    with c7: st.markdown(kpi_card("XFMR Loading", f"{xfmr_load:.1f}", "%", "amber" if xfmr_load > 80 else "blue", "Transformer Stress", "#3399ff"), unsafe_allow_html=True)
    with c8: st.markdown(kpi_card("Main Breaker", str(brk).upper(), "", "green" if str(brk).upper()=="CLOSED" else "red", "Switchgear Status", "#00e6b8"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(area_chart(batt, ["state_of_charge_soc_pct", "state_of_health_soh_pct"], ["#00e6b8", "#3399ff"], "SOC vs SOH Degradation Curve", smooth=True), use_container_width=True)
    with col_b:
        st.plotly_chart(area_chart(pcs, ["active_power_kw", "apparent_power_kva"], ["#ff6699", "#b388ff"], "Substation Active & Apparent Power", smooth=True), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 2. BATTERY STORAGE (LFP)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔋 Battery Storage (LFP)":
    v_pack = safe_val(batt, "pack_voltage_v")
    ir = safe_val(batt, "internal_resistance_mohm")
    cycles = safe_val(batt, "cycle_count")
    c_rate = safe_val(batt, "c_rate")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Pack Voltage", f"{v_pack:.1f}", "V", "white", "DC Link Voltage", "#00e6b8"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Internal Resistance", f"{ir:.3f}", "mΩ", "amber" if ir > 2.5 else "green", "Cell Aging Marker", "#ffb833"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Total Cycles", f"{cycles:.0f}", "", "blue", "LFP Target: 6000", "#3399ff"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("C-Rate", f"{c_rate:.2f}", "C", "purple", "Charge/Discharge Speed", "#b388ff"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(area_chart(batt, ["max_cell_temperature_c", "average_cell_temperature_c", "min_cell_temperature_c"], ["#ff3355", "#ffb833", "#00e6b8"], "Cell Thermal Dispersion Mapping (°C)"), use_container_width=True)
    with col_b:
        st.plotly_chart(area_chart(batt, ["cell_voltage_spread_v"], ["#b388ff"], "Cell Voltage Delta / Balancer Drift (V)"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 3. POWER CONVERSION (PCS)
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Power Conversion (PCS)":
    pf = safe_val(pcs, "power_factor_overall")
    thd = safe_val(pcs, "voltage_thd_pct")
    igbt_t = safe_val(pcs, "igbt_temperature_c")
    leak = safe_val(pcs, "leakage_current_ma")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Power Factor", f"{pf:.3f}", "", "blue" if pf > 0.95 else "amber", "Discharge Phase Efficiency", "#3399ff"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Voltage THD", f"{thd:.2f}", "%", "green" if thd < 3 else "amber", "Harmonic Distortion Matrix", "#b388ff"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("IGBT Temp", f"{igbt_t:.1f}", "°C", "amber" if igbt_t > 60 else "green", "Inverter Silicon Core Health", "#ffb833"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Leakage Current", f"{leak:.1f}", "mA", "green" if leak < 30 else "red", "Earth Fault Indicator", "#ff3355"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(area_chart(pcs, ["ac_voltage_phase_r_v", "ac_voltage_phase_y_v", "ac_voltage_phase_b_v"], ["#ff3355", "#ffb833", "#3399ff"], "Three-Phase Output Voltage Waveforms"), use_container_width=True)
    with col_b:
        st.plotly_chart(area_chart(pcs, ["conversion_efficiency_pct"], ["#00e6b8"], "Inverter DC-to-AC Conversion Efficiency (%)"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 4. DISTRIBUTION TRANSFORMER
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔌 Distribution Transformer":
    oil_temp = safe_val(xfmr, "top_oil_temp_c")
    hotspot = safe_val(xfmr, "hotspot_temp_c")
    h2_ppm = safe_val(xfmr, "dga_h2_ppm")
    loss = safe_val(xfmr, "total_loss_kw")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Top Oil Temp", f"{oil_temp:.1f}", "°C", "green" if oil_temp < 65 else "amber", "Insulation Heat State", "#ffb833"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Hotspot Temp", f"{hotspot:.1f}", "°C", "amber" if hotspot > 80 else "green", "Winding Critical Heat", "#ff3355"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("DGA Hydrogen", f"{h2_ppm:.1f}", "ppm", "green" if h2_ppm < 100 else "red", "Dissolved Gas Analysis", "#00e6b8"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Total Losses", f"{loss:.2f}", "kW", "purple", "Copper + Core Losses", "#b388ff"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(area_chart(xfmr, ["top_oil_temp_c", "hotspot_temp_c", "winding_temp_c"], ["#ffb833", "#ff3355", "#3399ff"], "Internal Thermal Gradient Profile (°C)"), use_container_width=True)
    with col_b:
        st.plotly_chart(area_chart(xfmr, ["dga_h2_ppm", "dga_ch4_ppm", "dga_co_ppm"], ["#ff6699", "#00e6b8", "#b388ff"], "Combustible DGA Gas Tracking (ppm)"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN SWITCHGEAR PANEL
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Main Switchgear Panel":
    sf6_p = safe_val(swgr, "sf6_gas_pressure_bar")
    bus_temp = safe_val(swgr, "busbar_temperature_phase_r_c")
    thd_i = safe_val(swgr, "current_thd_pct")
    fault_ka = safe_val(swgr, "short_circuit_fault_current_ka")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("SF6 Gas Pressure", f"{sf6_p:.2f}", "bar", "green" if sf6_p > 5.5 else "red", "Arc Quenching Medium", "#ffb833"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Busbar Core Temp", f"{bus_temp:.1f}", "°C", "green", "Contact Resistance Heat", "#3399ff"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Current THD", f"{thd_i:.2f}", "%", "white", "Line Wave Distortion", "#b388ff"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Fault Current", f"{fault_ka:.2f}", "kA", "green" if fault_ka == 0 else "red", "Short Circuit Detection", "#ff3355"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(area_chart(swgr, ["voltage_phase_a_v", "voltage_phase_b_v", "voltage_phase_c_v"], ["#ff3355", "#ffb833", "#3399ff"], "High-Voltage Switchgear Busbar Line Voltages"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 6. TRANSMISSION & GRID LINE
# ════════════════════════════════════════════════════════════════════════════
elif page == "📡 Transmission & Grid Line":
    drop = safe_val(tline, "voltage_drop_across_line_v")
    losses = safe_val(tline, "line_transmission_loss_kw")
    cond_temp = safe_val(tline, "line_conductor_temperature_c")
    sag_m = safe_val(tline, "conductor_sag_m")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(kpi_card("Line Voltage Drop", f"{drop:.2f}", "V", "white", "Sending vs Receiving Variance", "#3399ff"), unsafe_allow_html=True)
    with c2: st.markdown(kpi_card("Ohmic Line Loss", f"{losses:.2f}", "kW", "amber", "Real-Time Transmission Loss", "#ffb833"), unsafe_allow_html=True)
    with c3: st.markdown(kpi_card("Conductor Temp", f"{cond_temp:.1f}", "°C", "green" if cond_temp < 60 else "red", "Zebra ACSR Line Stress", "#b388ff"), unsafe_allow_html=True)
    with c4: st.markdown(kpi_card("Line Sag", f"{sag_m:.2f}", "m", "green", "Physical Cable Displacement", "#00e6b8"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(area_chart(tline, ["sending_end_voltage_phase_a_v", "receiving_end_voltage_v"], ["#00e6b8", "#ff6699"], "Voltage Attenuation Profile (Send vs Receive)"), use_container_width=True)
    with col_b:
        st.plotly_chart(area_chart(tline, ["active_power_sending_end_kw", "active_power_receiving_end_kw"], ["#3399ff", "#ffb833"], "Transmission End-to-End Power Flow (kW)"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# 7. ALARMS & FAULTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alarms & Faults":
    st.markdown("### Substation Fault Matrix Logs")
    
    components = [
        ("Battery Stack", batt), 
        ("PCS Inverter", pcs), 
        ("Distribution Transformer", xfmr), 
        ("Switchgear Panel", swgr), 
        ("Grid Transmission Asset", tline)
    ]
    
    nominal_flag = True
    for name, df in components:
        fault_t = safe_str(df, "fault_type", "normal")
        sever = safe_str(df, "fault_severity", "none")
        
        if sever.lower() in ["critical", "warning", "fault"] and sever.lower() != "none":
            nominal_flag = False
            style_cls = "alarm-c" if sever.lower() == "critical" else "alarm-w"
            indicator = "🔴 CRITICAL FAULT" if sever.lower() == "critical" else "🟡 ALERT"
            st.markdown(f'<div class="{style_cls}">{indicator} — Asset: **{name}** | Issue: `{str(fault_t).upper().replace("_"," ")}`</div>', unsafe_allow_html=True)
            
    if nominal_flag:
        st.markdown('<div class="alarm-n">🟢 OPERATIONAL HEALTH NOMINAL — Zero anomalous telemetry patterns or fault signatures identified for this site.</div>', unsafe_allow_html=True)

# ── Loop Thread Hold Execution ───────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()



