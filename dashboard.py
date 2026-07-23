

import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from agent.smart_agent import generate_rag_advisory
from streamlit_autorefresh import st_autorefresh

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeoAI EMS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
count = st_autorefresh(interval=10000, key="neoai_refresh")
st.session_state.tick = count

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #080d14; }
  .block-container { padding: 1rem 1.5rem; }
  [data-testid="stSidebar"] { background: #0a1628; }
  .kpi-row { display: flex; gap: 12px; margin-bottom: 12px; }
  .kpi-box {
    background: #0d1f3c; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 18px 20px; flex: 1; text-align: center;
  }
  .kpi-label { font-size: 11px; color: #5a7a9a; text-transform: uppercase;
               letter-spacing: 1.5px; margin-bottom: 4px; }
  .kpi-val   { font-size: 30px; font-weight: 700; color: #00d4aa; }
  .kpi-unit  { font-size: 11px; color: #5a7a9a; }
  .alarm-ok  { background:#001a12; border-left:3px solid #00d4aa;
               padding:8px 14px; border-radius:6px; margin:4px 0;
               color:#00d4aa; font-size:13px; }
  .alarm-warn{ background:#1a1000; border-left:3px solid #ffb347;
               padding:8px 14px; border-radius:6px; margin:4px 0;
               color:#ffb347; font-size:13px; }
  .alarm-crit{ background:#1a0510; border-left:3px solid #ff4d6d;
               padding:8px 14px; border-radius:6px; margin:4px 0;
               color:#ff4d6d; font-size:13px; }
  #MainMenu, footer {visibility: hidden;} 
  header {background-color: transparent !important;}
</style>
""", unsafe_allow_html=True)

PLOT_CFG = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#080d14",
    font=dict(color="#94aed4", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="#1a2744", showline=False),
    yaxis=dict(gridcolor="#1a2744", showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ── Safe value helper ─────────────────────────────────────────────────────────
def sv(row, col, default=0):
    try:
        v = row[col]
        return default if pd.isna(v) else v
    except Exception:
        return default

# ── Load all datasets (cached) ────────────────────────────────────────────────
# ── Helper to track file changes ──────────────────────────────────────────────
def get_file_mtimes():
    files = [
        "battery_neoai_live.parquet", "pcs_neoai_live.parquet", 
        "transformer_neoai_live.parquet", "switchgear_neoai_live.parquet", 
        "transmission_line_neoai_live.parquet"
    ]
    return {f: os.path.getmtime(f) if os.path.exists(f) else 0 for f in files}

# ── Enhanced Loader with Timestamp Watcher ────────────────────────────────────
@st.cache_data(show_spinner="Loading NeoAI datasets…")
def load_data(file_timestamps):
    dfs = {}
    # Keep all your original try/except blocks here
    try:
        dfs["battery_hist"] = pd.read_parquet("battery_neoai_historical.parquet")
        dfs["battery_live"] = pd.read_parquet("battery_neoai_live.parquet")
    except Exception as e:
        st.error(f"Battery dataset error: {e}")
        dfs["battery_hist"] = pd.DataFrame()
        dfs["battery_live"] = pd.DataFrame()

    try:
        dfs["pcs_hist"] = pd.read_parquet("pcs_neoai_historical.parquet")
        dfs["pcs_live"] = pd.read_parquet("pcs_neoai_live.parquet")
    except Exception as e:
        st.error(f"PCS dataset error: {e}")
        dfs["pcs_hist"] = pd.DataFrame()
        dfs["pcs_live"] = pd.DataFrame()

    try:
        dfs["transformer_hist"] = pd.read_parquet("transformer_neoai_historical.parquet")
        dfs["transformer_live"] = pd.read_parquet("transformer_neoai_live.parquet")
    except Exception as e:
        st.error(f"Transformer dataset error: {e}")
        dfs["transformer_hist"] = pd.DataFrame()
        dfs["transformer_live"] = pd.DataFrame()

    try:
        dfs["switchgear_hist"] = pd.read_parquet("switchgear_neoai_historical.parquet")
        dfs["switchgear_live"] = pd.read_parquet("switchgear_neoai_live.parquet")
    except Exception as e:
        st.error(f"Switchgear dataset error: {e}")
        dfs["switchgear_hist"] = pd.DataFrame()
        dfs["switchgear_live"] = pd.DataFrame()

    try:
        dfs["tline_hist"] = pd.read_parquet("transmission_line_neoai_historical.parquet")
        dfs["tline_live"] = pd.read_parquet("transmission_line_neoai_live.parquet")
    except Exception as e:
        st.error(f"Transmission line dataset error: {e}")
        dfs["tline_hist"] = pd.DataFrame()
        dfs["tline_live"] = pd.DataFrame()

    return dfs

# Call the loader with the timestamps
dfs = load_data(file_timestamps=get_file_mtimes())

# Default to live datasets
battery_df    = dfs["battery_live"]
pcs_df        = dfs["pcs_live"]
xfmr_df       = dfs["transformer_live"]
swgr_df       = dfs["switchgear_live"]
tline_df      = dfs["tline_live"]
# ── Session state (simulator tick) ────────────────────────────────────────────
# ── Session state (simulator tick) ────────────────────────────────────────────
if "tick" not in st.session_state:
    st.session_state.tick = 0

# ── AI Cache ─────────────────────────────────────────────────────────────────
# --- AI Cache --------------------------------------------

if "last_fault" not in st.session_state:
    st.session_state.last_fault = None

if "last_advice" not in st.session_state:
    st.session_state.last_advice = None

if "last_agent_call_time" not in st.session_state:
    st.session_state.last_agent_call_time = 0

if "acknowledged_faults" not in st.session_state:
    st.session_state.acknowledged_faults = set()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#00d4aa'>⚡ NeoAI EMS</h2>",
                unsafe_allow_html=True)
    st.markdown("*Neosol Energy Systems Pvt Ltd*")
    
    # --- LIVE DATE AND TIME (Forced to IST) ---
    ist = timezone(timedelta(hours=5, minutes=30))
    current_time = datetime.now(ist).strftime("%Y-%m-%d | %I:%M:%S %p")
    st.markdown(f"🕒 **System Time:** `{current_time}`")
    st.markdown("---")

    location = st.selectbox(
        "Plant Location",
        ["Prakasha_Nandurbar", "Bhandu_Rajasthan"]
    )

    page = st.radio("Component View", [
        "🏠 Master Overview",
        "🔋 Battery Storage (LFP)",
        "⚡ Power Conversion (PCS)",
        "🔌 Distribution Transformer",
        "🔧 Main Switchgear Panel",
        "📡 Transmission & Grid Line",
        "🚨 Alarms & Faults",
        "🔮 Forecast & AI Advisory",
    ])
    
   

    st.markdown("---")
    st.markdown(f"**Tick:** `{st.session_state.tick}`")
    st.markdown("🟢 **Live** — updates every 3s")

# ── Helper: get current simulated row ────────────────────────────────────────
def get_row(df, loc):
    if df is None or df.empty:
        return None
    if "location" in df.columns:
        loc_df = df[df["location"] == loc].reset_index(drop=True)
    else:
        loc_df = df.reset_index(drop=True)
    if len(loc_df) == 0:
        return None
    return loc_df.iloc[st.session_state.tick % len(loc_df)]

# ── Helper: get history window ────────────────────────────────────────────────
def get_window(df, loc, n=96):
    if df is None or df.empty:
        return pd.DataFrame()
    if "location" in df.columns:
        loc_df = df[df["location"] == loc].reset_index(drop=True)
    else:
        loc_df = df.reset_index(drop=True)
    if len(loc_df) == 0:
        return pd.DataFrame()
    idx   = st.session_state.tick % len(loc_df)
    start = max(0, idx - n)
    return loc_df.iloc[start:idx+1]

# ── Helper: time-series chart ─────────────────────────────────────────────────
def ts_chart(df, loc, y_col, title, color="#00d4aa", n=96):
    win = get_window(df, loc, n)

    if win.empty or y_col not in win.columns:
        st.info(f"No data for {y_col}")
        return

    fig = go.Figure(go.Scatter(
        x=win["timestamp"],
        y=win[y_col],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},"
                  f"{int(color[3:5],16)},"
                  f"{int(color[5:7],16)},0.06)"
    ))

    fig.update_layout(**PLOT_CFG, title=title, height=220)

    st.plotly_chart(fig, width="stretch")

# ── KPI helper ────────────────────────────────────────────────────────────────
def kpi_row(items):
    cols = st.columns(len(items))

    for col, (label, val, unit, color) in zip(cols, items):
        col.markdown(f"""
        <div class="kpi-box" style="border-top:3px solid {color}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-val" style="color:{color}">
                {val}<span class="kpi-unit">{unit}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MASTER OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Master Overview":

    st.markdown("""
# 🌍 NeoAI Energy Management System
### Live Plant Monitoring & Intelligent Asset Analytics
""")

    st.success(
        "🟢 Plant Status: HEALTHY   |   🔋 Battery Online   |   ⚡ PCS Online   |   🔌 Grid Connected"
    )

    batt_r = get_row(battery_df, location)
    pcs_r = get_row(pcs_df, location)
    swgr_r = get_row(swgr_df, location)
    xfmr_r = get_row(xfmr_df, location)

    soc = f"{sv(batt_r,'state_of_charge_soc_pct',0):.1f}" if batt_r is not None else "—"
    soh = f"{sv(batt_r,'state_of_health_soh_pct',0):.1f}" if batt_r is not None else "—"
    pwr = f"{sv(pcs_r,'active_power_kw',0):.1f}" if pcs_r is not None else "—"
    freq = f"{sv(pcs_r,'grid_frequency_hz',50):.3f}" if pcs_r is not None else "—"
    pf = f"{sv(pcs_r,'power_factor_overall',1):.4f}" if pcs_r is not None else "—"
    brk = str(sv(swgr_r,'main_breaker_position','—')) if swgr_r is not None else "—"

    kpi_row([
        ("State of Charge", soc, "%", "#00d4aa"),
        ("State of Health", soh, "%", "#4a9eff"),
        ("PCS Active Power", pwr, "kW", "#ffb347"),
        ("Grid Frequency", freq, "Hz", "#a78bfa"),
        ("Power Factor", pf, "", "#00d4aa"),
        ("Breaker Position", brk, "", "#ff6b9d"),
    ])

    # ---------- Live Banner ----------
    st.markdown(f"""
    <div style="
        background:#0d1f3c;
        border-left:5px solid #00d4aa;
        padding:12px 18px;
        border-radius:8px;
        color:white;
        margin-top:8px;
        margin-bottom:20px;
        font-size:15px;
    ">
        🟢 <b>LIVE</b>
        &nbsp;&nbsp;&nbsp;
        📍 <b>{location}</b>
        &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
        🔄 Refresh: <b>5 sec</b>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        ts_chart(
            battery_df,
            location,
            "state_of_charge_soc_pct",
            "🔋 Battery State of Charge Trend",
            "#00d4aa"
        )

    with c2:
        ts_chart(
            pcs_df,
            location,
            "active_power_kw",
            "⚡ PCS Active Power Trend",
            "#4a9eff"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("🩺 Equipment Health Overview")

    cards = st.columns(5)

    equipment = [
        ("🔋 Battery", battery_df),
        ("⚡ PCS", pcs_df),
        ("🔌 Transformer", xfmr_df),
        ("🔧 Switchgear", swgr_df),
        ("📡 Transmission", tline_df),
    ]

    for col, (name, df_ref) in zip(cards, equipment):

        r = get_row(df_ref, location)

        if r is None:
            status = "NO DATA"
            color = "#ffb347"
        else:
            fault = int(sv(r, "is_fault", 0))

            if fault:
                status = "⚠ FAULT"
                color = "#ff4d6d"
            else:
                status = "✅ NORMAL"
                color = "#00d4aa"

        col.markdown(f"""
        <div class="kpi-box" style="border-top:3px solid {color}">
            <div class="kpi-label">{name}</div>
            <div style="
                color:{color};
                font-size:18px;
                font-weight:700;
                margin-top:8px;
            ">
                {status}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader("📢 Plant Events")

    events = []

    if batt_r is not None:
        if float(sv(batt_r, "state_of_charge_soc_pct", 0)) >= 80:
            events.append("🟢 Battery SOC is healthy and above 80%.")
        else:
            events.append("🟡 Battery SOC is below 80%.")

    if pcs_r is not None:
        events.append(f"⚡ PCS delivering {pwr} kW.")

    if swgr_r is not None:
        events.append(f"🔧 Main breaker status: {brk}.")

    if len(events) == 0:
        st.info("No live events available.")
    else:
        for e in events:
            st.markdown(
                f'<div class="alarm-ok">{e}</div>',
                unsafe_allow_html=True
            )
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATTERY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔋 Battery Storage (LFP)":
    st.markdown("# 🔋 Battery Energy Storage System (LFP)")
    st.caption("Live Battery Management System • Cell Health • Thermal Monitoring • Lifetime Analytics")

    row = get_row(battery_df, location)

    if row is None:
        st.warning("Battery data not available.")

    else:
        soc  = sv(row, "state_of_charge_soc_pct", 0)
        soh  = sv(row, "state_of_health_soh_pct", 0)
        pwr  = sv(row, "battery_power_kw", 0)
        temp = sv(row, "average_cell_temperature_c", 0)
        ir   = sv(row, "internal_resistance_mohm", 0)
        rul  = sv(row, "remaining_useful_life_years", 0)
        cyc  = sv(row, "cycle_count", 0)
        volt = sv(row, "pack_voltage_v", 0)

        max_temp = sv(row, "max_cell_temperature_c", temp)
        min_temp = sv(row, "min_cell_temperature_c", temp)

        # ---------------- Battery Status ----------------

        if pwr > 5:
            battery_status = "🟢 Charging"
            status_color = "#00d4aa"

        elif pwr < -5:
            battery_status = "🔵 Discharging"
            status_color = "#4a9eff"

        else:
            battery_status = "🟡 Idle"
            status_color = "#ffb347"

        st.markdown(
            f"""
            <div style="
            background:#0d1f3c;
            border-left:6px solid {status_color};
            padding:14px;
            border-radius:10px;
            margin-bottom:18px;
            color:white;
            ">
            <b>{battery_status}</b>
            &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            🔋 Battery Pack Online
            &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;
            📍 {location}
            </div>
            """,
            unsafe_allow_html=True,
        )

        kpi_row([
            ("SOC", f"{soc:.1f}", "%", "#00d4aa"),
            ("SOH", f"{soh:.1f}", "%", "#4a9eff"),
            ("Pack Voltage", f"{volt:.1f}", "V", "#ffb347"),
            ("Battery Power", f"{pwr:.1f}", "kW", "#00d4aa"),
            ("Avg Temp", f"{temp:.1f}", "°C", "#ff6b9d"),
            ("Cycle Count", f"{cyc:.0f}", "", "#a78bfa"),
        ])

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=soc,
                number={
                    "suffix": "%",
                    "font": {"size": 32, "color": "#00d4aa"}
                },
                title={
                    "text": "Battery State of Charge",
                    "font": {"size": 14}
                },
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00d4aa"},
                    "bgcolor": "#0d1f3c",
                    "bordercolor": "#1e3a5f",
                    "steps": [
                        {"range": [0, 20], "color": "#2b0d0d"},
                        {"range": [20, 50], "color": "#3a2500"},
                        {"range": [50, 80], "color": "#13341f"},
                        {"range": [80, 100], "color": "#005c43"}
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "value": 20
                    }
                }
            ))

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=250,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, width="stretch")

        with c2:
            ts_chart(
                battery_df,
                location,
                "state_of_charge_soc_pct",
                "📈 State of Charge Trend",
                "#00d4aa"
            )

        with c3:
            win = get_window(battery_df, location)

            if not win.empty:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["max_cell_temperature_c"],
                    name="Maximum",
                    line=dict(color="#ff4d6d", width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["average_cell_temperature_c"],
                    name="Average",
                    line=dict(color="#ffb347", width=3)
                ))

                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["min_cell_temperature_c"],
                    name="Minimum",
                    line=dict(color="#00d4aa", width=2)
                ))

                fig.update_layout(
                    **PLOT_CFG,
                    title="🌡 Cell Temperature Profile",
                    height=250
                )

                st.plotly_chart(fig, width="stretch")

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("📊 Battery Performance Trends")
        
        c4, c5 = st.columns(2)

        with c4:
            ts_chart(
                battery_df,
                location,
                "state_of_health_soh_pct",
                "🔋 Battery Health Degradation",
                "#4a9eff"
            )

        with c5:
            ts_chart(
                battery_df,
                location,
                "internal_resistance_mohm",
                "⚙ Internal Resistance (Battery Aging)",
                "#a78bfa"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("🩺 Battery Health Assessment")

        col1, col2 = st.columns(2)

        # ---------------- LEFT : HEALTH ----------------

        with col1:
            thermal = "🟢 NORMAL"
            thermal_color = "#00d4aa"

            if temp >= 45:
                thermal = "🔴 HIGH TEMPERATURE"
                thermal_color = "#ff4d6d"
            elif temp >= 38:
                thermal = "🟡 ELEVATED"
                thermal_color = "#ffb347"

            if soh >= 90:
                aging = "🟢 EXCELLENT"
                aging_color = "#00d4aa"
            elif soh >= 80:
                aging = "🟡 GOOD"
                aging_color = "#ffb347"
            else:
                aging = "🔴 DEGRADED"
                aging_color = "#ff4d6d"

            if ir <= 2:
                resistance = "🟢 NORMAL"
                resistance_color = "#00d4aa"
            elif ir <= 4:
                resistance = "🟡 INCREASING"
                resistance_color = "#ffb347"
            else:
                resistance = "🔴 HIGH"
                resistance_color = "#ff4d6d"

            st.markdown(f"""
            <div class="kpi-box">

            ### Battery Diagnostics

            <b>Thermal Status</b><br>
            <span style="color:{thermal_color};font-size:18px">{thermal}</span>

            <hr>

            <b>Battery Aging</b><br>
            <span style="color:{aging_color};font-size:18px">{aging}</span>

            <hr>

            <b>Internal Resistance</b><br>
            <span style="color:{resistance_color};font-size:18px">{resistance}</span>

            </div>
            """, unsafe_allow_html=True)

        # ---------------- RIGHT : EVENTS ----------------

        with col2:
            st.markdown("""
            <div class="kpi-box">

            ### 📢 Live Battery Events
            """, unsafe_allow_html=True)

            if soc > 90:
                st.markdown(
                    '<div class="alarm-ok">🟢 Battery SOC above 90%</div>',
                    unsafe_allow_html=True
                )
            elif soc < 20:
                st.markdown(
                    '<div class="alarm-crit">🔴 Battery SOC critically low</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="alarm-ok">🟢 Battery SOC within operating range</div>',
                    unsafe_allow_html=True
                )

            if temp > 40:
                st.markdown(
                    '<div class="alarm-warn">🟡 Cell temperature elevated</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="alarm-ok">🟢 Cell temperature normal</div>',
                    unsafe_allow_html=True
                )

            if soh > 90:
                st.markdown(
                    '<div class="alarm-ok">🟢 Battery health excellent</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="alarm-warn">🟡 Battery aging observed</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                f'<div class="alarm-ok">🔄 Cycle Count : {int(cyc)}</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="alarm-ok">⏳ Estimated Remaining Life : {rul:.1f} years</div>',
                unsafe_allow_html=True
            )

            st.markdown("</div>", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PCS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Power Conversion (PCS)":

    st.markdown("# ⚡ Power Conversion System (PCS)")
    st.caption(
        "Live Inverter Monitoring • Grid Synchronization • Power Electronics"
    )

    row = get_row(pcs_df, location)

    if row is None:
        st.warning("PCS data not available.")

    else:

        eff = sv(row, "conversion_efficiency_pct", 0)
        pwr = sv(row, "active_power_kw", 0)
        rpwr = sv(row, "reactive_power_kvar", 0)
        freq = sv(row, "grid_frequency_hz", 50)
        pf = sv(row, "power_factor_overall", 1)
        igbt = sv(row, "igbt_temperature_c", 0)
        thd = sv(row, "voltage_thd_pct", 0)
        mode = str(sv(row, "operating_mode", "Standby"))

        dc_bus = sv(row, "dc_bus_voltage_v", 0)
        dc_current = sv(row, "dc_current_battery_side_a", 0)
        kva = sv(row, "apparent_power_kva", 0)
        op_hours = sv(row, "total_operating_hours", 0)

        # ---------------- STATUS ----------------

        if "grid" in mode.lower():
            status = "🟢 GRID CONNECTED"
            status_color = "#00d4aa"

        elif "standby" in mode.lower():
            status = "🟡 STANDBY"
            status_color = "#ffb347"

        else:
            status = "🔵 ACTIVE"
            status_color = "#4a9eff"

        st.markdown(
            f"""
<div style="
background:#0d1f3c;
border-left:6px solid {status_color};
padding:14px;
border-radius:10px;
margin-bottom:18px;
color:white;
">

<b>{status}</b>

&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;

⚡ Operating Mode :
<b>{mode.upper()}</b>

&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;

📍 {location}

</div>
""",
            unsafe_allow_html=True,
        )

        kpi_row([
            ("Active Power", f"{pwr:.1f}", "kW", "#4a9eff"),
            ("Reactive Power", f"{rpwr:.1f}", "kVAR", "#ffb347"),
            ("Efficiency", f"{eff:.2f}", "%", "#00d4aa"),
            ("Power Factor", f"{pf:.3f}", "", "#a78bfa"),
            ("Grid Frequency", f"{freq:.3f}", "Hz", "#00d4aa"),
            ("IGBT Temp", f"{igbt:.1f}", "°C", "#ff6b9d"),
        ])

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            st.metric("DC Bus Voltage", f"{dc_bus:.1f} V")

        with m2:
            st.metric("DC Current", f"{dc_current:.1f} A")

        with m3:
            st.metric("Apparent Power", f"{kva:.1f} kVA")

        with m4:
            st.metric("Voltage THD", f"{thd:.2f} %")

        with m5:
            st.metric("Operating Hours", f"{op_hours:.0f}")

        with m6:
            st.metric("Operating Mode", mode.upper())

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            ts_chart(pcs_df, location,
                     "active_power_kw", "Active Power (kW)", "#4a9eff")
        with c2:
            ts_chart(pcs_df, location,
                     "conversion_efficiency_pct",
                     "Conversion Efficiency (%)", "#00d4aa")
        with c3:
            ts_chart(pcs_df, location,
                     "igbt_temperature_c", "IGBT Temperature (°C)", "#ff4d6d")

        c4, c5 = st.columns(2)
        with c4:
            win = get_window(pcs_df, location)
            if not win.empty:
                fig = go.Figure()
                for col_n, color, lbl in [
                    ("ac_voltage_phase_r_v", "#ff4d6d", "Phase R"),
                    ("ac_voltage_phase_y_v", "#ffb347", "Phase Y"),
                    ("ac_voltage_phase_b_v", "#4a9eff", "Phase B"),
                ]:
                    fig.add_trace(go.Scatter(x=win["timestamp"], y=win[col_n],
                        line=dict(color=color, width=1.5), name=lbl))
                fig.update_layout(**PLOT_CFG, title="3-Phase Voltage (V)", height=220)
                st.plotly_chart(fig, use_container_width=True)

        with c5:
            ts_chart(pcs_df, location,
                     "grid_frequency_hz", "Grid Frequency (Hz)", "#a78bfa")
            st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("🩺 PCS Diagnostics")

        d1, d2 = st.columns(2)

        # ---------------- LEFT : DIAGNOSTICS ----------------

        with d1:

            heatsink = sv(row, "heatsink_temperature_c", 0)
            pq_score = sv(row, "power_quality_compliance_score", 0)

            comm_ok = (
                str(sv(row, "modbus_communication_status", "offline")).lower()
                == "online"
            )

            ground_fault = bool(sv(row, "ground_fault_status", False))
            arc_flash = bool(sv(row, "arc_flash_detection_status", False))

            # IGBT
            if igbt < 60:
                igbt_status = "🟢 NORMAL"
                igbt_color = "#00d4aa"
            elif igbt < 75:
                igbt_status = "🟡 ELEVATED"
                igbt_color = "#ffb347"
            else:
                igbt_status = "🔴 HIGH"
                igbt_color = "#ff4d6d"

            # Power Quality
            if pq_score >= 95:
                pq_status = "🟢 EXCELLENT"
                pq_color = "#00d4aa"
            elif pq_score >= 85:
                pq_status = "🟡 GOOD"
                pq_color = "#ffb347"
            else:
                pq_status = "🔴 POOR"
                pq_color = "#ff4d6d"

            # Communication
            if comm_ok:
                comm_status = "🟢 ONLINE"
                comm_color = "#00d4aa"
            else:
                comm_status = "🔴 OFFLINE"
                comm_color = "#ff4d6d"

            st.markdown(f"""
<div class="kpi-box">

### PCS Diagnostics

<b>IGBT Thermal Status</b><br>
<span style="color:{igbt_color};font-size:18px">{igbt_status}</span>

<hr>

<b>Power Quality</b><br>
<span style="color:{pq_color};font-size:18px">{pq_status}</span>

<hr>

<b>Communication</b><br>
<span style="color:{comm_color};font-size:18px">{comm_status}</span>

</div>
""", unsafe_allow_html=True)

        # ---------------- RIGHT : EVENTS ----------------

        with d2:

            st.markdown("""
<div class="kpi-box">

### 📢 Live PCS Events
""", unsafe_allow_html=True)

            if eff >= 98:
                st.markdown(
                    '<div class="alarm-ok">🟢 Conversion efficiency above 98%</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alarm-warn">🟡 Conversion efficiency below optimal</div>',
                    unsafe_allow_html=True,
                )

            if abs(thd) <= 5:
                st.markdown(
                    '<div class="alarm-ok">🟢 Voltage THD within IEEE limits</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alarm-warn">🟡 Voltage THD elevated</div>',
                    unsafe_allow_html=True,
                )

            if not ground_fault:
                st.markdown(
                    '<div class="alarm-ok">🟢 No ground fault detected</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alarm-crit">🔴 Ground fault detected</div>',
                    unsafe_allow_html=True,
                )

            if not arc_flash:
                st.markdown(
                    '<div class="alarm-ok">🟢 Arc flash monitoring normal</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="alarm-crit">🔴 Arc flash event detected</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="alarm-ok">⚙ Heatsink Temperature : {heatsink:.1f} °C</div>',
                unsafe_allow_html=True,
            )

            st.markdown("</div>", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔌 Distribution Transformer":

    st.markdown("# 🔌 Distribution Transformer")
    st.caption(
        "Oil-Cooled Power Transformer • Live Condition Monitoring • Protection & Thermal Analytics"
    )

    if xfmr_df.empty:
        st.warning("Transformer dataset not loaded.")

    else:

        row = get_row(xfmr_df, location)

        if row is None:
            st.warning("No transformer data for this location.")

        else:

            loading = sv(row, "transformer_loading_pct", 0)
            top_oil = sv(row, "top_oil_temp_c", 0)
            winding = sv(row, "winding_temp_c", 0)
            hotspot = sv(row, "hotspot_temp_c", 0)
            eff = sv(row, "transformer_efficiency_pct", 0)
            losses = sv(row, "total_loss_kw", 0)

            apparent = sv(row, "apparent_power_kva", 0)
            pf = sv(row, "power_factor", 0)
            hv = sv(row, "primary_voltage_hv_v", 0)
            lv = sv(row, "secondary_voltage_lv_v", 0)
            active = sv(row, "active_power_secondary_kw", 0)
            hours = sv(row, "total_operating_hours", 0)

            fault = bool(sv(row, "is_fault", False))

            # ---------------- STATUS ----------------

            if fault:

                status = "🔴 FAULT DETECTED"
                status_color = "#ff4d6d"

            elif loading >= 90:

                status = "🟡 HIGH LOADING"
                status_color = "#ffb347"

            else:

                status = "🟢 HEALTHY"
                status_color = "#00d4aa"

            st.markdown(
                f"""
<div style="
background:#0d1f3c;
border-left:6px solid {status_color};
padding:14px;
border-radius:10px;
margin-bottom:18px;
color:white;
">

<b>{status}</b>

&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;

🔌 Transformer Online

&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;

📍 {location}

</div>
""",
                unsafe_allow_html=True,
            )

            kpi_row([
                ("Loading", f"{loading:.1f}", "%", "#ffb347"),
                ("Top Oil", f"{top_oil:.1f}", "°C", "#ff4d6d"),
                ("Winding", f"{winding:.1f}", "°C", "#ff6b9d"),
                ("Hotspot", f"{hotspot:.1f}", "°C", "#a78bfa"),
                ("Efficiency", f"{eff:.2f}", "%", "#00d4aa"),
                ("Total Loss", f"{losses:.2f}", "kW", "#4a9eff"),
            ])

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3, m4, m5, m6 = st.columns(6)

            with m1:
                st.metric("Active Power", f"{active:.1f} kW")

            with m2:
                st.metric("Apparent Power", f"{apparent:.1f} kVA")

            with m3:
                st.metric("Power Factor", f"{pf:.3f}")

            with m4:
                st.metric("HV Voltage", f"{hv:.0f} V")

            with m5:
                st.metric("LV Voltage", f"{lv:.0f} V")

            with m6:
                st.metric("Operating Hours", f"{hours:.0f}")

            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            with c1:
                ts_chart(
                    xfmr_df,
                    location,
                    "transformer_loading_pct",
                    "📈 Transformer Loading Trend",
                    "#ffb347"
                )

            with c2:
                ts_chart(
                    xfmr_df,
                    location,
                    "top_oil_temp_c",
                    "🌡 Top Oil Temperature",
                    "#ff4d6d"
                )

            with c3:
                ts_chart(
                    xfmr_df,
                    location,
                    "winding_temp_c",
                    "🔥 Winding Temperature",
                    "#ff6b9d"
                )

            st.markdown("<br>", unsafe_allow_html=True)

            c4, c5 = st.columns(2)

            with c4:

                win = get_window(xfmr_df, location)

                if not win.empty:

                    fig = go.Figure()

                    fig.add_trace(go.Scatter(
                        x=win["timestamp"],
                        y=win["phase_a_voltage_v"],
                        name="Phase A",
                        line=dict(color="#ff4d6d", width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=win["timestamp"],
                        y=win["phase_b_voltage_v"],
                        name="Phase B",
                        line=dict(color="#ffb347", width=2)
                    ))

                    fig.add_trace(go.Scatter(
                        x=win["timestamp"],
                        y=win["phase_c_voltage_v"],
                        name="Phase C",
                        line=dict(color="#4a9eff", width=2)
                    ))

                    fig.update_layout(
                        **PLOT_CFG,
                        title="⚡ Three-Phase Voltage",
                        height=260
                    )

                    st.plotly_chart(fig, width="stretch")

            with c5:

                ts_chart(
                    xfmr_df,
                    location,
                    "transformer_efficiency_pct",
                    "⚙ Transformer Efficiency",
                    "#00d4aa"
                )
                st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("🩺 Transformer Diagnostics")

            d1, d2 = st.columns(2)

            # ---------------- LEFT : DIAGNOSTICS ----------------

            with d1:

                moisture = sv(row, "oil_moisture_ppm", 0)
                dielectric = sv(row, "oil_dielectric_strength_kv", 0)
                insulation = sv(row, "insulation_resistance_mohm", 0)

                buchholz = bool(sv(row, "buchholz_relay_status", False))
                diff_ok = bool(sv(row, "differential_protection_ok", True))

                # Thermal
                if hotspot < 90:
                    thermal = "🟢 NORMAL"
                    thermal_color = "#00d4aa"
                elif hotspot < 110:
                    thermal = "🟡 ELEVATED"
                    thermal_color = "#ffb347"
                else:
                    thermal = "🔴 HIGH"
                    thermal_color = "#ff4d6d"

                # Oil Condition
                if moisture < 20 and dielectric > 50:
                    oil = "🟢 GOOD"
                    oil_color = "#00d4aa"
                elif moisture < 35:
                    oil = "🟡 ACCEPTABLE"
                    oil_color = "#ffb347"
                else:
                    oil = "🔴 ATTENTION"
                    oil_color = "#ff4d6d"

                # Protection
                if diff_ok and not buchholz:
                    protection = "🟢 HEALTHY"
                    protection_color = "#00d4aa"
                else:
                    protection = "🔴 CHECK"
                    protection_color = "#ff4d6d"

                # Insulation
                if insulation >= 100:
                    ins = "🟢 EXCELLENT"
                    ins_color = "#00d4aa"
                elif insulation >= 50:
                    ins = "🟡 GOOD"
                    ins_color = "#ffb347"
                else:
                    ins = "🔴 LOW"
                    ins_color = "#ff4d6d"

                st.markdown(f"""
<div class="kpi-box">

### Transformer Diagnostics

<b>Thermal Condition</b><br>
<span style="color:{thermal_color};font-size:18px">{thermal}</span>

<hr>

<b>Oil Condition</b><br>
<span style="color:{oil_color};font-size:18px">{oil}</span>

<hr>

<b>Protection Status</b><br>
<span style="color:{protection_color};font-size:18px">{protection}</span>

<hr>

<b>Insulation Health</b><br>
<span style="color:{ins_color};font-size:18px">{ins}</span>

</div>
""", unsafe_allow_html=True)

            # ---------------- RIGHT : EVENTS ----------------

            with d2:

                st.markdown("""
<div class="kpi-box">

### 📢 Live Transformer Events
""", unsafe_allow_html=True)

                if loading < 80:
                    st.markdown(
                        '<div class="alarm-ok">🟢 Transformer loading within limits</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="alarm-warn">🟡 High transformer loading observed</div>',
                        unsafe_allow_html=True
                    )

                if top_oil < 70:
                    st.markdown(
                        '<div class="alarm-ok">🟢 Top oil temperature normal</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="alarm-warn">🟡 Oil temperature elevated</div>',
                        unsafe_allow_html=True
                    )

                if diff_ok:
                    st.markdown(
                        '<div class="alarm-ok">🟢 Differential protection healthy</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="alarm-crit">🔴 Differential protection alarm</div>',
                        unsafe_allow_html=True
                    )

                earth_fault = bool(sv(row, "earth_fault_status", False))

                if earth_fault:
                    st.markdown(
                        '<div class="alarm-crit">🔴 Earth fault detected</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="alarm-ok">🟢 No earth fault detected</div>',
                        unsafe_allow_html=True
                    )

                tap = sv(row, "tap_changer_position", 0)

                st.markdown(
                    f'<div class="alarm-ok">⚙ Tap Changer Position : {tap}</div>',
                    unsafe_allow_html=True
                )

                st.markdown("</div>", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SWITCHGEAR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Main Switchgear Panel":
    st.markdown("## 🔧 Main Switchgear Panel")
    st.caption(
        "Real-time monitoring of switchgear health, breaker operation, "
        "busbar condition, protection systems and power quality."
    )
    
    row = get_row(swgr_df, location)

    if row is None:
        st.warning("Switchgear data not available.")
    else:
        brk    = str(sv(row, "main_breaker_position",      "unknown"))
        sf6    = sv(row, "sf6_gas_pressure_bar",           0)
        busR   = sv(row, "busbar_temperature_phase_r_c",   0)
        freq   = sv(row, "grid_frequency_hz",             50)
        pd_lvl = sv(row, "partial_discharge_level_pc",     0)
        wear   = sv(row, "contact_wear_index_pct",         0)
        pwr    = sv(row, "active_power_kw",                0)
        pf     = sv(row, "power_factor",                   1)

        status = "🟢 HEALTHY"
        if pd_lvl > 20:
            status = "🟡 WARNING"
        if pd_lvl > 50:
            status = "🔴 CRITICAL"

        st.info(
            f"**System Status:** {status} | "
            f"Breaker: **{brk.upper()}** | "
            f"Grid Frequency: **{freq:.2f} Hz**"
        )

        kpi_row([
            ("Breaker Position",  brk.upper(),    "", "#a78bfa"),
            ("SF6 Gas Pressure",  f"{sf6:.2f}",  "bar","#4a9eff"),
            ("Contact Wear",      f"{wear:.1f}", "%", "#ffb347"),
            ("PD Level", f"{pd_lvl:.0f}","pC","#ff4d6d"),
            ("Active Power",      f"{pwr:.2f}",  "kW","#00d4aa"),
            ("Power Factor",      f"{pf:.4f}",   "",  "#a78bfa"),
        ])
        
        st.markdown("### ⚙️ Engineering Metrics")
        
        kpi_row([
            ("Breaker Temp", f"{sv(row,'breaker_contact_temperature_c',0):.1f}", "°C", "#ff4d6d"),
            ("Load Factor", f"{sv(row,'load_factor_pct',0):.1f}", "%", "#4a9eff"),
            ("Operations", f"{sv(row,'breaker_total_operations_counter',0):,.0f}", "", "#a78bfa"),
            ("Open Time", f"{sv(row,'breaker_opening_time_ms',0):.1f}", "ms", "#00d4aa"),
            ("Close Time", f"{sv(row,'breaker_closing_time_ms',0):.1f}", "ms", "#00d4aa"),
            ("Contact Resistance", f"{sv(row,'contact_resistance_microohm',0):.1f}", "µΩ", "#ffb347"),
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            win = get_window(swgr_df, location)
            if not win.empty:
                fig = go.Figure()
                for col_n, color, lbl in [
                    ("busbar_temperature_phase_r_c", "#ff4d6d", "Phase R"),
                    ("busbar_temperature_phase_y_c", "#ffb347", "Phase Y"),
                    ("busbar_temperature_phase_b_c", "#4a9eff", "Phase B"),
                ]:
                    fig.add_trace(go.Scatter(x=win["timestamp"], y=win[col_n],
                        line=dict(color=color, width=1.5), name=lbl))
                fig.update_layout(**PLOT_CFG, title="Busbar Temp (°C)", height=220)
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            ts_chart(swgr_df, location,
                     "partial_discharge_level_pc",
                     "Partial Discharge (pC)", "#ff4d6d")
                     
        with c3:
            ts_chart(swgr_df, location,
                     "insulation_resistance_mohm",
                     "Insulation Resistance (MΩ)", "#00d4aa")

        c4, c5 = st.columns(2)

        with c4:
            win = get_window(swgr_df, location)
            if not win.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["current_phase_a_a"],
                    name="Phase A",
                    line=dict(color="#ff4d6d", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["current_phase_b_a"],
                    name="Phase B",
                    line=dict(color="#ffb347", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["current_phase_c_a"],
                    name="Phase C",
                    line=dict(color="#4a9eff", width=2)
                ))
                fig.update_layout(
                    **PLOT_CFG,
                    title="Three Phase Current (A)",
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)

        with c5:
            win = get_window(swgr_df, location)
            if not win.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["voltage_phase_a_v"],
                    name="Phase A",
                    line=dict(color="#ff4d6d", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["voltage_phase_b_v"],
                    name="Phase B",
                    line=dict(color="#ffb347", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=win["timestamp"],
                    y=win["voltage_phase_c_v"],
                    name="Phase C",
                    line=dict(color="#4a9eff", width=2)
                ))
                fig.update_layout(
                    **PLOT_CFG,
                    title="Three Phase Voltage (V)",
                    height=250
                )
                st.plotly_chart(fig, use_container_width=True)
                
        st.markdown("---")
        st.markdown("### 🩺 Diagnostics")

        d1, d2, d3, d4 = st.columns(4)

        # ---------------------------------------------------
        # Breaker
        # ---------------------------------------------------
        with d1:
            st.metric("Breaker", brk.upper())
            st.metric("Operations", f"{int(sv(row,'breaker_total_operations_counter',0)):,}")
            st.metric("Wear", f"{wear:.1f}%")

        # ---------------------------------------------------
        # Protection
        # ---------------------------------------------------
        with d2:
            st.metric("OC Relay", sv(row, "overcurrent_relay_51_status", "N/A"))
            st.metric("Earth Fault", sv(row, "earth_fault_relay_51n_status", "N/A"))
            st.metric("87 Relay", sv(row, "differential_protection_87_status", "N/A"))

        # ---------------------------------------------------
        # Communication
        # ---------------------------------------------------
        with d3:
            st.metric("SCADA", sv(row, "scada_ems_communication_status", "N/A"))
            st.metric("IEC 61850", sv(row, "iec61850_goose_comm_status", "N/A"))
            st.metric("Mode", sv(row, "breaker_control_mode", "N/A"))

        # ---------------------------------------------------
        # Equipment
        # ---------------------------------------------------
        with d4:
            st.metric("SF6", f"{sf6:.2f} bar")
            st.metric("Humidity", f"{sv(row,'enclosure_humidity_pct',0):.1f}%")
            st.metric("Cabinet Temp", f"{sv(row,'enclosure_internal_temperature_c',0):.1f} °C")
            
        st.markdown("---")
        st.markdown("### 📋 Live Events")

        events = []

        if brk.lower() == "closed":
            events.append("🟢 Main breaker closed")
        else:
            events.append("🟡 Main breaker open")

        if pd_lvl > 50:
            events.append("🔴 High partial discharge detected")
        elif pd_lvl > 20:
            events.append("🟡 Partial discharge increasing")

        if wear > 80:
            events.append("🟡 Contact wear approaching maintenance limit")

        if str(sv(row, "scada_ems_communication_status", "")).lower() != "connected":
            events.append("🔴 SCADA communication lost")

        if sf6 < 5.5:
            events.append("🔴 SF6 gas pressure low")

        if not events:
            events.append("🟢 No active alarms")

        for event in events:
            st.markdown(event)

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRANSMISSION LINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Transmission & Grid Line":
    st.markdown("## 📡 Transmission & Grid Line")
    st.caption(
        "Real-time monitoring of transmission line loading, electrical parameters, "
        "protection systems and grid conditions."
    )
    
    row = get_row(tline_df, location)

    if row is None:
        st.warning("Transmission line data not available.")

    else:
        loading = sv(row, "line_loading_pct", 0)
        loss    = sv(row, "line_transmission_loss_kw", 0)
        cond_t  = sv(row, "line_conductor_temperature_c", 0)
        freq    = sv(row, "grid_frequency_receiving_end_hz", 50)
        vdrop   = sv(row, "voltage_drop_across_line_v", 0)
        pf      = sv(row, "power_factor_at_pcc", 1)
        
        # ---------------------------------------------------
        # Overall Status
        # ---------------------------------------------------

        status = "🟢 HEALTHY"

        if loading > 80 or cond_t > 75:
            status = "🟡 WARNING"

        if loading > 95 or cond_t > 90:
            status = "🔴 CRITICAL"

        st.info(
            f"**Status:** {status}   |   "
            f"**Loading:** {loading:.1f}%   |   "
            f"**Receiving Frequency:** {freq:.2f} Hz"
        )

        kpi_row([
            ("Line Loading",     f"{loading:.1f}", "%", "#ff6b9d"),
            ("Trans. Loss",      f"{loss:.2f}", "kW", "#ff4d6d"),
            ("Conductor Temp",   f"{cond_t:.1f}", "°C", "#ffb347"),
            ("Receiving Freq",   f"{freq:.3f}", "Hz", "#00d4aa"),
            ("Voltage Drop",     f"{vdrop:.1f}", "V", "#a78bfa"),
            ("Power Factor PCC", f"{pf:.4f}", "", "#4a9eff"),
        ])
        
        st.markdown("### ⚙️ Engineering Metrics")

        kpi_row([
            ("Ampacity", f"{sv(row,'dynamic_line_rating_a',0):.0f}", "A", "#00d4aa"),
            ("Utilisation", f"{sv(row,'line_utilisation_factor_pct',0):.1f}", "%", "#4a9eff"),
            ("Ambient Temp", f"{sv(row,'ambient_air_temperature_c',0):.1f}", "°C", "#ffb347"),
            ("Wind Speed", f"{sv(row,'wind_speed_m_per_s',0):.1f}", "m/s", "#a78bfa"),
            ("Voltage THD", f"{sv(row,'voltage_thd_receiving_end_pct',0):.2f}", "%", "#ff6b9d"),
            ("Current Unbalance", f"{sv(row,'current_unbalance_factor_pct',0):.2f}", "%", "#ff4d6d"),
        ])
        
        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        with c1:
            ts_chart(
                tline_df,
                location,
                "line_loading_pct",
                "Line Loading (%)",
                "#ff6b9d",
            )

        with c2:
            ts_chart(
                tline_df,
                location,
                "line_transmission_loss_kw",
                "Transmission Loss (kW)",
                "#ff4d6d",
            )

        with c3:
            win = get_window(tline_df, location)

            if not win.empty:

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=win["timestamp"],
                        y=win["line_current_phase_a_a"],
                        name="Phase A",
                        line=dict(color="#ff4d6d", width=1.5),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=win["timestamp"],
                        y=win["line_current_phase_b_a"],
                        name="Phase B",
                        line=dict(color="#ffb347", width=1.5),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=win["timestamp"],
                        y=win["line_current_phase_c_a"],
                        name="Phase C",
                        line=dict(color="#4a9eff", width=1.5),
                    )
                )

                fig.update_layout(
                    **PLOT_CFG,
                    title="Line Current (A)",
                    height=220,
                )

                st.plotly_chart(fig, use_container_width=True)

        c4, c5 = st.columns(2)

        with c4:
            win = get_window(tline_df, location)

            if win.empty:
                st.info("No transmission voltage history available.")

            elif "timestamp" not in win.columns:
                st.warning("Timestamp column missing.")

            else:
                required_cols = [
                    "sending_end_voltage_phase_a_v",
                    "sending_end_voltage_phase_b_v",
                    "sending_end_voltage_phase_c_v",
                ]

                missing = [c for c in required_cols if c not in win.columns]

                if missing:
                    st.warning(
                        f"Missing voltage columns: {', '.join(missing)}"
                    )

                else:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=win["timestamp"],
                            y=win["sending_end_voltage_phase_a_v"],
                            name="Phase A",
                            line=dict(color="#ff4d6d", width=1.5),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=win["timestamp"],
                            y=win["sending_end_voltage_phase_b_v"],
                            name="Phase B",
                            line=dict(color="#ffb347", width=1.5),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=win["timestamp"],
                            y=win["sending_end_voltage_phase_c_v"],
                            name="Phase C",
                            line=dict(color="#4a9eff", width=1.5),
                        )
                    )

                    fig.update_layout(
                        **PLOT_CFG,
                        title="Sending End Voltage (V)",
                        height=220,
                    )

                    st.plotly_chart(fig, use_container_width=True)

        with c5:
            ts_chart(
                tline_df,
                location,
                "line_conductor_temperature_c",
                "Conductor Temperature (°C)",
                "#ffb347",
            )
            
        st.markdown("---")
        st.markdown("### 🩺 Diagnostics")

        d1, d2, d3, d4 = st.columns(4)
        
        with d1:
            st.metric("Availability", sv(row, "line_availability_status", "N/A"))
            st.metric("Loading", f"{loading:.1f}%")
            st.metric("Ampacity", f"{sv(row,'dynamic_line_rating_a',0):.0f} A")
            
        with d2:
            st.metric("Distance Relay", sv(row, "distance_relay_21_status", "N/A"))
            st.metric("OC Relay", sv(row, "overcurrent_relay_51_status", "N/A"))
            st.metric("Earth Fault", sv(row, "earth_fault_relay_51n_status", "N/A"))
            
        with d3:
            st.metric("Protection Comm", sv(row, "line_protection_comm_status", "N/A"))
            st.metric("Telemetry", f"{sv(row,'scada_telemetry_latency_ms',0):.1f} ms")
            st.metric("Latency", f"{sv(row,'command_response_time_ms',0):.1f} ms")
            
        with d4:
            st.metric("Ambient", f"{sv(row,'ambient_air_temperature_c',0):.1f} °C")
            st.metric("Wind", f"{sv(row,'wind_speed_m_per_s',0):.1f} m/s")
            st.metric("Lightning", f"{int(sv(row,'lightning_strike_count_day',0))}")
            
        st.markdown("---")
        st.markdown("### 📋 Live Grid Events")

        events = []

        if loading > 90:
            events.append("🔴 Transmission line heavily loaded")
        elif loading > 75:
            events.append("🟡 High transmission line loading")

        if cond_t > 80:
            events.append("🟡 Conductor temperature elevated")

        if bool(sv(row, "vegetation_clearance_violation_status", False)):
            events.append("🟡 Vegetation clearance violation")

        if bool(sv(row, "right_of_way_intrusion_alert", False)):
            events.append("🔴 Right-of-way intrusion detected")

        if bool(sv(row, "bird_streamer_flashover_risk", False)):
            events.append("🟡 Bird streamer flashover risk")

        if int(sv(row, "lightning_strike_count_day", 0)) > 0:
            events.append(
                f"⚡ {int(sv(row,'lightning_strike_count_day',0))} lightning strike(s) recorded today"
            )

        if not events:
            events.append("🟢 No active transmission line events")

        for event in events:
            st.markdown(event)
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALARMS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alarms & Faults":
    st.markdown("## 🚨 Alarms & Faults")
    st.caption(
        "Centralized monitoring of active alarms, equipment faults and historical fault events."
    )

    comps = [
        ("Battery", battery_df),
        ("PCS", pcs_df),
        ("Switchgear", swgr_df),
        ("Trans. Line", tline_df),
    ]

    any_fault = False
    critical_count = 0
    warning_count = 0
    healthy_count = 0

    for name, df_ref in comps:
        r = get_row(df_ref, location)

        if r is None:
            st.markdown(
                f'<div class="alarm-warn">⚠ [{name}] Data not loaded</div>',
                unsafe_allow_html=True
            )
            continue

        # Battery dataset uses fault_code instead of fault_type
        ft = str(sv(r, "fault_type", sv(r, "fault_code", "normal")))

        # Battery doesn't have fault_severity
        sev = str(
            sv(
                r,
                "fault_severity",
                "critical" if sv(r, "is_critical", 0) else "low"
            )
        )

        is_f = int(sv(r, "is_fault", 0))
        is_c = int(sv(r, "is_critical", 0))

        if is_c:
            cls = "alarm-crit"
            icon = "🔴"
            any_fault = True
            critical_count += 1
        elif is_f:
            cls = "alarm-warn"
            icon = "🟡"
            any_fault = True
            warning_count += 1
        else:
            cls = "alarm-ok"
            icon = "🟢"
            healthy_count += 1

        msg = f"{icon} [{name}] {ft.replace('_',' ').upper()} — Severity: {sev.upper()}"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    st.markdown("### 📊 Alarm Summary")

    kpi_row([
        ("Critical", f"{critical_count}", "", "#ff4d6d"),
        ("Warnings", f"{warning_count}", "", "#ffb347"),
        ("Healthy", f"{healthy_count}", "", "#00d4aa"),
        ("Components", f"{len(comps)}", "", "#4a9eff"),
    ])
    
    st.markdown("### 📊 Alarm Summary")

    kpi_row([
        ("Critical", f"{critical_count}", "", "#ff4d6d"),
        ("Warnings", f"{warning_count}", "", "#ffb347"),
        ("Healthy", f"{healthy_count}", "", "#00d4aa"),
        ("Components", f"{len(comps)}", "", "#4a9eff"),
    ])

    if not any_fault:
        st.success("✅ All systems normal — no active faults.")

    st.markdown("---")
    st.markdown("### 📈 Fault Timeline")
    fault_frames = []

    for name, df_ref in comps:

        if df_ref is None or df_ref.empty:
            continue

        if "is_fault" not in df_ref.columns:
            continue

        loc_df = (
            df_ref[df_ref["location"] == location]
            if "location" in df_ref.columns
            else df_ref
        )

        faults = loc_df[loc_df["is_fault"] == 1].copy()

        if faults.empty:
            continue

        # Battery fix
        if "fault_type" not in faults.columns and "fault_code" in faults.columns:
            faults["fault_type"] = faults["fault_code"]

        if "fault_severity" not in faults.columns:
            faults["fault_severity"] = faults["is_critical"].apply(
                lambda x: "critical" if x == 1 else "low"
            )

        faults = faults[
            ["timestamp", "fault_type", "fault_severity", "is_critical"]
        ].copy()
        
        # Fix mixed type columns — Arrow needs consistent types
        faults["fault_type"] = faults["fault_type"].astype(str)
        faults["fault_severity"] = faults["fault_severity"].astype(str)
        faults["is_critical"] = faults["is_critical"].astype(int)

        faults["component"] = name
        fault_frames.append(faults)

    if fault_frames:
        all_faults = (
            pd.concat(fault_frames)
            .sort_values("timestamp", ascending=False)
            .head(50)
        )

        if len(all_faults) > 0:
            import plotly.express as px

            fig = px.scatter(
                all_faults,
                x="timestamp",
                y="component",
                color="fault_severity",
                color_discrete_map={
                    "critical": "#ff4d6d",
                    "high": "#ff8c00",
                    "medium": "#ffb347",
                    "low": "#00d4aa",
                    "none": "#3d5a7a",
                },
                title="Fault Timeline",
                height=280,
            )
            
            fig.update_layout(
                **PLOT_CFG,
                legend_title="Severity",
                hovermode="closest",
            )

            fig.update_yaxes(categoryorder="total ascending")

            fig.update_traces(marker=dict(size=10))

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📊 Fault Severity Distribution")

            severity_counts = (
                all_faults["fault_severity"]
                .astype(str)
                .str.upper()
                .value_counts()
                .reset_index()
            )

            severity_counts.columns = ["Severity", "Count"]

            fig2 = px.pie(
                severity_counts,
                names="Severity",
                values="Count",
                hole=0.6,
                color="Severity",
                color_discrete_map={
                    "CRITICAL": "#ff4d6d",
                    "HIGH": "#ff8c00",
                    "MEDIUM": "#ffb347",
                    "LOW": "#00d4aa",
                },
            )

            fig2.update_layout(
                **PLOT_CFG,
                height=300,
            )

            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("### 📋 Recent Fault Log")
            
            st.dataframe(
                all_faults.reset_index(drop=True),
                hide_index=True,
                width="stretch",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST & AI ADVISORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast & AI Advisory":

    st.markdown("## 🤖 Forecast & AI Advisory")
    
    if st.session_state.last_agent_call_time > 0:
        age = int(time.time() - st.session_state.last_agent_call_time)
        st.caption(f"Last AI analysis: {age} seconds ago")

    st.caption(
        "Fault diagnosis, risk assessment and operational recommendations "
        "using live BESS telemetry and OEM knowledge."
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Autonomous Agent")

    # =====================================================
    # TEST MODE
    # =====================================================
    # Change to False when using real faults

    fault_component = None
    active_row = None

    components = [
        ("Battery", battery_df),
        ("PCS", pcs_df),
        ("Transformer", xfmr_df),
        ("Switchgear", swgr_df),
        ("Transmission Line", tline_df),
    ]

    # Collect all active faults
    active_faults = []

    for name, df in components:
        row = get_row(df, location)

        if row is None:
            continue

        if int(sv(row, "is_fault", 0)) != 1:
            continue

        if name in st.session_state.acknowledged_faults:
            continue

        active_faults.append((name, row))

    if active_faults:
        # Only one active fault → select it automatically
        if len(active_faults) == 1:
            fault_component, active_row = active_faults[0]

        # Multiple active faults → let user choose
        else:
            selected_component = st.selectbox(
                "Select Active Fault",
                [name for name, _ in active_faults]
            )

            for name, row in active_faults:
                if name == selected_component:
                    fault_component = name
                    active_row = row
                    break

    # -----------------------------
    # Fake fault for demo
    # -----------------------------
    

    # =====================================================

    if active_row is None:
        st.session_state.last_fault = None
        st.session_state.last_advice = None
        st.success("✅ System Nominal. No expert intervention required.")

    else:
        fault_name = str(sv(active_row, "fault_type", "Unknown Fault"))

        st.warning(
            f"🚨 **Active Fault:** {fault_component} • {fault_name}"
        )

        # ------------------------------------------
        # Only useful telemetry
        # ------------------------------------------
        if fault_component == "Battery":
            telemetry = {
                "Temperature (°C)": sv(active_row, "average_cell_temperature_c", 0),
                "SOC (%)": sv(active_row, "state_of_charge_soc_pct", 0),
                "SOH (%)": sv(active_row, "state_of_health_soh_pct", 0),
                "Voltage (V)": sv(active_row, "pack_voltage_v", 0),
                "Power (kW)": sv(active_row, "battery_power_kw", 0),
            }
        elif fault_component == "PCS":
            telemetry = {
                "IGBT Temperature (°C)": sv(active_row, "igbt_temperature_c", 0),
                "Efficiency (%)": sv(active_row, "conversion_efficiency_pct", 0),
                "Active Power (kW)": sv(active_row, "active_power_kw", 0),
                "Grid Frequency (Hz)": sv(active_row, "grid_frequency_hz", 0),
                "DC Bus Voltage (V)": sv(active_row, "dc_bus_voltage_v", 0),
            }
        elif fault_component == "Transformer":
            telemetry = {
                "Top Oil Temp (°C)": sv(active_row, "top_oil_temp_c", 0),
                "Winding Temp (°C)": sv(active_row, "winding_temp_c", 0),
                "Hotspot Temp (°C)": sv(active_row, "hotspot_temp_c", 0),
                "Loading (%)": sv(active_row, "transformer_loading_pct", 0),
                "Efficiency (%)": sv(active_row, "transformer_efficiency_pct", 0),
            }
        elif fault_component == "Switchgear":
            telemetry = {
                "Breaker Position": sv(active_row, "main_breaker_position", ""),
                "SF6 Pressure (bar)": sv(active_row, "sf6_gas_pressure_bar", 0),
                "Partial Discharge (pC)": sv(active_row, "partial_discharge_level_pc", 0),
                "Contact Wear (%)": sv(active_row, "contact_wear_index_pct", 0),
                "Active Power (kW)": sv(active_row, "active_power_kw", 0),
            }
        elif fault_component == "Transmission Line":
            telemetry = {
                "Line Loading (%)": sv(active_row, "line_loading_pct", 0),
                "Conductor Temp (°C)": sv(active_row, "line_conductor_temperature_c", 0),
                "Transmission Loss (kW)": sv(active_row, "line_transmission_loss_kw", 0),
                "Receiving Frequency (Hz)": sv(active_row, "grid_frequency_receiving_end_hz", 0),
                "Voltage Drop (V)": sv(active_row, "voltage_drop_across_line_v", 0),
            }
        else:
            telemetry = {}

        current_fault = (
            location,
            fault_component,
            fault_name,
        )

        time_since_last_call = (
            time.time() - st.session_state.last_agent_call_time
        )

        fault_changed = (
            st.session_state.last_fault != current_fault
        )

        regenerate = (
            fault_changed
            or (
                st.session_state.last_advice is None
                and time_since_last_call > 300
            )
        )

        if regenerate:
            with st.spinner("Consulting ABB Expert Agent..."):
                prompt = f"""
You are an ABB BESS Plant Operations Expert.

You are an expert in:
- Battery Energy Storage Systems
- Power Conversion Systems (PCS)
- Distribution Transformers
- Switchgear
- Transmission Lines

Analyze the fault using the live telemetry below.

Component:
{fault_component}

Fault:
{fault_name}

Telemetry:
{telemetry}

Provide:

1. Severity
2. Risk Analysis
3. Recommended Actions
"""
                try:
                    advice = generate_rag_advisory(prompt)

                    st.session_state.last_fault = current_fault
                    st.session_state.last_advice = advice
                    st.session_state.last_agent_call_time = time.time()

                except Exception as e:
                    st.error(f"Agent error: {e}")
                    st.stop()
        else:
            advice = st.session_state.last_advice

        # Restored the line requested, aligned to the correct indentation level
        advice = st.session_state.last_advice
        
        st.success("✅ AI analysis completed successfully")

        kpi_row([
            ("Severity", advice.severity.upper(), "", "#ff4d6d"),
            ("Component", advice.matched_component, "", "#4a9eff"),
            ("AI Status", "READY", "", "#00d4aa"),
        ])

        st.markdown("### ⚠️ AI Risk Analysis")

        with st.container(border=True):
            st.markdown(advice.risk_analysis)

        st.markdown("### ✅ Recommended Actions")

        with st.container(border=True):
            for action in advice.actions_required:
                st.markdown(f"✔ {action}")
        
        st.markdown("---")

        if st.button("✅ Mark Actions Completed"):
            st.session_state.acknowledged_faults.add(fault_component)

            st.session_state.last_fault = None
            st.session_state.last_advice = None

            st.success("Fault acknowledged. Monitoring next fault...")

            st.rerun()

    # Footer elements shown regardless of fault status
    st.markdown("---")

    st.caption(
        "Analysis generated using NeoAI Expert Agent • "
        "Groq Llama 3.1 • OEM Knowledge Base • Live Telemetry"
    )
