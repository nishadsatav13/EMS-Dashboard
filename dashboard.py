

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
count = st_autorefresh(interval=5000, key="neoai_refresh")
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
if "last_fault" not in st.session_state:
    st.session_state.last_fault = None

if "last_advice" not in st.session_state:
    st.session_state.last_advice = None

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

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔌 Distribution Transformer":
    st.markdown("## 🔌 Distribution Transformer")

    if xfmr_df.empty:
        st.warning("Transformer dataset not loaded. Add transformer_neoai_dataset.csv to repo.")
    else:
        row = get_row(xfmr_df, location)
        if row is None:
            st.warning("No transformer data for this location.")
        else:
            def find_col(df, patterns):
                for p in patterns:
                    for c in df.columns:
                        if p.lower() in c.lower():
                            return c
                return None

            load_col  = find_col(xfmr_df, ["load_pct","load_factor","loading"])
            oil_col   = find_col(xfmr_df, ["oil_temp","oil_temperature"])
            wind_col  = find_col(xfmr_df, ["winding_temp","winding_temperature"])
            hot_col   = find_col(xfmr_df, ["hotspot","hot_spot"])

            load_v = f"{sv(row, load_col, 0):.1f}"  if load_col  else "—"
            oil_v  = f"{sv(row, oil_col,  0):.1f}"  if oil_col   else "—"
            wind_v = f"{sv(row, wind_col, 0):.1f}"  if wind_col  else "—"
            hot_v  = f"{sv(row, hot_col,  0):.1f}"  if hot_col   else "—"

            kpi_row([
                ("Load Factor",     load_v, "%",  "#ffb347"),
                ("Oil Temp",        oil_v,  "°C", "#ff4d6d"),
                ("Winding Temp",    wind_v, "°C", "#ff6b9d"),
                ("Hotspot Temp",    hot_v,  "°C", "#a78bfa"),
            ])

            if load_col:
                ts_chart(xfmr_df, location, load_col,
                         "Transformer Load (%)", "#ffb347")
            if oil_col:
                ts_chart(xfmr_df, location, oil_col,
                         "Oil Temperature (°C)", "#ff4d6d")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SWITCHGEAR
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧 Main Switchgear Panel":
    st.markdown("## 🔧 Main Switchgear Panel")
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

        kpi_row([
            ("Breaker Position",  brk.upper(),    "", "#a78bfa"),
            ("SF6 Gas Pressure",  f"{sf6:.2f}",  "bar","#4a9eff"),
            ("Contact Wear",      f"{wear:.1f}", "%", "#ffb347"),
            ("Partial Discharge", f"{pd_lvl:.0f}","pC","#ff4d6d"),
            ("Active Power",      f"{pwr:.2f}",  "kW","#00d4aa"),
            ("Power Factor",      f"{pf:.4f}",   "",  "#a78bfa"),
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
            ts_chart(swgr_df, location,
                     "active_power_kw", "Active Power (kW)", "#a78bfa")
        with c5:
            ts_chart(swgr_df, location,
                     "grid_frequency_hz", "Grid Frequency (Hz)", "#4a9eff")

# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRANSMISSION LINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Transmission & Grid Line":
    st.markdown("## 📡 Transmission & Grid Line")
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

        kpi_row([
            ("Line Loading",     f"{loading:.1f}", "%", "#ff6b9d"),
            ("Trans. Loss",      f"{loss:.2f}", "kW", "#ff4d6d"),
            ("Conductor Temp",   f"{cond_t:.1f}", "°C", "#ffb347"),
            ("Receiving Freq",   f"{freq:.3f}", "Hz", "#00d4aa"),
            ("Voltage Drop",     f"{vdrop:.1f}", "V", "#a78bfa"),
            ("Power Factor PCC", f"{pf:.4f}", "", "#4a9eff"),
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
            ts_chart(
                tline_df,
                location,
                "grid_frequency_receiving_end_hz",
                "Receiving End Frequency (Hz)",
                "#00d4aa",
            )

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
# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALARMS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alarms & Faults":
    st.markdown("## 🚨 Alarms & Faults")

    comps = [
        ("Battery", battery_df),
        ("PCS", pcs_df),
        ("Switchgear", swgr_df),
        ("Trans. Line", tline_df),
    ]

    any_fault = False

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
        elif is_f:
            cls = "alarm-warn"
            icon = "🟡"
            any_fault = True
        else:
            cls = "alarm-ok"
            icon = "🟢"

        msg = f"{icon} [{name}] {ft.replace('_',' ').upper()} — Severity: {sev.upper()}"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    if not any_fault:
        st.success("✅ All systems normal — no active faults.")

    st.markdown("---")
    st.markdown("### Fault Event History")

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

            fig.update_layout(**PLOT_CFG)
            fig.update_traces(marker=dict(size=10))

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(all_faults.reset_index(drop=True), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST & AI ADVISORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast & AI Advisory":
    st.markdown("## 🔮 Forecast & AI Advisory")

    # ... [Keep your existing forecast charts here] ...

    st.markdown("---")
    st.markdown("### 🤖 Autonomous ABB Expert Agent")

    # Find the first component with an active fault
    fault_component = None
    active_row = None

    components = [
        ("Battery", battery_df),
        ("PCS", pcs_df),
        ("Transformer", xfmr_df),
        ("Switchgear", swgr_df),
        ("Transmission Line", tline_df)
    ]

    for name, df in components:
        row = get_row(df, location)

        if row is not None and int(sv(row, "is_fault", 0)) == 1:
            fault_component = name
            active_row = row
            break

    # =====================================================
    # TEMPORARY TEST MODE (REMOVE AFTER TESTING)
    # =====================================================
    if active_row is None:
        test_row = get_row(battery_df, location)

        if test_row is not None:
            active_row = test_row.copy()
            fault_component = "Battery"

            active_row["is_fault"] = 1
            active_row["fault_type"] = "Cell Over Temperature"

            # Optional: make telemetry look realistic
            active_row["temperature"] = 72
            active_row["soc"] = 91
    # =====================================================

    if active_row is not None:

        fault_name = sv(active_row, "fault_type", "Unknown Anomaly")
        telemetry = active_row.to_dict()

        st.error(f"🚨 **Critical Alert Detected:** {fault_name}")

        current_fault = f"{fault_component}:{fault_name}"

        # Call AI only when the fault changes
        if st.session_state.last_fault != current_fault:

            with st.spinner("ABB Agent is consulting the manual..."):

                advice = generate_rag_advisory(
                    f"""
Component: {fault_component}

Fault: {fault_name}

Telemetry:
{telemetry}
"""
                )

            st.session_state.last_fault = current_fault
            st.session_state.last_advice = advice

        else:
            advice = st.session_state.last_advice

        # Display AI Result
        st.write(f"**Severity:** `{advice.severity}`")
        st.write(f"**Matched Component:** {advice.matched_component}")
        st.write(f"**Risk Analysis:** {advice.risk_analysis}")

        st.write("**Required Actions:**")
        for action in advice.actions_required:
            st.markdown(f"- ✅ {action}")

    else:
        # Clear cache when system returns to normal
        st.session_state.last_fault = None
        st.session_state.last_advice = None

        st.success("✅ System Nominal. No expert intervention required.")
# ═══════════════════════════════════════════════════════════════════════════════
# ALWAYS-ON LIVE REFRESH — 3 second tick, no toggle needed
# ═══════════════════════════════════════════════════════════════════════════════
