"""
NeoAI — Enterprise EMS Dashboard
Streamlit Cloud ready — reads datasets from repo, simulates live data
No separate simulator.py needed
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeoAI EMS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #080d14; }
  .block-container { padding: 1rem 1.5rem; }
  [data-testid="stSidebar"] { background: #0a1628; }
  .kpi-row { display: flex; gap: 12px; margin-bottom: 12px; }
  .kpi-box {
    background: #0d1f3c; border: 1px solid #1e3a5f;
    border-radius: 10px; padding: 14px 18px; flex: 1; text-align: center;
  }
  .kpi-label { font-size: 10px; color: #5a7a9a; text-transform: uppercase;
               letter-spacing: 1.5px; margin-bottom: 4px; }
  .kpi-val   { font-size: 24px; font-weight: 700; color: #00d4aa; }
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
  #MainMenu,footer,header{visibility:hidden;}
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
@st.cache_data(show_spinner="Loading NeoAI datasets…")
def load_data():
    dfs = {}
    try:
        dfs["battery"] = pd.read_csv("battery_neoai_dataset.csv",
                                      parse_dates=["timestamp"])
    except Exception as e:
        st.error(f"Battery dataset error: {e}")
        dfs["battery"] = pd.DataFrame()

    try:
        dfs["pcs"] = pd.read_csv("pcs_neoai_dataset.csv",
                                  parse_dates=["timestamp"])
    except Exception as e:
        st.error(f"PCS dataset error: {e}")
        dfs["pcs"] = pd.DataFrame()

    try:
        dfs["transformer"] = pd.read_csv("transformer_neoai_dataset.csv",
                                          parse_dates=["timestamp"])
    except Exception as e:
        st.warning(f"Transformer dataset not found: {e}")
        dfs["transformer"] = pd.DataFrame()

    try:
        dfs["switchgear"] = pd.read_excel("switchgear_neoai_dataset.xlsx")
        dfs["switchgear"]["timestamp"] = pd.to_datetime(
            dfs["switchgear"]["timestamp"])
    except Exception as e:
        st.error(f"Switchgear dataset error: {e}")
        dfs["switchgear"] = pd.DataFrame()

    try:
        dfs["tline"] = pd.read_excel("transmission_line_neoai_dataset.xlsx")
        dfs["tline"]["timestamp"] = pd.to_datetime(dfs["tline"]["timestamp"])
    except Exception as e:
        st.error(f"Transmission line dataset error: {e}")
        dfs["tline"] = pd.DataFrame()

    return dfs

dfs = load_data()
battery_df    = dfs["battery"]
pcs_df        = dfs["pcs"]
xfmr_df       = dfs["transformer"]
swgr_df       = dfs["switchgear"]
tline_df      = dfs["tline"]

# ── Session state (simulator tick) ────────────────────────────────────────────
if "tick" not in st.session_state:
    st.session_state.tick = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#00d4aa'>⚡ NeoAI EMS</h2>",
                unsafe_allow_html=True)
    st.markdown("*Neosol Energy Systems Pvt Ltd*")
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

    refresh_rate = st.slider("Auto-refresh (seconds)", 5, 60, 10)
   live_mode    = st.toggle("Live Mode", value=False)

    st.markdown("---")
    st.markdown(f"**Tick:** `{st.session_state.tick}`")

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
        x=win["timestamp"], y=win[y_col],
        mode="lines", line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba({int(color[1:3],16)},"
                  f"{int(color[3:5],16)},"
                  f"{int(color[5:7],16)},0.06)"
    ))
    fig.update_layout(**PLOT_CFG, title=title, height=220)
    st.plotly_chart(fig, use_container_width=True)

# ── KPI helper ────────────────────────────────────────────────────────────────
def kpi_row(items):
    cols = st.columns(len(items))
    for col, (label, val, unit, color) in zip(cols, items):
        col.markdown(f"""
        <div class="kpi-box" style="border-top:3px solid {color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-val" style="color:{color}">{val}
            <span class="kpi-unit">{unit}</span>
          </div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MASTER OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Master Overview":
    st.markdown("## 🌍 Master System Overview")

    batt_r = get_row(battery_df, location)
    pcs_r  = get_row(pcs_df,     location)
    swgr_r = get_row(swgr_df,    location)

    soc  = f"{sv(batt_r,'state_of_charge_soc_pct',0):.1f}"    if batt_r is not None else "—"
    soh  = f"{sv(batt_r,'state_of_health_soh_pct',0):.1f}"    if batt_r is not None else "—"
    pwr  = f"{sv(pcs_r,'active_power_kw',0):.1f}"             if pcs_r  is not None else "—"
    freq = f"{sv(pcs_r,'grid_frequency_hz',50):.3f}"           if pcs_r  is not None else "—"
    pf   = f"{sv(pcs_r,'power_factor_overall',1):.4f}"         if pcs_r  is not None else "—"
    brk  = str(sv(swgr_r,'main_breaker_position','—'))          if swgr_r is not None else "—"

    kpi_row([
        ("State of Charge",   soc,  "%",  "#00d4aa"),
        ("State of Health",   soh,  "%",  "#4a9eff"),
        ("PCS Active Power",  pwr,  "kW", "#ffb347"),
        ("Grid Frequency",    freq, "Hz", "#a78bfa"),
        ("Power Factor",      pf,   "",   "#00d4aa"),
        ("Breaker Position",  brk,  "",   "#ff6b9d"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        ts_chart(battery_df, location,
                 "state_of_charge_soc_pct", "Battery SOC (%)", "#00d4aa")
    with c2:
        ts_chart(pcs_df, location,
                 "active_power_kw", "PCS Active Power (kW)", "#4a9eff")

    st.markdown("---")
    st.markdown("### Component Health Summary")
    c1, c2, c3, c4 = st.columns(4)
    for col, name, df_ref in [
        (c1, "🔋 Battery",      battery_df),
        (c2, "⚡ PCS",          pcs_df),
        (c3, "🔧 Switchgear",   swgr_df),
        (c4, "📡 Trans. Line",  tline_df),
    ]:
        r = get_row(df_ref, location)
        is_f = int(sv(r, "is_fault", 0)) if r is not None else -1
        status = "NO DATA" if is_f == -1 else ("⚠ FAULT" if is_f else "✅ NORMAL")
        color  = "#ffb347" if is_f == -1 else ("#ff4d6d" if is_f else "#00d4aa")
        col.markdown(f"""
        <div class="kpi-box" style="border-top:3px solid {color}">
          <div class="kpi-label">{name}</div>
          <div class="kpi-val" style="color:{color};font-size:16px">{status}</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: BATTERY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔋 Battery Storage (LFP)":
    st.markdown("## 🔋 Battery Storage — LFP")
    row = get_row(battery_df, location)

    if row is None:
        st.warning("Battery data not available.")
    else:
        soc  = sv(row, "state_of_charge_soc_pct",    0)
        soh  = sv(row, "state_of_health_soh_pct",    0)
        pwr  = sv(row, "battery_power_kw",            0)
        temp = sv(row, "average_cell_temperature_c",  0)
        ir   = sv(row, "internal_resistance_mohm",    0)
        rul  = sv(row, "remaining_useful_life_years",  0)
        cyc  = sv(row, "cycle_count",                 0)
        volt = sv(row, "pack_voltage_v",              0)

        kpi_row([
            ("SOC",           f"{soc:.1f}",  "%",   "#00d4aa"),
            ("SOH",           f"{soh:.2f}",  "%",   "#4a9eff"),
            ("Battery Power", f"{pwr:.2f}",  "kW",  "#ffb347"),
            ("Avg Cell Temp", f"{temp:.1f}", "°C",  "#ff6b9d"),
            ("RUL",           f"{rul:.1f}",  "yrs", "#a78bfa"),
            ("Cycle Count",   f"{cyc:.0f}",  "",    "#00d4aa"),
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=soc,
                number={"suffix": "%", "font": {"color": "#00d4aa", "size": 28}},
                title={"text": "SOC", "font": {"color": "#5a7a9a", "size": 12}},
                gauge={
                    "axis":  {"range": [0, 100], "tickcolor": "#1a2744"},
                    "bar":   {"color": "#00d4aa"},
                    "bgcolor": "#0d1f3c",
                    "bordercolor": "#1e3a5f",
                    "steps": [
                        {"range": [0, 20],  "color": "#2d0a0a"},
                        {"range": [20, 50], "color": "#1a1000"},
                        {"range": [50, 100],"color": "#001a12"},
                    ],
                    "threshold": {"line": {"color": "#ff4d6d", "width": 3},
                                  "thickness": 0.75, "value": 20},
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                              height=230, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            ts_chart(battery_df, location,
                     "state_of_charge_soc_pct", "SOC Trend (%)", "#00d4aa")

        with c3:
            win = get_window(battery_df, location)
            if not win.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=win["timestamp"],
                    y=win["max_cell_temperature_c"],
                    line=dict(color="#ff4d6d", width=1.5), name="Max"))
                fig.add_trace(go.Scatter(x=win["timestamp"],
                    y=win["average_cell_temperature_c"],
                    line=dict(color="#ffb347", width=2), name="Avg"))
                fig.add_trace(go.Scatter(x=win["timestamp"],
                    y=win["min_cell_temperature_c"],
                    line=dict(color="#00d4aa", width=1.5), name="Min"))
                fig.update_layout(**PLOT_CFG, title="Cell Temp (°C)", height=230)
                st.plotly_chart(fig, use_container_width=True)

        c4, c5 = st.columns(2)
        with c4:
            ts_chart(battery_df, location,
                     "state_of_health_soh_pct", "SOH Trend (%)", "#4a9eff")
        with c5:
            ts_chart(battery_df, location,
                     "internal_resistance_mohm",
                     "Internal Resistance (mΩ) — Aging Indicator", "#a78bfa")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PCS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Power Conversion (PCS)":
    st.markdown("## ⚡ Power Conversion System (PCS)")
    row = get_row(pcs_df, location)

    if row is None:
        st.warning("PCS data not available.")
    else:
        eff  = sv(row, "conversion_efficiency_pct", 0)
        pwr  = sv(row, "active_power_kw",           0)
        rpwr = sv(row, "reactive_power_kvar",        0)
        freq = sv(row, "grid_frequency_hz",         50)
        pf   = sv(row, "power_factor_overall",       1)
        igbt = sv(row, "igbt_temperature_c",         0)
        thd  = sv(row, "voltage_thd_pct",            0)
        mode = str(sv(row, "operating_mode",    "standby"))

        kpi_row([
            ("Active Power",   f"{pwr:.2f}",  "kW",  "#4a9eff"),
            ("Conv. Eff.",      f"{eff:.2f}",  "%",   "#00d4aa"),
            ("Power Factor",   f"{pf:.4f}",   "",    "#a78bfa"),
            ("Grid Frequency", f"{freq:.3f}", "Hz",  "#ffb347"),
            ("IGBT Temp",      f"{igbt:.1f}", "°C",  "#ff6b9d"),
            ("Mode",           mode.upper(),  "",    "#4a9eff"),
        ])

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
            # Use generic column search in case column names differ
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
# PAGE: TRANSMISSION LINE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Transmission & Grid Line":
    st.markdown("## 📡 Transmission & Grid Line")
    row = get_row(tline_df, location)

    if row is None:
        st.warning("Transmission line data not available.")
    else:
        loading = sv(row, "line_loading_pct",                  0)
        loss    = sv(row, "line_transmission_loss_kw",          0)
        cond_t  = sv(row, "line_conductor_temperature_c",       0)
        freq    = sv(row, "grid_frequency_receiving_end_hz",   50)
        vdrop   = sv(row, "voltage_drop_across_line_v",         0)
        pf      = sv(row, "power_factor_at_pcc",                1)

        kpi_row([
            ("Line Loading",      f"{loading:.1f}", "%",  "#ff6b9d"),
            ("Trans. Loss",       f"{loss:.2f}",    "kW", "#ff4d6d"),
            ("Conductor Temp",    f"{cond_t:.1f}",  "°C", "#ffb347"),
            ("Receiving Freq",    f"{freq:.3f}",    "Hz", "#00d4aa"),
            ("Voltage Drop",      f"{vdrop:.1f}",   "V",  "#a78bfa"),
            ("Power Factor PCC",  f"{pf:.4f}",      "",   "#4a9eff"),
        ])

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            ts_chart(tline_df, location,
                     "line_loading_pct", "Line Loading (%)", "#ff6b9d")
        with c2:
            ts_chart(tline_df, location,
                     "line_transmission_loss_kw",
                     "Transmission Loss (kW)", "#ff4d6d")
        with c3:
            ts_chart(tline_df, location,
                     "grid_frequency_receiving_end_hz",
                     "Receiving End Frequency (Hz)", "#00d4aa")

        c4, c5 = st.columns(2)
        with c4:
            win = get_window(tline_df, location)
            if not win.empty:
                fig = go.Figure()
                for col_n, color, lbl in [
                    ("sending_end_voltage_phase_a_v", "#ff4d6d", "Phase A"),
                    ("sending_end_voltage_phase_b_v", "#ffb347", "Phase B"),
                    ("sending_end_voltage_phase_c_v", "#4a9eff", "Phase C"),
                ]:
                    fig.add_trace(go.Scatter(x=win["timestamp"], y=win[col_n],
                        line=dict(color=color, width=1.5), name=lbl))
                fig.update_layout(**PLOT_CFG,
                                   title="Sending End Voltage (V)", height=220)
                st.plotly_chart(fig, use_container_width=True)

        with c5:
            ts_chart(tline_df, location,
                     "line_conductor_temperature_c",
                     "Conductor Temperature (°C)", "#ffb347")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ALARMS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Alarms & Faults":
    st.markdown("## 🚨 Alarms & Faults")

    comps = [
        ("Battery",        battery_df),
        ("PCS",            pcs_df),
        ("Switchgear",     swgr_df),
        ("Trans. Line",    tline_df),
    ]

    any_fault = False
    for name, df_ref in comps:
        r = get_row(df_ref, location)
        if r is None:
            st.markdown(f'<div class="alarm-warn">⚠ [{name}] Data not loaded</div>',
                        unsafe_allow_html=True)
            continue
        ft   = str(sv(r, "fault_type",     "normal"))
        sev  = str(sv(r, "fault_severity", "none"))
        is_f = int(sv(r, "is_fault",        0))
        is_c = int(sv(r, "is_critical",     0))

        if is_c:
            cls = "alarm-crit"; icon = "🔴"; any_fault = True
        elif is_f:
            cls = "alarm-warn"; icon = "🟡"; any_fault = True
        else:
            cls = "alarm-ok";   icon = "🟢"

        msg = f"{icon} [{name}]  {ft.replace('_',' ').upper()}  —  Severity: {sev.upper()}"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    if not any_fault:
        st.success("✅ All systems normal — no active faults.")

    st.markdown("---")
    st.markdown("### Fault Event History")

    fault_frames = []
    for name, df_ref in comps:
        if df_ref is not None and not df_ref.empty and "is_fault" in df_ref.columns:
            loc_df = df_ref[df_ref["location"] == location] \
                     if "location" in df_ref.columns else df_ref
            faults = loc_df[loc_df["is_fault"] == 1][
                ["timestamp", "fault_type", "fault_severity", "is_critical"]
            ].copy()
            faults["component"] = name
            fault_frames.append(faults)

    if fault_frames:
        all_faults = pd.concat(fault_frames).sort_values(
            "timestamp", ascending=False).head(50)
        if len(all_faults) > 0:
            import plotly.express as px
            fig = px.scatter(
                all_faults, x="timestamp", y="component",
                color="fault_severity",
                color_discrete_map={
                    "critical": "#ff4d6d", "high": "#ff8c00",
                    "medium":   "#ffb347", "low": "#00d4aa",
                    "none":     "#3d5a7a",
                },
                title="Fault Timeline",
                height=280,
            )
            fig.update_layout(**PLOT_CFG)
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(all_faults.reset_index(drop=True),
                         use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST & AI ADVISORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast & AI Advisory":
    st.markdown("## 🔮 Forecast & AI Advisory")

    hours         = np.arange(0, 24)
    pv_forecast   = np.clip(np.sin(hours / 24 * 2 * np.pi) * 30 + 20, 0, None)
    load_forecast = np.cos(hours / 24 * 2 * np.pi) * 10 + 25
    np.random.seed(st.session_state.tick % 100)
    price_forecast = np.random.normal(6, 0.5, 24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=pv_forecast,
                             name="PV Forecast (kW)",
                             line=dict(color="#ffb347", width=2)))
    fig.add_trace(go.Scatter(x=hours, y=load_forecast,
                             name="Load Forecast (kW)",
                             line=dict(color="#4a9eff", width=2)))
    fig.add_trace(go.Scatter(x=hours, y=price_forecast,
                             name="Price Forecast (₹/kWh)",
                             line=dict(color="#a78bfa", width=2)))
    fig.update_layout(**PLOT_CFG,
                      title="Next 24h Forecast (Probabilistic)",
                      xaxis_title="Hour of Day",
                      yaxis_title="kW / ₹/kWh",
                      height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🤖 AI Dispatch Recommendation *(Human-Gated)*")

    batt_r = get_row(battery_df, location)
    soc_now = sv(batt_r, "state_of_charge_soc_pct", 50) if batt_r is not None else 50

    if soc_now < 30:
        advice = "🔋 SOC is low ({:.1f}%). **Charge from grid now** during off-peak (00:00–06:00) to save cost.".format(soc_now)
    elif soc_now > 80:
        advice = "⚡ SOC is high ({:.1f}%). **Discharge now** during evening peak (18:00–22:00) to maximise savings.".format(soc_now)
    else:
        advice = "✅ SOC is optimal ({:.1f}%). **Maintain** current charge level. Discharge during peak tariff (12:00–22:00).".format(soc_now)

    st.info(advice)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Execute", type="primary"):
            st.success("Command approved and sent to EMS.")
    with col2:
        if st.button("❌ Reject"):
            st.warning("Command rejected.")

    st.markdown("---")
    st.markdown("### Estimated Savings")
    savings_df = pd.DataFrame({
        "Hour": hours,
        "Estimated Savings (₹)": np.clip(
            (price_forecast - 4.5) * np.abs(load_forecast) * 0.3, 0, None
        )
    })
    fig2 = go.Figure(go.Bar(
        x=savings_df["Hour"],
        y=savings_df["Estimated Savings (₹)"],
        marker_color="#00d4aa",
    ))
    fig2.update_layout(**PLOT_CFG,
                       title="Hourly Estimated Peak-Shaving Savings (₹)",
                       height=220)
    st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE REFRESH — must be at the very bottom after all content renders
# ═══════════════════════════════════════════════════════════════════════════════
if live_mode:
    import time
    time.sleep(refresh_rate)
    st.session_state.tick += 1
    st.rerun()
