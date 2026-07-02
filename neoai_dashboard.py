"""
NeoAI Smart EMS — Live Dashboard
==================================
Neosol Energy Systems Pvt Ltd
Run: streamlit run neoai_dashboard.py

For Streamlit Cloud: put datasets in data/ folder next to this file
For local:           set BASE path below to your Documents folder
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeoAI — Smart EMS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #080d14; }
  .block-container { padding: 1rem 1.5rem; }
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #060b14 100%);
    border-right: 1px solid #1a2744;
  }
  [data-testid="stSidebar"] * { color: #94aed4 !important; }
  .kpi-card {
    background: linear-gradient(135deg, #0d1f3c 0%, #091525 100%);
    border: 1px solid #1e3a5f; border-radius: 12px;
    padding: 18px 20px; text-align: center;
    position: relative; overflow: hidden; margin-bottom: 4px;
  }
  .kpi-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:2px;
    background: var(--accent, #00d4aa);
  }
  .kpi-label { font-size:10px; color:#5a7a9a; text-transform:uppercase; letter-spacing:1.5px; font-weight:500; margin-bottom:6px; }
  .kpi-value { font-family:'JetBrains Mono',monospace; font-size:26px; font-weight:600; margin:2px 0; }
  .kpi-unit  { font-size:12px; color:#5a7a9a; margin-left:3px; }
  .kpi-sub   { font-size:10px; color:#3d5a7a; margin-top:5px; }
  .green  { color:#00d4aa; } .red    { color:#ff4d6d; }
  .amber  { color:#ffb347; } .blue   { color:#4a9eff; }
  .purple { color:#a78bfa; } .white  { color:#e8f0fe; }
  .alarm-critical { background:#1a0510; border-left:3px solid #ff4d6d; padding:8px 14px; border-radius:6px; margin:4px 0; font-size:12px; color:#ff4d6d; font-family:'JetBrains Mono',monospace; }
  .alarm-warning  { background:#1a1000; border-left:3px solid #ffb347; padding:8px 14px; border-radius:6px; margin:4px 0; font-size:12px; color:#ffb347; font-family:'JetBrains Mono',monospace; }
  .alarm-normal   { background:#001a12; border-left:3px solid #00d4aa; padding:8px 14px; border-radius:6px; margin:4px 0; font-size:12px; color:#00d4aa; font-family:'JetBrains Mono',monospace; }
  .section-header { font-size:11px; color:#3d5a7a; text-transform:uppercase; letter-spacing:2px; font-weight:600; margin:16px 0 8px 0; border-bottom:1px solid #1a2744; padding-bottom:6px; }
  .live-badge { display:inline-block; background:#ff4d6d22; border:1px solid #ff4d6d55; color:#ff4d6d; font-size:10px; padding:2px 8px; border-radius:20px; font-weight:600; letter-spacing:1px; }
  #MainMenu, footer, header { visibility:hidden; }
  .stDeployButton { display:none; }
</style>
""", unsafe_allow_html=True)

# ── Plot base theme ───────────────────────────────────────────────────────────
PLOT_BG = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#080d14",
    font=dict(color="#94aed4", family="Inter", size=11),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(gridcolor="#1a2744", showline=False, tickfont=dict(size=9, color="#3d5a7a")),
    yaxis=dict(gridcolor="#1a2744", showline=False, tickfont=dict(size=9, color="#3d5a7a")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#94aed4")),
)

# ── Safe value getter ─────────────────────────────────────────────────────────
def sv(row, col, default=0):
    """Safely get value from a pandas Series row."""
    try:
        val = row[col]
        if pd.isna(val):
            return default
        return val
    except Exception:
        return default

# ── Data path — change BASE for local vs cloud ────────────────────────────────
# LOCAL:  BASE = Path("C:/Users/admin/Documents")
# CLOUD:  BASE = Path(__file__).parent / "data"
BASE = Path("C:/Users/admin/Documents")

# ── Load datasets ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading NeoAI datasets…")
def load_all():
    errors = []
    data = {}

    files = {
        "batt":  ("battery_neoai_dataset.csv",          "csv"),
        "pcs":   ("pcs_neoai_dataset.csv",              "csv"),
        "swgr":  ("switchgear_neoai_dataset.xlsx",      "xlsx"),
        "tline": ("transmission_line_neoai_dataset.xlsx","xlsx"),
    }

    for key, (fname, ftype) in files.items():
        path = BASE / fname
        try:
            if ftype == "csv":
                df = pd.read_csv(path, parse_dates=["timestamp"])
            else:
                df = pd.read_excel(path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            data[key] = df
        except FileNotFoundError:
            errors.append(f"❌ Missing: {fname}")
            data[key] = pd.DataFrame()
        except Exception as e:
            errors.append(f"❌ Error loading {fname}: {e}")
            data[key] = pd.DataFrame()

    return data, errors

data, load_errors = load_all()
batt  = data["batt"]
pcs   = data["pcs"]
swgr  = data["swgr"]
tline = data["tline"]

# Show load errors if any
if load_errors:
    for err in load_errors:
        st.error(err)

# ── Session state ─────────────────────────────────────────────────────────────
if "row_idx" not in st.session_state:
    st.session_state.row_idx = 0
if "location" not in st.session_state:
    st.session_state.location = "Bhandu_Rajasthan"

# ── Helper: get live window from a dataframe ──────────────────────────────────
def get_window(df, n=48):
    if df.empty:
        return pd.DataFrame(), pd.Series(dtype=object)
    loc_df = df[df["location"] == st.session_state.location].reset_index(drop=True)
    if loc_df.empty:
        return pd.DataFrame(), pd.Series(dtype=object)
    idx   = st.session_state.row_idx % len(loc_df)
    start = max(0, idx - n)
    return loc_df.iloc[start:idx+1], loc_df.iloc[idx]

# ── KPI card helper ───────────────────────────────────────────────────────────
def kpi(label, value, unit, color, sub, accent="#00d4aa"):
    color_map = {
        "green":"#00d4aa","red":"#ff4d6d",
        "amber":"#ffb347","blue":"#4a9eff",
        "purple":"#a78bfa","white":"#e8f0fe"
    }
    hex_accent = color_map.get(color, accent)
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{hex_accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value {color}">{value}<span class="kpi-unit">{unit}</span></div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ NeoAI")
    st.markdown("**Smart Energy Management System**")
    st.markdown("*Neosol Energy Systems Pvt Ltd*")
    st.markdown("---")

    page = st.radio("Navigation", [
        "🏠  Overview",
        "🔋  Battery",
        "⚡  PCS",
        "🔧  Switchgear",
        "📡  Transmission Line",
        "🚨  Alarms",
        "📊  Reports",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p class="section-header">Site Settings</p>', unsafe_allow_html=True)

    loc_choice = st.selectbox(
        "Location",
        ["Bhandu_Rajasthan", "Prakasha_Nandurbar"],
        label_visibility="visible"
    )
    st.session_state.location = loc_choice

    refresh_sec = st.slider("Refresh (seconds)", 3, 60, 10)
    live_on     = st.toggle("Live Mode", value=False)   # OFF by default — safer

    st.markdown("---")

    # Quick sidebar status
    batt_w, batt_now = get_window(batt)
    if not batt_now.empty:
        is_f   = int(sv(batt_now, "is_fault", 0))
        mode   = str(sv(batt_now, "charge_discharge_mode", "idle"))
        ts_val = batt_now.get("timestamp", None)
        ts_str = pd.Timestamp(ts_val).strftime("%H:%M:%S") if ts_val is not None else "—"
        sc     = "green" if is_f == 0 else "red"
        st.markdown(f'<span class="live-badge">● LIVE</span>', unsafe_allow_html=True)
        st.markdown(f"**Status:** <span class='{sc}'>{'Normal' if is_f==0 else 'FAULT'}</span>", unsafe_allow_html=True)
        st.markdown(f"**Mode:** {mode.capitalize()}")
        st.markdown(f"**Time:** {ts_str}")
    else:
        st.warning("No data loaded")

# ── Get all live windows ──────────────────────────────────────────────────────
batt_w,  batt_now  = get_window(batt)
pcs_w,   pcs_now   = get_window(pcs)
swgr_w,  swgr_now  = get_window(swgr)
tline_w, tline_now = get_window(tline)

# ── Page title bar ────────────────────────────────────────────────────────────
PAGE_META = {
    "🏠  Overview":          ("Overview",           "#00d4aa"),
    "🔋  Battery":           ("Battery",            "#00d4aa"),
    "⚡  PCS":               ("Power Conversion",   "#4a9eff"),
    "🔧  Switchgear":        ("Switchgear",         "#a78bfa"),
    "📡  Transmission Line": ("Transmission Line",  "#ff6b9d"),
    "🚨  Alarms":            ("Alarms & Faults",    "#ff4d6d"),
    "📊  Reports":           ("Reports",            "#00d4aa"),
}
title, accent = PAGE_META[page]
ts_display = ""
if not batt_now.empty:
    try:
        ts_display = pd.Timestamp(batt_now["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        pass

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:16px;border-bottom:1px solid #1a2744;padding-bottom:10px">
  <div>
    <span style="font-size:20px;font-weight:700;color:{accent}">{title}</span>
    <span style="font-size:11px;color:#3d5a7a;margin-left:12px">NeoAI Smart EMS</span>
  </div>
  <div style="display:flex;gap:12px;align-items:center">
    <span class="live-badge">● LIVE</span>
    <span style="font-size:11px;color:#3d5a7a;font-family:'JetBrains Mono',monospace">{loc_choice}</span>
    <span style="font-size:11px;color:#3d5a7a;font-family:'JetBrains Mono',monospace">{ts_display}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    if batt_now.empty:
        st.warning("Battery dataset not loaded. Check file path.")
    else:
        soc   = float(sv(batt_now, "state_of_charge_soc_pct",  0))
        soh   = float(sv(batt_now, "state_of_health_soh_pct",  0))
        power = float(sv(batt_now, "battery_power_kw",          0))
        temp  = float(sv(batt_now, "average_cell_temperature_c",0))
        freq  = float(sv(pcs_now,  "grid_frequency_hz",        50)) if not pcs_now.empty else 50.0
        pf    = float(sv(pcs_now,  "power_factor_overall",      1)) if not pcs_now.empty else 1.0
        sav   = float(sv(pcs_now,  "savings_inr",               0)) if not pcs_now.empty else 0.0
        co2   = float(sv(pcs_now,  "co2_avoided_kg",            0)) if not pcs_now.empty else 0.0

        soc_c = "green" if soc > 50 else ("amber" if soc > 20 else "red")
        pw_c  = "blue"  if power > 0 else ("green" if power < 0 else "white")

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: kpi("State of Charge", f"{soc:.1f}",   "%",  soc_c, f"SOH {soh:.1f}%")
        with c2: kpi("Battery Power",   f"{power:.1f}", "kW", pw_c,  "Charging" if power>0 else "Discharging")
        with c3: kpi("Cell Temp",       f"{temp:.1f}",  "°C", "amber" if temp>38 else "green", "Avg all cells")
        with c4: kpi("Grid Frequency",  f"{freq:.3f}",  "Hz", "green" if abs(freq-50)<0.1 else "red", "Nominal 50 Hz")
        with c5: kpi("Power Factor",    f"{pf:.3f}",    "",   "green" if pf>0.95 else "amber", "Target >0.95")
        with c6: kpi("CO₂ Avoided",    f"{co2:.3f}",   "kg", "green", "This reading")

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)
        with col_l:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=batt_w["timestamp"], y=batt_w["state_of_charge_soc_pct"],
                fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
                line=dict(color="#00d4aa", width=2), name="SOC %"
            ))
            fig.update_layout(**PLOT_BG, title="State of Charge (%)", height=220)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            pw_vals = batt_w["battery_power_kw"]
            fig = go.Figure(go.Bar(
                x=batt_w["timestamp"], y=pw_vals,
                marker_color=["#00d4aa" if v>=0 else "#ff4d6d" for v in pw_vals],
                name="kW"
            ))
            fig.update_layout(**PLOT_BG, title="Battery Power — Charge(+) / Discharge(−) kW", height=220)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown('<p class="section-header">System Health — All Components</p>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        comps = [
            (c1, "🔋 Battery",      batt_now,  "#00d4aa"),
            (c2, "⚡ PCS",          pcs_now,   "#4a9eff"),
            (c3, "🔧 Switchgear",   swgr_now,  "#a78bfa"),
            (c4, "📡 Trans. Line",  tline_now, "#ff6b9d"),
        ]
        for col, name, now_row, ac in comps:
            with col:
                fault = int(sv(now_row, "is_fault", 0)) if not (now_row is None or (hasattr(now_row,"empty") and now_row.empty)) else -1
                status = "NO DATA" if fault==-1 else ("FAULT" if fault else "NORMAL")
                sc = "amber" if fault==-1 else ("red" if fault else "green")
                kpi(name, status, "", sc, "", accent=ac)

# ═══════════════════════════════════════════════════════════════════════════════
# BATTERY PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔋  Battery":
    if batt_now.empty:
        st.warning("Battery dataset not loaded.")
    else:
        soc  = float(sv(batt_now, "state_of_charge_soc_pct",    0))
        soh  = float(sv(batt_now, "state_of_health_soh_pct",    0))
        temp = float(sv(batt_now, "average_cell_temperature_c", 0))
        ir   = float(sv(batt_now, "internal_resistance_mohm",   0))
        rul  = float(sv(batt_now, "remaining_useful_life_years", 0))
        cyc  = float(sv(batt_now, "cycle_count",                 0))
        pwr  = float(sv(batt_now, "battery_power_kw",            0))
        volt = float(sv(batt_now, "pack_voltage_v",              0))
        therm= int(sv(batt_now,   "thermal_runaway_warning_status", 0))

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("SOC",         f"{soc:.1f}",  "%",   "green" if soc>50 else "amber",  "State of Charge")
        with c2: kpi("SOH",         f"{soh:.2f}",  "%",   "green" if soh>85 else "amber",  "State of Health")
        with c3: kpi("Cell Temp",   f"{temp:.1f}", "°C",  "amber" if temp>38 else "green", "Avg temperature")
        with c4: kpi("RUL",         f"{rul:.1f}",  "yrs", "green" if rul>3 else "red",     "Remaining Useful Life")

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Pack Voltage",   f"{volt:.1f}",  "V",  "white",                             "Full pack voltage")
        with c2: kpi("Battery Power",  f"{pwr:.2f}",   "kW", "blue" if pwr>0 else "green",        "Live power")
        with c3: kpi("Internal Res.",  f"{ir:.3f}",    "mΩ", "green" if ir<2.5 else "amber",      "Aging indicator")
        with c4: kpi("Cycle Count",    f"{cyc:.0f}",   "",   "green" if cyc<3000 else "amber",    "Cumulative cycles")

        if therm:
            st.error("🔴 THERMAL RUNAWAY WARNING ACTIVE — Check battery immediately!")

        st.markdown("<br>", unsafe_allow_html=True)
        c_gauge, c_soc, c_temp = st.columns([1, 2, 2])

        with c_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=soc,
                number={"suffix":"%","font":{"size":28,"color":"#00d4aa"}},
                title={"text":"SOC","font":{"size":12,"color":"#5a7a9a"}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#1a2744","tickfont":{"color":"#3d5a7a"}},
                    "bar":{"color":"#00d4aa"}, "bgcolor":"#0d1f3c","bordercolor":"#1e3a5f",
                    "steps":[{"range":[0,20],"color":"#2d0a0a"},{"range":[20,50],"color":"#1a1000"},{"range":[50,100],"color":"#001a12"}],
                    "threshold":{"line":{"color":"#ff4d6d","width":3},"thickness":0.75,"value":20}
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=230, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c_soc:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=batt_w["timestamp"], y=batt_w["state_of_charge_soc_pct"],
                                     fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
                                     line=dict(color="#00d4aa",width=2), name="SOC %"))
            fig.add_trace(go.Scatter(x=batt_w["timestamp"], y=batt_w["state_of_health_soh_pct"],
                                     line=dict(color="#4a9eff",width=1.5,dash="dot"), name="SOH %"))
            fig.update_layout(**PLOT_BG, title="SOC & SOH Trend", height=230)
            st.plotly_chart(fig, use_container_width=True)

        with c_temp:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=batt_w["timestamp"], y=batt_w["max_cell_temperature_c"],
                                     line=dict(color="#ff4d6d",width=1.5), name="Max"))
            fig.add_trace(go.Scatter(x=batt_w["timestamp"], y=batt_w["average_cell_temperature_c"],
                                     line=dict(color="#ffb347",width=2), name="Avg"))
            fig.add_trace(go.Scatter(x=batt_w["timestamp"], y=batt_w["min_cell_temperature_c"],
                                     line=dict(color="#00d4aa",width=1.5), name="Min"))
            fig.update_layout(**PLOT_BG, title="Cell Temperature (°C)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        c_ir, c_power = st.columns(2)
        with c_ir:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=batt_w["timestamp"], y=batt_w["internal_resistance_mohm"],
                                     line=dict(color="#a78bfa",width=2), name="IR (mΩ)"))
            fig.update_layout(**PLOT_BG, title="Internal Resistance (mΩ) — Aging Indicator", height=200)
            st.plotly_chart(fig, use_container_width=True)

        with c_power:
            pw = batt_w["battery_power_kw"]
            fig = go.Figure(go.Bar(x=batt_w["timestamp"], y=pw,
                                   marker_color=["#00d4aa" if v>=0 else "#ff4d6d" for v in pw]))
            fig.update_layout(**PLOT_BG, title="Battery Power Charge(+)/Discharge(−) kW", height=200)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PCS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚡  PCS":
    if pcs_now.empty:
        st.warning("PCS dataset not loaded.")
    else:
        eff  = float(sv(pcs_now, "conversion_efficiency_pct", 0))
        igbt = float(sv(pcs_now, "igbt_temperature_c",         0))
        pf   = float(sv(pcs_now, "power_factor_overall",       1))
        freq = float(sv(pcs_now, "grid_frequency_hz",         50))
        thd  = float(sv(pcs_now, "voltage_thd_pct",            0))
        mode = str(sv(pcs_now,   "operating_mode",        "standby"))
        pwr  = float(sv(pcs_now, "active_power_kw",            0))
        rte  = float(sv(pcs_now, "roundtrip_efficiency_pct",   0))

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Conversion Eff.", f"{eff:.2f}",  "%",  "green" if eff>95 else "amber", "PCS efficiency",    "#4a9eff")
        with c2: kpi("IGBT Temp",       f"{igbt:.1f}", "°C", "amber" if igbt>55 else "green","Main switch chip",  "#4a9eff")
        with c3: kpi("Power Factor",    f"{pf:.4f}",   "",   "green" if pf>0.95 else "amber","Overall PF",        "#4a9eff")
        with c4: kpi("Grid Frequency",  f"{freq:.3f}", "Hz", "green" if abs(freq-50)<0.1 else "red","Nominal 50 Hz","#4a9eff")

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Active Power",   f"{pwr:.2f}",  "kW", "blue",                          str(mode).capitalize(), "#4a9eff")
        with c2: kpi("Voltage THD",    f"{thd:.2f}",  "%",  "green" if thd<3 else "amber",   "Harmonic distortion",  "#4a9eff")
        with c3: kpi("Round-Trip Eff.",f"{rte:.2f}",  "%",  "green" if rte>92 else "amber",  "Total efficiency",     "#4a9eff")
        with c4: kpi("Mode",           mode.upper(),  "",   "blue" if mode=="charging" else ("green" if mode=="discharging" else "white"),"Operating mode","#4a9eff")

        st.markdown("<br>", unsafe_allow_html=True)
        c_eff, c_thermal, c_grid = st.columns(3)

        with c_eff:
            fig = go.Figure()
            eff_s = pcs_w["conversion_efficiency_pct"].replace(0, None)
            fig.add_trace(go.Scatter(x=pcs_w["timestamp"], y=eff_s,
                                     line=dict(color="#4a9eff",width=2), name="Eff %"))
            fig.update_layout(**PLOT_BG, title="Conversion Efficiency (%)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        with c_thermal:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pcs_w["timestamp"], y=pcs_w["igbt_temperature_c"],
                                     line=dict(color="#ff4d6d",width=2), name="IGBT"))
            fig.add_trace(go.Scatter(x=pcs_w["timestamp"], y=pcs_w["heatsink_temperature_c"],
                                     line=dict(color="#ffb347",width=1.5), name="Heatsink"))
            fig.add_trace(go.Scatter(x=pcs_w["timestamp"], y=pcs_w["capacitor_temperature_c"],
                                     line=dict(color="#a78bfa",width=1.5), name="Capacitor"))
            fig.update_layout(**PLOT_BG, title="Thermal Monitoring (°C)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        with c_grid:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pcs_w["timestamp"], y=pcs_w["grid_frequency_hz"],
                                     line=dict(color="#00d4aa",width=2), name="Freq Hz"))
            fig.add_hline(y=50.0, line_dash="dot", line_color="#3d5a7a", annotation_text="50 Hz")
            fig.update_layout(**PLOT_BG, title="Grid Frequency (Hz)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        c_phase, c_pf = st.columns(2)
        with c_phase:
            fig = go.Figure()
            for col_name, color, lbl in [
                ("ac_voltage_phase_r_v","#ff4d6d","Phase R"),
                ("ac_voltage_phase_y_v","#ffb347","Phase Y"),
                ("ac_voltage_phase_b_v","#4a9eff","Phase B"),
            ]:
                fig.add_trace(go.Scatter(x=pcs_w["timestamp"], y=pcs_w[col_name],
                                         line=dict(color=color,width=1.5), name=lbl))
            fig.update_layout(**PLOT_BG, title="3-Phase Voltage (V)", height=200)
            st.plotly_chart(fig, use_container_width=True)

        with c_pf:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pcs_w["timestamp"], y=pcs_w["power_factor_overall"],
                                     fill="tozeroy", fillcolor="rgba(0,212,170,0.06)",
                                     line=dict(color="#00d4aa",width=2), name="PF"))
            fig.add_hline(y=0.95, line_dash="dot", line_color="#ffb347", annotation_text="Min 0.95")
            fig.update_layout(**PLOT_BG, title="Power Factor", height=200,
                              yaxis=dict(range=[0.8,1.0], gridcolor="#1a2744"))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SWITCHGEAR PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔧  Switchgear":
    if swgr_now.empty:
        st.warning("Switchgear dataset not loaded.")
    else:
        brk    = str(sv(swgr_now, "main_breaker_position",         "unknown"))
        sf6    = float(sv(swgr_now,"sf6_gas_pressure_bar",          0))
        busR   = float(sv(swgr_now,"busbar_temperature_phase_r_c",  0))
        busY   = float(sv(swgr_now,"busbar_temperature_phase_y_c",  0))
        busB   = float(sv(swgr_now,"busbar_temperature_phase_b_c",  0))
        wear   = float(sv(swgr_now,"contact_wear_index_pct",        0))
        pd_lvl = float(sv(swgr_now,"partial_discharge_level_pc",    0))
        freq   = float(sv(swgr_now,"grid_frequency_hz",            50))
        rocof  = float(sv(swgr_now,"rocof_hz_per_s",                0))
        ins_r  = float(sv(swgr_now,"insulation_resistance_mohm",    0))
        pf     = float(sv(swgr_now,"power_factor",                  1))
        pwr    = float(sv(swgr_now,"active_power_kw",               0))

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Breaker Position",  brk.upper(),     "", "green" if brk.lower()=="closed" else "red",    "Main breaker",      "#a78bfa")
        with c2: kpi("SF6 Gas Pressure",  f"{sf6:.2f}",    "bar","green" if sf6>5.0 else "red",              "Gas insulation",    "#a78bfa")
        with c3: kpi("Contact Wear",      f"{wear:.1f}",   "%", "amber" if wear>60 else "green",             "Breaker contacts",  "#a78bfa")
        with c4: kpi("Partial Discharge", f"{pd_lvl:.0f}", "pC","amber" if pd_lvl>200 else "green",          "Insulation health", "#a78bfa")

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Active Power",  f"{pwr:.2f}",   "kW",    "blue",                                    "Through switchgear","#a78bfa")
        with c2: kpi("Power Factor",  f"{pf:.4f}",    "",      "green" if pf>0.95 else "amber",           "Switchgear PF",     "#a78bfa")
        with c3: kpi("Grid Freq",     f"{freq:.3f}",  "Hz",    "green" if abs(freq-50)<0.1 else "red",    "Nominal 50 Hz",     "#a78bfa")
        with c4: kpi("ROCOF",         f"{rocof:.4f}", "Hz/s",  "red" if abs(rocof)>0.5 else "green",      "Rate of freq change","#a78bfa")

        st.markdown("<br>", unsafe_allow_html=True)
        c_bus, c_power, c_prot = st.columns(3)

        with c_bus:
            fig = go.Figure()
            for col_name, color, lbl in [
                ("busbar_temperature_phase_r_c","#ff4d6d","Phase R"),
                ("busbar_temperature_phase_y_c","#ffb347","Phase Y"),
                ("busbar_temperature_phase_b_c","#4a9eff","Phase B"),
            ]:
                fig.add_trace(go.Scatter(x=swgr_w["timestamp"], y=swgr_w[col_name],
                                         line=dict(color=color,width=1.5), name=lbl))
            fig.update_layout(**PLOT_BG, title="Busbar Temperature (°C)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        with c_power:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=swgr_w["timestamp"], y=swgr_w["active_power_kw"],
                                     line=dict(color="#a78bfa",width=2), name="Active kW"))
            fig.add_trace(go.Scatter(x=swgr_w["timestamp"], y=swgr_w["reactive_power_kvar"],
                                     line=dict(color="#4a9eff",width=1.5,dash="dot"), name="Reactive kVAR"))
            fig.update_layout(**PLOT_BG, title="Active & Reactive Power", height=230)
            st.plotly_chart(fig, use_container_width=True)

        with c_prot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=swgr_w["timestamp"], y=swgr_w["partial_discharge_level_pc"],
                                     line=dict(color="#ff4d6d",width=2), name="PD (pC)"))
            fig.add_hline(y=200, line_dash="dot", line_color="#ffb347", annotation_text="Warning 200 pC")
            fig.update_layout(**PLOT_BG, title="Partial Discharge Level (pC)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        c_ins, c_vol = st.columns(2)
        with c_ins:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=swgr_w["timestamp"], y=swgr_w["insulation_resistance_mohm"],
                                     line=dict(color="#00d4aa",width=2), name="IR (MΩ)"))
            fig.update_layout(**PLOT_BG, title="Insulation Resistance (MΩ)", height=200)
            st.plotly_chart(fig, use_container_width=True)

        with c_vol:
            fig = go.Figure()
            for col_name, color, lbl in [
                ("voltage_phase_a_v","#ff4d6d","Phase A"),
                ("voltage_phase_b_v","#ffb347","Phase B"),
                ("voltage_phase_c_v","#4a9eff","Phase C"),
            ]:
                fig.add_trace(go.Scatter(x=swgr_w["timestamp"], y=swgr_w[col_name],
                                         line=dict(color=color,width=1.5), name=lbl))
            fig.update_layout(**PLOT_BG, title="3-Phase Voltage (V)", height=200)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TRANSMISSION LINE PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📡  Transmission Line":
    if tline_now.empty:
        st.warning("Transmission Line dataset not loaded.")
    else:
        loading = float(sv(tline_now,"line_loading_pct",               0))
        loss    = float(sv(tline_now,"line_transmission_loss_kw",       0))
        cond_t  = float(sv(tline_now,"line_conductor_temperature_c",    0))
        freq    = float(sv(tline_now,"grid_frequency_receiving_end_hz",50))
        vdrop   = float(sv(tline_now,"voltage_drop_across_line_v",      0))
        pf      = float(sv(tline_now,"power_factor_at_pcc",             1))
        rocof   = float(sv(tline_now,"rocof_hz_per_s",                  0))
        sag     = float(sv(tline_now,"conductor_sag_m",                 0))

        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Line Loading",      f"{loading:.1f}","%" , "amber" if loading>80 else "green","% of rated",       "#ff6b9d")
        with c2: kpi("Transmission Loss", f"{loss:.2f}",   "kW","amber" if loss>5 else "green",    "Power lost",        "#ff6b9d")
        with c3: kpi("Conductor Temp",    f"{cond_t:.1f}", "°C","amber" if cond_t>70 else "green", "Line conductor",    "#ff6b9d")
        with c4: kpi("Voltage Drop",      f"{vdrop:.1f}",  "V", "amber" if abs(vdrop)>20 else "green","Send − Receive","#ff6b9d")

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        with c1: kpi("Grid Frequency", f"{freq:.3f}",  "Hz",   "green" if abs(freq-50)<0.1 else "red","Receiving end", "#ff6b9d")
        with c2: kpi("Power Factor",   f"{pf:.4f}",    "",     "green" if pf>0.95 else "amber",       "At PCC",        "#ff6b9d")
        with c3: kpi("ROCOF",          f"{rocof:.4f}", "Hz/s", "red" if abs(rocof)>0.5 else "green",  "Freq rate",     "#ff6b9d")
        with c4: kpi("Conductor Sag",  f"{sag:.2f}",   "m",   "amber" if sag>10 else "green",        "Overhead sag",  "#ff6b9d")

        st.markdown("<br>", unsafe_allow_html=True)
        c_load, c_loss, c_freq = st.columns(3)

        with c_load:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tline_w["timestamp"], y=tline_w["line_loading_pct"],
                                     fill="tozeroy", fillcolor="rgba(255,107,157,0.08)",
                                     line=dict(color="#ff6b9d",width=2), name="Loading %"))
            fig.add_hline(y=80, line_dash="dot", line_color="#ffb347", annotation_text="80% limit")
            fig.update_layout(**PLOT_BG, title="Line Loading (%)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        with c_loss:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tline_w["timestamp"], y=tline_w["line_transmission_loss_kw"],
                                     fill="tozeroy", fillcolor="rgba(255,77,109,0.06)",
                                     line=dict(color="#ff4d6d",width=2), name="Loss kW"))
            fig.update_layout(**PLOT_BG, title="Transmission Loss (kW)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        with c_freq:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tline_w["timestamp"], y=tline_w["grid_frequency_receiving_end_hz"],
                                     line=dict(color="#00d4aa",width=2), name="Freq Hz"))
            fig.add_hline(y=50.0, line_dash="dot", line_color="#3d5a7a")
            fig.update_layout(**PLOT_BG, title="Receiving End Frequency (Hz)", height=230)
            st.plotly_chart(fig, use_container_width=True)

        c_v, c_cur = st.columns(2)
        with c_v:
            fig = go.Figure()
            for col_n, color, lbl in [
                ("sending_end_voltage_phase_a_v","#ff4d6d","Send A"),
                ("sending_end_voltage_phase_b_v","#ffb347","Send B"),
                ("sending_end_voltage_phase_c_v","#4a9eff","Send C"),
            ]:
                fig.add_trace(go.Scatter(x=tline_w["timestamp"], y=tline_w[col_n],
                                         line=dict(color=color,width=1.5), name=lbl))
            fig.update_layout(**PLOT_BG, title="Sending End Voltage (V)", height=200)
            st.plotly_chart(fig, use_container_width=True)

        with c_cur:
            fig = go.Figure()
            for col_n, color, lbl in [
                ("line_current_phase_a_a","#ff4d6d","Phase A"),
                ("line_current_phase_b_a","#ffb347","Phase B"),
                ("line_current_phase_c_a","#4a9eff","Phase C"),
            ]:
                fig.add_trace(go.Scatter(x=tline_w["timestamp"], y=tline_w[col_n],
                                         line=dict(color=color,width=1.5), name=lbl))
            fig.update_layout(**PLOT_BG, title="Line Current (A)", height=200)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ALARMS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨  Alarms":
    st.markdown('<p class="section-header">Active System Alarms — All Components</p>', unsafe_allow_html=True)

    components = [
        ("Battery",        batt_now),
        ("PCS",            pcs_now),
        ("Switchgear",     swgr_now),
        ("Trans. Line",    tline_now),
    ]

    any_fault = False
    for comp_name, now_row in components:
        if now_row is None or (hasattr(now_row,"empty") and now_row.empty):
            st.markdown(f'<div class="alarm-warning">⚠  [{comp_name}]  No data loaded</div>', unsafe_allow_html=True)
            continue
        ft   = str(sv(now_row, "fault_type",     "normal"))
        sev  = str(sv(now_row, "fault_severity", "none"))
        is_f = int(sv(now_row, "is_fault",        0))
        is_c = int(sv(now_row, "is_critical",     0))

        if is_c:
            icon = "🔴"; cls = "alarm-critical"; any_fault = True
        elif is_f:
            icon = "🟡"; cls = "alarm-warning";  any_fault = True
        else:
            icon = "🟢"; cls = "alarm-normal"

        msg = f"{icon}  [{comp_name}]  {ft.upper().replace('_',' ')}  —  Severity: {sev.upper()}"
        st.markdown(f'<div class="{cls}">{msg}</div>', unsafe_allow_html=True)

    if not any_fault:
        st.success("✅ All systems normal — no active faults across any component.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Fault Event Timeline — Last 48 Hours</p>', unsafe_allow_html=True)

    fault_dfs = []
    for comp_name, df_w in [("Battery",batt_w),("PCS",pcs_w),("Switchgear",swgr_w),("Trans. Line",tline_w)]:
        if df_w is not None and not df_w.empty and "is_fault" in df_w.columns:
            faults = df_w[df_w["is_fault"]==1][["timestamp","fault_type","fault_severity"]].copy()
            faults["component"] = comp_name
            fault_dfs.append(faults)

    if fault_dfs:
        all_faults = pd.concat(fault_dfs).sort_values("timestamp", ascending=False)
        if len(all_faults) > 0:
            fig = px.scatter(
                all_faults, x="timestamp", y="component",
                color="fault_severity",
                color_discrete_map={"critical":"#ff4d6d","high":"#ff8c00","medium":"#ffb347","low":"#00d4aa","none":"#3d5a7a"},
                symbol="fault_type", height=300,
                title="Fault Event Timeline"
            )
            fig.update_layout(**PLOT_BG)
            fig.update_traces(marker=dict(size=10))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<p class="section-header">Recent Fault Events</p>', unsafe_allow_html=True)
            st.dataframe(
                all_faults.head(20).reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.markdown('<div class="alarm-normal">🟢  No fault events in last 48 hours</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# REPORTS PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Reports":
    total_sav = float(pcs_w["savings_inr"].sum())     if (not pcs_w.empty and "savings_inr" in pcs_w.columns)  else 0
    total_co2 = float(pcs_w["co2_avoided_kg"].sum())  if (not pcs_w.empty and "co2_avoided_kg" in pcs_w.columns) else 0
    avg_soc   = float(batt_w["state_of_charge_soc_pct"].mean()) if not batt_w.empty else 0
    avg_soh   = float(batt_w["state_of_health_soh_pct"].mean()) if not batt_w.empty else 0
    avg_eff   = float(pcs_w["conversion_efficiency_pct"].replace(0,None).mean()) if not pcs_w.empty else 0
    fault_rate_batt = float(batt_w["is_fault"].mean()*100) if (not batt_w.empty and "is_fault" in batt_w.columns) else 0
    fault_rate_pcs  = float(pcs_w["is_fault"].mean()*100)  if (not pcs_w.empty  and "is_fault" in pcs_w.columns)  else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Savings",  f"₹{total_sav:,.0f}","",   "green", "Cumulative")
    with c2: kpi("CO₂ Avoided",   f"{total_co2:.2f}",  "kg", "green", "Cumulative")
    with c3: kpi("Avg SOC",        f"{avg_soc:.1f}",    "%",  "blue",  "Current window")
    with c4: kpi("Avg PCS Eff.",   f"{avg_eff:.2f}",    "%",  "blue",  "Current window")

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1: kpi("Avg SOH",         f"{avg_soh:.2f}",       "%", "green" if avg_soh>85 else "amber","Battery health")
    with c2: kpi("Battery Fault %", f"{fault_rate_batt:.2f}","%","green" if fault_rate_batt<5 else "red","Last window")
    with c3: kpi("PCS Fault %",     f"{fault_rate_pcs:.2f}", "%","green" if fault_rate_pcs<5 else "red","Last window")

    st.markdown("<br>", unsafe_allow_html=True)

    if not batt_w.empty:
        c_soc, c_soh = st.columns(2)
        with c_soc:
            fig = px.line(batt_w, x="timestamp", y="state_of_charge_soc_pct",
                          title="SOC Trend", color_discrete_sequence=["#00d4aa"])
            fig.update_layout(**PLOT_BG, height=250)
            st.plotly_chart(fig, use_container_width=True)

        with c_soh:
            fig = px.line(batt_w, x="timestamp", y="state_of_health_soh_pct",
                          title="SOH Degradation Trend", color_discrete_sequence=["#4a9eff"])
            fig.update_layout(**PLOT_BG, height=250)
            st.plotly_chart(fig, use_container_width=True)

    if not pcs_w.empty and "savings_inr" in pcs_w.columns:
        fig = go.Figure(go.Bar(
            x=pcs_w["timestamp"], y=pcs_w["savings_inr"],
            marker_color="#00d4aa", name="Savings ₹"
        ))
        fig.update_layout(**PLOT_BG, title="Peak Shaving Savings (₹)", height=220)
        st.plotly_chart(fig, use_container_width=True)

    # Fault distribution chart
    st.markdown('<p class="section-header">Fault Type Distribution</p>', unsafe_allow_html=True)
    c_b, c_p = st.columns(2)
    for col_w, df_w, title_str, color in [
        (c_b, batt_w, "Battery Fault Distribution", "#00d4aa"),
        (c_p, pcs_w,  "PCS Fault Distribution",     "#4a9eff"),
    ]:
        with col_w:
            if not df_w.empty and "fault_type" in df_w.columns:
                fc = df_w["fault_type"].value_counts().reset_index()
                fc.columns = ["fault_type","count"]
                fig = px.bar(fc, x="fault_type", y="count",
                             title=title_str, color_discrete_sequence=[color])
                fig.update_layout(**PLOT_BG, height=220)
                st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LIVE REFRESH — must be at VERY BOTTOM after all page content
# ═══════════════════════════════════════════════════════════════════════════════
if live_on:
    time.sleep(refresh_sec)
    st.session_state.row_idx += 1
    st.rerun()
