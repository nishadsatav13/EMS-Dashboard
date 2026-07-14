"""
NeoAI — Enterprise EMS Dashboard v13.0
Simulator Embedded, All Components, Charts + Forecast & AI Advisory
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="NeoAI EMS", page_icon="⚡", layout="wide")

# ── Load datasets from repo ───────────────────────────────────
battery_df     = pd.read_csv("battery_neoai_dataset.csv")
pcs_df         = pd.read_csv("pcs_neoai_dataset.csv")
xfmr_df        = pd.read_csv("transformer_neoai_dataset.csv")
swgr_df        = pd.read_excel("switchgear_neoai_dataset.xlsx")
tline_df       = pd.read_excel("transmission_line_neoai_dataset.xlsx")

# ── Ensure timestamp column exists ────────────────────────────
for df in [battery_df, pcs_df, xfmr_df, swgr_df, tline_df]:
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range("2026-07-14", periods=len(df), freq="5s")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#00FFAA;'>NeoAI EMS</h2>", unsafe_allow_html=True)
    location = st.selectbox("Select Plant Location", ["Prakasha_Nandurbar", "Bhandu_Rajasthan"])
    page = st.radio("Select Component View", [
        "🏠 Master Overview",
        "🔋 Battery Storage (LFP)",
        "⚡ Power Conversion (PCS)",
        "🔌 Distribution Transformer",
        "🔧 Main Switchgear Panel",
        "📡 Transmission & Grid Line",
        "🚨 Alarms & Faults",
        "🔮 Forecast & AI Advisory"
    ])
    refresh_rate = st.slider("Telemetry Refresh (sec)", 2, 10, 5)

# ── Simulator Logic ──────────────────────────────────────────
if "tick" not in st.session_state:
    st.session_state.tick = 0
st.session_state.tick += 1

def get_row(df, location):
    if "location" in df.columns:
        loc_df = df[df["location"] == location].reset_index(drop=True)
    else:
        loc_df = df.reset_index(drop=True)
    if len(loc_df) == 0:
        return None
    idx = st.session_state.tick % len(loc_df)
    return loc_df.iloc[idx]

def plot_timeseries(df, x_col, y_col, title, location):
    if "location" in df.columns:
        df = df[df["location"] == location]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="lines+markers", line=dict(color="#00FFAA")))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title=y_col, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ── Pages ────────────────────────────────────────────────────
if page == "🏠 Master Overview":
    st.markdown("### 🌍 Master System Overview")
    batt_row = get_row(battery_df, location)
    pcs_row  = get_row(pcs_df, location)
    xfmr_row = get_row(xfmr_df, location)
    cols = st.columns(3)
    if batt_row is not None:
        cols[0].metric("SOC (%)", f"{batt_row['state_of_charge_soc_pct']:.1f}")
        cols[0].metric("SOH (%)", f"{batt_row['state_of_health_soh_pct']:.1f}")
    if pcs_row is not None:
        cols[1].metric("PCS Output (kW)", f"{pcs_row['active_power_kw']:.1f}")
    if xfmr_row is not None:
        cols[2].metric("Transformer Load (%)", f"{xfmr_row['load_pct']:.1f}")

if page == "🔋 Battery Storage (LFP)":
    row = get_row(battery_df, location)
    if row is not None:
        st.metric("SOC (%)", f"{row['state_of_charge_soc_pct']:.1f}")
        st.metric("SOH (%)", f"{row['state_of_health_soh_pct']:.1f}")
        st.metric("Power (kW)", f"{row['battery_power_kw']:.1f}")
        st.metric("Avg Cell Temp (°C)", f"{row['average_cell_temperature_c']:.1f}")
        plot_timeseries(battery_df, "timestamp", "state_of_charge_soc_pct", "Battery SOC Trend", location)

if page == "⚡ Power Conversion (PCS)":
    row = get_row(pcs_df, location)
    if row is not None:
        st.metric("Active Power (kW)", f"{row['active_power_kw']:.1f}")
        st.metric("Reactive Power (kVAR)", f"{row['reactive_power_kvar']:.1f}")
        st.metric("Frequency (Hz)", f"{row['frequency_hz']:.2f}")
        plot_timeseries(pcs_df, "timestamp", "active_power_kw", "PCS Active Power Trend", location)

if page == "🔌 Distribution Transformer":
    row = get_row(xfmr_df, location)
    if row is not None:
        st.metric("Load (%)", f"{row['load_pct']:.1f}")
        st.metric("Oil Temp (°C)", f"{row['oil_temperature_c']:.1f}")
        st.metric("Winding Temp (°C)", f"{row['winding_temperature_c']:.1f}")
        plot_timeseries(xfmr_df, "timestamp", "load_pct", "Transformer Load Trend", location)

if page == "🔧 Main Switchgear Panel":
    row = get_row(swgr_df, location)
    if row is not None:
        st.metric("Breaker Status", row['breaker_status'])
        st.metric("Bus Voltage (kV)", f"{row['bus_voltage_kv']:.2f}")
        st.metric("Bus Current (A)", f"{row['bus_current_a']:.1f}")
        plot_timeseries(swgr_df, "timestamp", "bus_voltage_kv", "Bus Voltage Trend", location)

if page == "📡 Transmission & Grid Line":
    row = get_row(tline_df, location)
    if row is not None:
        st.metric("Line Voltage (kV)", f"{row['line_voltage_kv']:.2f}")
        st.metric("Line Current (A)", f"{row['line_current_a']:.1f}")
        st.metric("Power Flow (MW)", f"{row['power_flow_mw']:.2f}")
        plot_timeseries(tline_df, "timestamp", "power_flow_mw", "Transmission Power Flow Trend", location)

if page == "🚨 Alarms & Faults":
    batt_row = get_row(battery_df, location)
    pcs_row  = get_row(pcs_df, location)
    xfmr_row = get_row(xfmr_df, location)
    alarms = []
    if batt_row is not None and batt_row['average_cell_temperature_c'] > 38:
        alarms.append("⚠️ Battery Overheating")
    if pcs_row is not None and abs(pcs_row['frequency_hz'] - 50) > 0.5:
        alarms.append("⚠️ PCS Frequency Deviation")
    if xfmr_row is not None and xfmr_row['load_pct'] > 90:
        alarms.append("⚠️ Transformer Overload")
    if alarms:
        for a in alarms:
            st.error(a)
    else:
        st.success("✅ No active alarms")

if page == "🔮 Forecast & AI Advisory":
    st.markdown("### 🔮 Forecast & AI Advisory")
    hours = np.arange(0, 24)
    pv_forecast = np.sin(hours/24*2*np.pi)*30 + 20
    load_forecast = np.cos(hours/24*2*np.pi)*10 + 25
    price_forecast = np.random.normal(6, 0.5, 24)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=pv_forecast, name="PV Forecast", line=dict(color="yellow")))
    fig.add_trace(go.Scatter(x=hours, y=load_forecast, name="Load Forecast", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=hours, y=price_forecast, name="Price Forecast", line=dict(color="magenta")))
    fig.update_layout(title="Next 24h Forecast (Probabilistic)", xaxis_title="Hour", yaxis_title="MW / ₹", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    st.info("**Dispatch Optimizer (AI Recommends – Human-Gated):**\n\n"
            "Charge 8 MW until 14:00 → discharge 22 MW across 18:00–21:
