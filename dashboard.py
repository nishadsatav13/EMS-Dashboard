"""
NeoAI — Enterprise EMS Dashboard v12.0
Simulator Embedded, All Components, Charts + Forecast & AI Advisory
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="NeoAI EMS", page_icon="⚡", layout="wide")

# ── Load datasets from repo ───────────────────────────────────
battery_df     = pd.read_csv("battery_neoai_dataset.csv")
pcs_df         = pd.read_csv("pcs_neoai_dataset.csv")
xfmr_df        = pd.read_csv("transformer_neoai_dataset.csv")
swgr_df        = pd.read_excel("switchgear_neoai_dataset.xlsx")
tline_df       = pd.read_excel("transmission_line_neoai_dataset.xlsx")

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

# ── Forecast & AI Advisory ───────────────────────────────────
if page == "🔮 Forecast & AI Advisory":
    st.markdown("### 🔮 Forecast & AI Advisory")

    # Mock forecast data
    hours = np.arange(0, 24)
    pv_forecast = np.sin(hours/24*2*np.pi)*30 + 20
    load_forecast = np.cos(hours/24*2*np.pi)*10 + 25
    price_forecast = np.random.normal(6, 0.5, 24)

    # Forecast chart with uncertainty bands
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=pv_forecast, name="PV Forecast", line=dict(color="yellow")))
    fig.add_trace(go.Scatter(x=hours, y=load_forecast, name="Load Forecast", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=hours, y=price_forecast, name="Price Forecast", line=dict(color="magenta")))
    fig.update_layout(title="Next 24h Forecast (Probabilistic)", xaxis_title="Hour", yaxis_title="MW / ₹", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Dispatch Optimizer (mock advisory text)
    st.info("**Dispatch Optimizer (AI Recommends – Human-Gated):**\n\n"
            "Charge 8 MW until 14:00 to capture surplus PV → discharge 22 MW across 18:00–21:00 evening peak (₹8.9/kWh).\n"
            "Expected gain +₹2.1L vs baseline.\n\n"
            "Confidence: 0.91 ✓")

    # Safety & Protection indicators
    st.markdown("### 🛡️ Safety & Protection")
    cols = st.columns(3)
    cols[0].success("Thermal Interlock ✓")
    cols[0].success("Overcurrent ✓")
    cols[1].success("Ground Fault ✓")
    cols[1].success("Fire Suppression ✓")
    cols[2].success("Contactors ✓")
    cols[2].success("BMS Heartbeat ✓")

    # Rack Health Map (mock anomaly scores)
    st.markdown("### 🗺️ Rack Health Map (Anomaly Overlay)")
    anomaly_scores = np.random.rand(5,5)
    fig = go.Figure(data=go.Heatmap(z=anomaly_scores, colorscale="Viridis"))
    fig.update_layout(title="Rack Health Anomaly Scores", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Events & Alarms (mock log)
    st.markdown("### 📜 Events & Alarms")
    st.warning("13:20 — PCS frequency deviation detected (auto-corrected)")
    st.success("13:25 — Transformer load normalized")
    st.error("13:30 — Battery temp spike (watch mode)")

# ── Auto Refresh ─────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
