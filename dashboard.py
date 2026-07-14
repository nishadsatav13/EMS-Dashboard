"""
NeoAI — Enterprise EMS Dashboard v8.0 (Simulator Embedded, All Components)
"""

import streamlit as st
import pandas as pd
import time

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(page_title="NeoAI EMS", page_icon="⚡", layout="wide")

# ── Load datasets from repo ───────────────────────────────────
battery_df     = pd.read_csv("battery_neoai_dataset.csv")
pcs_df         = pd.read_csv("pcs_neoai_dataset.csv")
xfmr_df        = pd.read_excel("transformer_neoai_dataset.xlsx")
swgr_df        = pd.read_excel("switchgear_neoai_dataset.xlsx")
tline_df       = pd.read_excel("transmission_line_neoai_dataset.xlsx")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    location = st.selectbox("Select Plant Location", ["Prakasha_Nandurbar", "Bhandu_Rajasthan"])
    page = st.radio("Select Component View", [
        "🏠 Master Overview",
        "🔋 Battery Storage (LFP)",
        "⚡ Power Conversion (PCS)",
        "🔌 Distribution Transformer",
        "🔧 Main Switchgear Panel",
        "📡 Transmission & Grid Line",
        "🚨 Alarms & Faults"
    ])
    refresh_rate = st.slider("Telemetry Refresh (sec)", 2, 10, 5)

# ── Simulator Logic ──────────────────────────────────────────
if "tick" not in st.session_state:
    st.session_state.tick = 0
st.session_state.tick += 1

def get_row(df, location):
    loc_df = df[df["location"] == location].reset_index(drop=True)
    if len(loc_df) == 0:
        return None
    idx = st.session_state.tick % len(loc_df)
    return loc_df.iloc[idx]

# ── Master Overview ──────────────────────────────────────────
if page == "🏠 Master Overview":
    batt_row = get_row(battery_df, location)
    pcs_row  = get_row(pcs_df, location)
    xfmr_row = get_row(xfmr_df, location)

    if batt_row is not None:
        st.metric("System SOC (%)", f"{batt_row['state_of_charge_soc_pct']:.1f}")
        st.metric("System SOH (%)", f"{batt_row['state_of_health_soh_pct']:.1f}")
        st.metric("Battery Power (kW)", f"{batt_row['battery_power_kw']:.1f}")
    if pcs_row is not None:
        st.metric("PCS Output (kW)", f"{pcs_row['active_power_kw']:.1f}")
    if xfmr_row is not None:
        st.metric("Transformer Load (%)", f"{xfmr_row['load_pct']:.1f}")

# ── Battery Storage ──────────────────────────────────────────
if page == "🔋 Battery Storage (LFP)":
    row = get_row(battery_df, location)
    if row is not None:
        st.metric("SOC (%)", f"{row['state_of_charge_soc_pct']:.1f}")
        st.metric("SOH (%)", f"{row['state_of_health_soh_pct']:.1f}")
        st.metric("Power (kW)", f"{row['battery_power_kw']:.1f}")
        st.metric("Avg Cell Temp (°C)", f"{row['average_cell_temperature_c']:.1f}")

# ── PCS ──────────────────────────────────────────────────────
if page == "⚡ Power Conversion (PCS)":
    row = get_row(pcs_df, location)
    if row is not None:
        st.metric("Active Power (kW)", f"{row['active_power_kw']:.1f}")
        st.metric("Reactive Power (kVAR)", f"{row['reactive_power_kvar']:.1f}")
        st.metric("Frequency (Hz)", f"{row['frequency_hz']:.2f}")

# ── Transformer ──────────────────────────────────────────────
if page == "🔌 Distribution Transformer":
    row = get_row(xfmr_df, location)
    if row is not None:
        st.metric("Load (%)", f"{row['load_pct']:.1f}")
        st.metric("Oil Temp (°C)", f"{row['oil_temperature_c']:.1f}")
        st.metric("Winding Temp (°C)", f"{row['winding_temperature_c']:.1f}")

# ── Switchgear ───────────────────────────────────────────────
if page == "🔧 Main Switchgear Panel":
    row = get_row(swgr_df, location)
    if row is not None:
        st.metric("Breaker Status", row['breaker_status'])
        st.metric("Bus Voltage (kV)", f"{row['bus_voltage_kv']:.2f}")
        st.metric("Bus Current (A)", f"{row['bus_current_a']:.1f}")

# ── Transmission Line ────────────────────────────────────────
if page == "📡 Transmission & Grid Line":
    row = get_row(tline_df, location)
    if row is not None:
        st.metric("Line Voltage (kV)", f"{row['line_voltage_kv']:.2f}")
        st.metric("Line Current (A)", f"{row['line_current_a']:.1f}")
        st.metric("Power Flow (MW)", f"{row['power_flow_mw']:.2f}")

# ── Alarms & Faults ──────────────────────────────────────────
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

# ── Auto Refresh ─────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
