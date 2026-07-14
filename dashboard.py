"""
NeoAI — Enterprise EMS Dashboard v7.0 (Simulator Embedded)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import os
import random
import time
from datetime import datetime

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeoAI — Enterprise EMS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Database Connection ──────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL")

def push_fake_data(location):
    """Simulates telemetry rows directly into Postgres."""
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Example: Battery table
        soc = random.uniform(40, 100)
        soh = random.uniform(85, 100)
        pwr = random.uniform(-50, 50)
        temp = random.uniform(25, 40)

        cur.execute("""
            INSERT INTO battery (timestamp, location, state_of_charge_soc_pct,
                                 state_of_health_soh_pct, battery_power_kw,
                                 average_cell_temperature_c)
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (now_str, location, soc, soh, pwr, temp))

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Simulator error: {e}")

def load_table(table_name, location, limit=80):
    try:
        conn = psycopg2.connect(DB_URL)
        query = f'SELECT * FROM "{table_name}" WHERE location=%s ORDER BY timestamp DESC LIMIT %s'
        df = pd.read_sql(query, conn, params=(location, limit))
        conn.close()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df.iloc[::-1].reset_index(drop=True)
    except Exception as e:
        st.error(f"DB error: {e}")
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

# ── Sidebar Navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2>NeoAI EMS</h2>", unsafe_allow_html=True)
    selected_location = st.selectbox("Select Plant Location", ["Prakasha_Nandurbar", "Bhandu_Rajasthan"])
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

# ── Embedded Simulator ──────────────────────────────────────────────────────
push_fake_data(selected_location)

# ── Load Data from Postgres ──────────────────────────────────────────────────
batt  = load_table("battery", selected_location)

if len(batt) == 0:
    st.warning(f"⏳ Waiting for simulator to push data for {selected_location}...")
    st.stop()

# ── Example Master Overview KPIs ────────────────────────────────────────────
if page == "🏠 Master Overview":
    soc = safe_val(batt, "state_of_charge_soc_pct")
    soh = safe_val(batt, "state_of_health_soh_pct")
    pwr = safe_val(batt, "battery_power_kw")
    temp = safe_val(batt, "average_cell_temperature_c")

    st.metric("System SOC (%)", f"{soc:.1f}")
    st.metric("System SOH (%)", f"{soh:.1f}")
    st.metric("Active Power (kW)", f"{pwr:.1f}")
    st.metric("Avg Cell Temp (°C)", f"{temp:.1f}")

# ── Auto Refresh ────────────────────────────────────────────────────────────
time.sleep(refresh_rate)
st.rerun()
