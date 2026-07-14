

import pandas as pd
import sqlite3
import time
import os
from datetime import datetime

INTERVAL_SEC  = 5
DB_PATH       = "neoai_live.db"
MAX_ROWS      = 1000  # Keep DB lightweight

# ── FIX: Adjusted paths to match your actual file extensions ─────────────────
FILES = {
    "battery":     r"C:\Users\admin\Documents\battery_neoai_dataset.csv",
    "pcs":         r"C:\Users\admin\Documents\pcs_neoai_dataset.csv",
    "transformer": r"C:\Users\admin\Documents\transformer_neoai_dataset.csv",
    "switchgear":  r"C:\Users\admin\Documents\switchgear_neoai_dataset.xlsx", 
    "tline":       r"C:\Users\admin\Documents\transmission_line_neoai_dataset.xlsx",
}

def load_and_prep_datasets():
    datasets = {}
    for name, path in FILES.items():
        try:
            # Dynamically read CSV or Excel based on the file extension
            if path.endswith(".csv"):
                df = pd.read_csv(path, low_memory=False)
            elif path.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                print(f" ✗ {name:12s} ERROR: Unsupported file type.")
                continue
            
            # Smart Sorting: Force SOC to discharge logically if it's the battery dataset
            if name == "battery" and "state_of_charge_soc_pct" in df.columns:
                df = df.sort_values(by="state_of_charge_soc_pct", ascending=False).reset_index(drop=True)
            elif "timestamp" in df.columns:
                df = df.sort_values(by="timestamp").reset_index(drop=True)
                
            datasets[name] = df
            print(f" ✓ {name:12s} loaded ({len(df)} rows)")
        except Exception as e:
            print(f" ✗ {name:12s} ERROR: {e}")
    return datasets

def init_db(conn, datasets):
    cur = conn.cursor()
    for name, df in datasets.items():
        cols = []
        for col in df.columns:
            if pd.api.types.is_integer_dtype(df[col]): type_str = "INTEGER"
            elif pd.api.types.is_float_dtype(df[col]): type_str = "REAL"
            else: type_str = "TEXT"
            cols.append(f'"{col}" {type_str}')
        cur.execute(f'CREATE TABLE IF NOT EXISTS {name} ({", ".join(cols)})')
    
    cur.execute("CREATE TABLE IF NOT EXISTS meta_tracker (component TEXT, location TEXT, current_row INTEGER)")
    conn.commit()

def get_row_idx(conn, comp, loc):
    cur = conn.cursor()
    cur.execute("SELECT current_row FROM meta_tracker WHERE component=? AND location=?", (comp, loc))
    res = cur.fetchone()
    return res[0] if res else 0

def update_row_idx(conn, comp, loc, idx):
    cur = conn.cursor()
    cur.execute("DELETE FROM meta_tracker WHERE component=? AND location=?", (comp, loc))
    cur.execute("INSERT INTO meta_tracker VALUES (?, ?, ?)", (comp, loc, idx))
    conn.commit()

def main():
    print("=" * 55)
    print("  NeoAI Multi-Site Simulator Started")
    print("=" * 55)
    datasets = load_and_prep_datasets()
    if not datasets: 
        print("No datasets loaded. Halting.")
        return

    conn = sqlite3.connect(DB_PATH)
    init_db(conn, datasets)
    
    locations = ["Prakasha_Nandurbar", "Bhandu_Rajasthan"]
    print(f"\nStreaming data for sites: {locations}")
    print("Press Ctrl+C to stop.\n")

    tick = 0
    while True:
        tick += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for loc in locations:
            for name, df in datasets.items():
                if "location" not in df.columns: continue
                
                # Filter df by location
                loc_df = df[df["location"] == loc].reset_index(drop=True)
                if len(loc_df) == 0: continue
                
                idx = get_row_idx(conn, name, loc) % len(loc_df)
                row = loc_df.iloc[idx].copy()
                row["timestamp"] = now_str # Override with live time
                
                # Insert into DB
                cols = ", ".join([f'"{c}"' for c in row.index])
                vals = [str(v) if isinstance(v, bool) else v for v in row.values]
                plcs = ", ".join(["?" for _ in vals])
                conn.execute(f'INSERT INTO {name} ({cols}) VALUES ({plcs})', vals)
                
                # Keep DB small
                conn.execute(f"""DELETE FROM {name} WHERE location='{loc}' AND rowid NOT IN 
                                 (SELECT rowid FROM {name} WHERE location='{loc}' ORDER BY rowid DESC LIMIT {MAX_ROWS})""")
                
                update_row_idx(conn, name, loc, idx + 1)
        
        conn.commit()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Tick #{tick:04d} -> Pushed data for Prakasha & Bhandu")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulator stopped manually.")



