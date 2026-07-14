import pandas as pd
import psycopg2
import os
import time
from datetime import datetime

INTERVAL_SEC = 5
MAX_ROWS = 1000

# Render/Heroku will inject DATABASE_URL into env vars
DB_URL = os.getenv("DATABASE_URL")

FILES = {
    "battery":     "battery_neoai_dataset.csv",
    "pcs":         "pcs_neoai_dataset.csv",
    "transformer": "transformer_neoai_dataset.csv",
    "switchgear":  "switchgear_neoai_dataset.xlsx",
    "tline":       "transmission_line_neoai_dataset.xlsx",
}

def load_and_prep_datasets():
    datasets = {}
    for name, path in FILES.items():
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path, low_memory=False)
            elif path.endswith(".xlsx"):
                df = pd.read_excel(path)
            else:
                print(f" ✗ {name:12s} ERROR: Unsupported file type.")
                continue

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
            if pd.api.types.is_integer_dtype(df[col]): type_str = "INT"
            elif pd.api.types.is_float_dtype(df[col]): type_str = "REAL"
            else: type_str = "TEXT"
            cols.append(f'"{col}" {type_str}')
        cur.execute(f'CREATE TABLE IF NOT EXISTS {name} ({", ".join(cols)})')
    conn.commit()

def main():
    print("=" * 55)
    print("  NeoAI Multi-Site Simulator Started (Postgres)")
    print("=" * 55)

    datasets = load_and_prep_datasets()
    if not datasets:
        print("No datasets loaded. Halting.")
        return

    conn = psycopg2.connect(DB_URL)
    init_db(conn, datasets)
    cur = conn.cursor()

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
                loc_df = df[df["location"] == loc].reset_index(drop=True)
                if len(loc_df) == 0: continue

                idx = tick % len(loc_df)
                row = loc_df.iloc[idx].copy()
                row["timestamp"] = now_str

                cols = ", ".join([f'"{c}"' for c in row.index])
                plcs = ", ".join(["%s" for _ in row.values])
                vals = list(row.values)

                cur.execute(f'INSERT INTO {name} ({cols}) VALUES ({plcs})', vals)

                # Keep DB small
                cur.execute(f"""
                    DELETE FROM {name}
                    WHERE location=%s AND ctid NOT IN (
                        SELECT ctid FROM {name}
                        WHERE location=%s ORDER BY ctid DESC LIMIT %s
                    )
                """, (loc, loc, MAX_ROWS))

        conn.commit()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Tick #{tick:04d} -> Pushed data for sites")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulator stopped manually.")
