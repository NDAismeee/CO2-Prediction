#!/usr/bin/env python3
# drop_fixed.py
import argparse
from pathlib import Path
import pandas as pd

# Columns to drop (fixed list)
DROP_COLS = [
    "riceEmissionCO2",
    "livestockEmissionCO2",
    "carbonSequestration",
    "CO2Emission",
]

def main():
    ap = argparse.ArgumentParser(
        description="Drop fixed columns from Excel and save as new file"
    )
    ap.add_argument("infile", help="Path to input .xlsx")
    ap.add_argument("-o", "--outfile", required=True, help="Path to output .xlsx")
    ap.add_argument("--sheet", default=None,
                    help="Sheet name to read (default: first sheet)")
    args = ap.parse_args()

    # Load Excel properly
    if args.sheet:
        df = pd.read_excel(args.infile, sheet_name=args.sheet)
    else:
        # read first sheet only
        df = pd.read_excel(args.infile, sheet_name=0)

    # Drop only those columns that exist
    to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)

    # Save new Excel
    out = Path(args.outfile)
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=args.sheet or "Sheet1")

    print(f"✓ Wrote {len(df)} rows to {out}, dropped columns: {to_drop}")

if __name__ == "__main__":
    main()
