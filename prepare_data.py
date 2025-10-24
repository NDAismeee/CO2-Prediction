#!/usr/bin/env python3
# combine_emissions_dir.py
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

WANTED_COLS = [
    "year", "id", "name", "area",
    "riceEmissionCH4", "riceEmissionN2O", "riceEmissionCO2",
    "livestockEmissionCH4", "livestockEmissionN2O", "livestockEmissionCO2",
    "carbonSequestration", "carbonSequestrationBelowGround", "carbonSequestrationAboveGround",
    "CH4Emission", "N2OEmission", "CO2Emission",
]

YEAR_RE = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")

def year_from_filename(path: Path) -> int | None:
    m = YEAR_RE.search(path.name)
    return int(m.group(0)) if m else None

def parse_row(obj: dict, year: int | None) -> dict:
    ward = obj.get("ward", {}) if isinstance(obj, dict) else {}
    forest = obj.get("forestCarbonSequestration", {}) if isinstance(obj, dict) else {}

    return {
        "year": year,
        "id": ward.get("id"),
        "name": ward.get("name"),
        "area": ward.get("area"),  # change to forest.get("area") if you prefer that area
        "riceEmissionCH4": obj.get("riceEmissionCH4"),
        "riceEmissionN2O": obj.get("riceEmissionN2O"),
        "riceEmissionCO2": obj.get("riceEmissionCO2"),
        "livestockEmissionCH4": obj.get("livestockEmissionCH4"),
        "livestockEmissionN2O": obj.get("livestockEmissionN2O"),
        "livestockEmissionCO2": obj.get("livestockEmissionCO2"),
        "carbonSequestration": forest.get("carbonSequestration"),
        "carbonSequestrationBelowGround": forest.get("carbonSequestrationBelowGround"),
        "carbonSequestrationAboveGround": forest.get("carbonSequestrationAboveGround"),
        "CH4Emission": obj.get("CH4Emission"),
        "N2OEmission": obj.get("N2OEmission"),
        "CO2Emission": obj.get("CO2Emission"),
    }

def combine_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for jp in sorted(folder.glob("*.json")):
        yr = year_from_filename(jp)
        try:
            text = jp.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = jp.read_text(encoding="utf-8-sig")
        try:
            payload = json.loads(text)
        except Exception as e:
            print(f"[warn] skip {jp.name}: {e}", file=sys.stderr)
            continue

        data = payload.get("data", [])
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            print(f"[warn] {jp.name} has unexpected 'data' type: {type(data)}", file=sys.stderr)
            continue

        for item in data:
            if isinstance(item, dict):
                rows.append(parse_row(item, yr))

    df = pd.DataFrame(rows, columns=WANTED_COLS)
    if not df.empty:
        df = df.sort_values(by=[c for c in ("year", "id") if c in df.columns]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser(description="Combine yearly ward_emission JSON files in a folder into one Excel.")
    ap.add_argument("folder", default="ward_emission", help="Path to folder containing files like 2018.json, 2019.json, ...")
    ap.add_argument("-o", "--out", default="ward_emission_combined.xlsx", help="Output Excel path")
    ap.add_argument("--sheet", default="emissions", help="Excel sheet name")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: not a folder: {folder}", file=sys.stderr)
        sys.exit(1)

    df = combine_folder(folder)
    with pd.ExcelWriter(args.out, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name=args.sheet)

    print(f"✓ Wrote {len(df)} rows from {folder} to {args.out}")

if __name__ == "__main__":
    main()
