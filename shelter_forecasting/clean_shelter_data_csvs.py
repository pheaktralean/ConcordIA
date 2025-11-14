from pathlib import Path
import re
from datetime import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#configuration
CLEAN_OUTPUTS = False  # set True to delete previous outputs before writing new ones

FORMAT_A_2017_2020 = {
}

#Headers for 2021–2024(2025).
FORMAT_B_2021_2024 = {
    "OCCUPANCY_DATE": "date",
    "SERVICE_USER_COUNT": "value",
}

#Root folders
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
CURATED = ROOT / "data" / "curated"
INTERIM.mkdir(parents=True, exist_ok=True)
CURATED.mkdir(parents=True, exist_ok=True)

#Output files
DAILY_OUT = INTERIM / "overnight_occupancy_daily_cleaned.csv"
WEEKLY_OUT = CURATED / "overnight_occupancy_weekly_avg.csv"
REPORT_OUT = INTERIM / "validation_report.csv"
CHART_DAILY = CURATED / "chart_daily.png"
CHART_WEEKLY = CURATED / "chart_weekly.png"
CHART_MONTHLY = CURATED / "chart_monthly_seasonality.png"

CURRENT_YEAR = datetime.now().year

#Helpers
def read_csv_loose(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")

def pick_format(header: list[str]) -> dict:
    cols = set(c.strip() for c in header)
    if FORMAT_A_2017_2020 and set(FORMAT_A_2017_2020.keys()).issubset(cols):
        return FORMAT_A_2017_2020
    if FORMAT_B_2021_2024 and set(FORMAT_B_2021_2024.keys()).issubset(cols):
        return FORMAT_B_2021_2024
    return {}

def guess_date_column(df: pd.DataFrame) -> str | None:
    candidates = [c for c in df.columns if re.search(r"(date|report)", c, flags=re.I)]
    if df.columns.size:
        candidates.append(df.columns[0])
    for c in dict.fromkeys(candidates):
        dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() >= 0.6:
            return c
    for c in df.columns:
        dt = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
        if dt.notna().mean() >= 0.6:
            return c
    return None

def guess_value_column(df: pd.DataFrame) -> str | None:
    patt = re.compile(r"(occup|overnight|beds|capacity|count|clients|user)", re.I)
    priority = [c for c in df.columns if patt.search(c)]
    numericish = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
    for c in priority + numericish:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= 0.6 and (s.dropna() >= 0).mean() > 0.95:
            return c
    return None

def detect_id_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.lower() == "_id":
            return c
    for c in df.columns:
        if re.search(r"_id$", c, flags=re.I):
            return c
    for c in df.columns:
        if c.lower() == "id":
            return c
    return None

def years_from_filename(name: str) -> list[int]:
    return [int(y) for y in re.findall(r"(20\d{2})", name)]

def _yy_to_yyyy(yy: int) -> int:
    return 2000 + yy if yy < 100 else yy

def clean_date_token(s: str) -> str:
    """Keep only digits and common separators; drop trailing duplicate year after '_'."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    #drop anything after space
    s = s.split(" ")[0]
    #if an underscore repeats the year, keep the part before first underscore
    if "_" in s:
        s = s.split("_")[0]
    #keep digits and separators
    s = re.sub(r"[^0-9/\-\.]", "", s)
    return s

def try_parse_with_order(tokens: list[int], order: str) -> datetime | None:
    """order is 'ymd', 'ydm', 'mdy', 'dmy' where first element is year (2 or 4), etc."""
    t = tokens.copy()
    if len(t) != 3:
        return None
    #Map tokens to y,m,d depending on order
    idx = {"y": order.index("y"), "m": order.index("m"), "d": order.index("d")}
    y = _yy_to_yyyy(t[idx["y"]])
    m = t[idx["m"]]
    d = t[idx["d"]]
    try:
        return datetime(y, m, d)
    except Exception:
        return None

def normalize_and_parse_dates(raw_series: pd.Series, fname: str) -> tuple[pd.Series, float, str]:
    """
    Aggressively normalize strings like '21-31-01_21', '21/01/31', '2021-1-3', etc.
    Try multiple orders and pick the one that yields the most dates within filename year window.
    Returns: (parsed_datetimes, good_ratio [0..1], chosen_order)
    """
    yrs = years_from_filename(fname)
    lo = datetime(min(yrs) - 1, 1, 1) if yrs else datetime(2010, 1, 1)
    hi = datetime(max(yrs) + 1, 12, 31) if yrs else datetime(CURRENT_YEAR + 1, 12, 31)

    cleaned = raw_series.astype(str).map(clean_date_token)
    #Extract numbers; allow 1 or 2 digits for m/d, and 2 or 4 for year
    tokenized = cleaned.map(lambda s: [int(x) for x in re.split(r"[/\-\.]", s) if x.isdigit()])

    orders = ["ymd", "ydm", "mdy", "dmy"]  #test all plausible combos
    scores = {o: 0 for o in orders}
    parsed_by_order = {}

    #Sample up to 500 rows for speed to choose the best order
    sample_idx = tokenized.index[: min(500, len(tokenized))]
    for o in orders:
        parsed = []
        ok_in_window = 0
        for idx in sample_idx:
            toks = tokenized.loc[idx]
            if len(toks) == 3:
                dt = try_parse_with_order(toks, o)
            else:
                dt = None
            parsed.append(dt)
            if dt and lo <= dt <= hi:
                ok_in_window += 1
        scores[o] = ok_in_window
        parsed_by_order[o] = parsed

    #Choose best order
    best_order = max(scores, key=scores.get)
    #Parse full series with best order
    full_parsed = []
    ok_total = 0
    for toks in tokenized:
        if len(toks) == 3:
            dt = try_parse_with_order(toks, best_order)
        else:
            #Try pandas as a last resort
            try:
                dt = pd.to_datetime("-".join(map(str, toks)), errors="coerce")
                dt = pd.NaT if pd.isna(dt) else dt.to_pydatetime()
            except Exception:
                dt = None
        full_parsed.append(dt)
        if dt and lo <= dt <= hi:
            ok_total += 1

    good_ratio = ok_total / max(1, len(tokenized))
    parsed_series = pd.to_datetime(pd.Series(full_parsed), errors="coerce")
    #clamp to window
    parsed_series = parsed_series.where((parsed_series >= pd.Timestamp(lo)) & (parsed_series <= pd.Timestamp(hi)))
    return parsed_series, good_ratio, best_order

def days_in_year(year: int) -> int:
    return 366 if pd.Timestamp(year=year, month=12, day=31).is_leap_year else 365

def bin_rows_into_days(df: pd.DataFrame, idcol: str, fname: str) -> pd.DataFrame:
    """
    Sort by _id and evenly bin rows into days of the (earliest) year in filename.
    Aggregate 'value' per day by sum (or mean if sum is NaN).
    """
    yrs = years_from_filename(fname)
    if yrs:
        start_year = min(yrs)
    else:
        start_year = 2010
    n_days = days_in_year(start_year)
    start_date = pd.Timestamp(f"{start_year}-01-01")

    df = df.sort_values(by=idcol, kind="stable").reset_index(drop=True)
    n = len(df)
    #Assign a bin index 0..n_days-1 proportionally along the sorted rows
    #e.g., first ~n/n_days rows -> day 0, next -> day 1, ...
    #this preserves chronology without assuming fixed rows-per-day.
    bin_idx = np.floor(np.linspace(0, n - 1, n) / (n / n_days)).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_days - 1)
    df["_daybin"] = bin_idx
    df["_date"] = start_date + pd.to_timedelta(df["_daybin"], unit="D")

    #Aggregate value per date (sum across rows in the same day)
    agg = df.groupby("_date")["value"].sum(min_count=1).reset_index().rename(columns={"_date": "date"})
    return agg[["date", "value"]]

def standardize_one(df: pd.DataFrame, source_label: str, mapping: dict) -> tuple[pd.DataFrame, dict]:
    info = {
        "file": source_label,
        "detected_date_col": None,
        "detected_value_col": None,
        "detected_id_col": None,
        "used_id_for_dates": False,
        "date_parse_quality": None,   # 0..1
        "binning_note": "",
        "rows_raw": len(df),
        "rows_with_dates": 0,
        "missing_value_pct": None,
        "date_min": None,
        "date_max": None,
    }

    #Map known headers or autodetect
    dcol = vcol = None
    if mapping:
        for k, v in mapping.items():
            if v == "date" and k in df.columns:
                dcol = k
            if v == "value" and k in df.columns:
                vcol = k
    if dcol is None:
        dcol = guess_date_column(df)
    if vcol is None:
        vcol = guess_value_column(df)

    #Detect _id
    idcol = detect_id_column(df)
    if idcol:
        info["detected_id_col"] = idcol

    #Parse dates robustly from the date column (if present)
    parsed_dates = pd.Series([pd.NaT] * len(df))
    best_order = "n/a"
    good_ratio = 0.0
    if dcol:
        parsed_dates, good_ratio, best_order = normalize_and_parse_dates(df[dcol], source_label)
    info["date_parse_quality"] = round(good_ratio, 3)

    #Values
    values = pd.to_numeric(df[vcol], errors="coerce") if vcol else pd.Series([np.nan] * len(df))
    values = values.where((values.isna()) | (values >= 0))  #non-negative check

    #If dates look decent, use them directly and aggregate per day
    use_dates_direct = good_ratio >= 0.9  # threshold: at least 90% of rows parse to plausible dates
    if use_dates_direct:
        out = pd.DataFrame({"date": parsed_dates, "value": values})
        out = out.dropna(subset=["date"]).sort_values("date")
        #aggregate to one value per date (sum across rows)
        out = out.groupby("date", as_index=False)["value"].sum(min_count=1)
    else:
        #Fallback: if we have _id, bin rows into daily buckets in order
        if idcol is not None:
            info["used_id_for_dates"] = True
            out = pd.DataFrame({"value": values.copy()})
            out[idcol] = df[idcol].values
            out = bin_rows_into_days(out, idcol=idcol, fname=source_label)
            info["binning_note"] = "evenly binned rows into days by _id order"
        else:
            #Last resort: drop rows without good dates and aggregate by any remaining parsed dates
            out = pd.DataFrame({"date": parsed_dates, "value": values})
            out = out.dropna(subset=["date"]).sort_values("date")
            out = out.groupby("date", as_index=False)["value"].sum(min_count=1)
            info["binning_note"] = "no _id; used only reliably parsed dates"

    #Filter unrealistic spikes: cap at 99.9th percentile (but keep NaNs)
    if "value" in out and out["value"].notna().any():
        cap = out["value"].quantile(0.999)
        out["value"] = out["value"].where(out["value"].isna() | (out["value"] <= cap), cap)

    info["detected_date_col"] = dcol
    info["detected_value_col"] = vcol
    info["rows_with_dates"] = len(out)
    info["missing_value_pct"] = float(out["value"].isna().mean() * 100) if len(out) else None
    info["date_min"] = out["date"].min() if "date" in out and len(out) else None
    info["date_max"] = out["date"].max() if "date" in out and len(out) else None
    return out, info

def clean_outputs_if_requested():
    if not CLEAN_OUTPUTS:
        return
    for p in [DAILY_OUT, WEEKLY_OUT, REPORT_OUT, CHART_DAILY, CHART_WEEKLY, CHART_MONTHLY]:
        if p.exists():
            try:
                p.unlink()
                print(f"Deleted old file: {p}")
            except Exception as e:
                print(f"Could not delete {p}: {e}")

#main
def main():
    print(f"Project root: {ROOT}")
    print(f"Reading CSVs from: {RAW}")
    clean_outputs_if_requested()

    csv_paths = sorted(RAW.glob("*.csv"))
    if not csv_paths:
        print(f"No CSV files found in {RAW}. Put your files there.")
        return

    cleaned_parts = []
    rows_info = []

    for p in csv_paths:
        df = read_csv_loose(p)
        mapping = pick_format(df.columns.tolist())
        out, info = standardize_one(df, p.name, mapping)
        cleaned_parts.append(out)
        rows_info.append(info)
        print(
            f"{p.name:45s}  date={str(info['detected_date_col']):20s} "
            f"value={str(info['detected_value_col']):20s} "
            f"id={str(info['detected_id_col']):10s} "
            f"used_id_for_dates={info['used_id_for_dates']} "
            f"parse_q={info['date_parse_quality']}"
        )

    #Merge all files to daily totals
    merged = pd.concat(cleaned_parts, ignore_index=True)
    daily = merged.groupby("date", as_index=False)["value"].sum(min_count=1).sort_values("date")

    #Save cleaned daily
    daily.to_csv(DAILY_OUT, index=False, encoding="utf-8-sig")

    #Weekly average (Mon-based week)
    weekly = daily.set_index("date")["value"].resample("W-MON").mean().reset_index()
    weekly.to_csv(WEEKLY_OUT, index=False, encoding="utf-8-sig")

    #Validation report
    report = pd.DataFrame(rows_info)
    for c in ("date_min", "date_max"):
        report[c] = report[c].astype(str)
    report.to_csv(REPORT_OUT, index=False, encoding="utf-8-sig")

    #Charts
    plt.figure()
    plt.plot(daily["date"], daily["value"])
    plt.title("Daily Overnight Occupancy")
    plt.xlabel("Date"); plt.ylabel("Occupancy"); plt.tight_layout()
    plt.savefig(CHART_DAILY, dpi=140)
    plt.close()

    weekly["roll_12w"] = weekly["value"].rolling(12, min_periods=4).mean()
    plt.figure()
    plt.plot(weekly["date"], weekly["value"], label="Weekly Avg")
    plt.plot(weekly["date"], weekly["roll_12w"], label="Rolling 12w")
    plt.legend(); plt.title("Weekly Avg Overnight Occupancy (+ 12w trend)")
    plt.xlabel("Week Start (Mon)"); plt.ylabel("Occupancy"); plt.tight_layout()
    plt.savefig(CHART_WEEKLY, dpi=140)
    plt.close()

    monthly = daily.assign(month=pd.to_datetime(daily["date"]).dt.month).groupby("month")["value"].mean().reset_index()
    plt.figure()
    plt.plot(monthly["month"], monthly["value"])
    plt.title("Average Occupancy by Month")
    plt.xlabel("Month (1–12)"); plt.ylabel("Avg Occupancy"); plt.tight_layout()
    plt.savefig(CHART_MONTHLY, dpi=140)
    plt.close()

    print("\nSaved:")
    print(f" - {DAILY_OUT}")
    print(f" - {WEEKLY_OUT}")
    print(f" - {REPORT_OUT}")
    print(f" - {CHART_DAILY}")
    print(f" - {CHART_WEEKLY}")
    print(f" - {CHART_MONTHLY}")

if __name__ == "__main__":
    main()
