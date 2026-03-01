import re
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

IN_PATH  = "Rates.xlsx"
OUT_PATH = "Rates_SpreadsFlys_MR.xlsx"

# Map month-tenors that Bloomberg uses (13M, 26M, ...) to standard year labels
MONTH_TO_YEAR = {
    12:"1Y", 13:"1Y",
    24:"2Y", 25:"2Y", 26:"2Y",
    36:"3Y", 37:"3Y", 38:"3Y", 39:"3Y",
    48:"4Y", 49:"4Y", 50:"4Y", 51:"4Y", 52:"4Y",
    60:"5Y", 61:"5Y", 62:"5Y", 63:"5Y", 64:"5Y", 65:"5Y",
    84:"7Y", 85:"7Y", 90:"7Y", 91:"7Y", 92:"7Y",
    120:"10Y", 121:"10Y", 130:"10Y",
    180:"15Y", 195:"15Y",
    240:"20Y", 260:"20Y",
    300:"25Y",
    360:"30Y", 390:"30Y",
}

def parse_tenor_label(lbl):
    if lbl is None or (isinstance(lbl, float) and np.isnan(lbl)):
        return None
    s = str(lbl).strip()
    m = re.fullmatch(r"(\d+)\s*M", s, re.IGNORECASE)
    if m:
        mm = int(m.group(1))
        return MONTH_TO_YEAR.get(mm, f"{mm}M")
    m = re.fullmatch(r"(\d+)\s*Y", s, re.IGNORECASE)
    if m:
        return f"{int(m.group(1))}Y"
    return s

def convert_date_series(s: pd.Series) -> pd.Series:
    # key fix: if object but numeric-like, coerce to numeric first
    if s.dtype == object:
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().sum() >= max(5, int(0.5 * len(s))):
            s = sn

    if np.issubdtype(s.dtype, np.number):
        med = np.nanmedian(s.astype(float))
        # Excel serial days usually in ~ 20k–80k range
        if 20000 <= med <= 80000:
            return pd.to_datetime(s, unit="D", origin="1899-12-30", errors="coerce")
        # fallback: unix timestamps (rare here)
        if med > 1e11:
            return pd.to_datetime(s, unit="ms", errors="coerce")
        if med > 1e9:
            return pd.to_datetime(s, unit="s", errors="coerce")

    return pd.to_datetime(s, errors="coerce")

def clean_rates_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(xls, sheet_name=sheet, header=None)

    headers = raw.iloc[0].tolist()      # Tenor, 3M, 6M, ...
    df = raw.iloc[2:].copy()            # drop header row + ticker row
    df.columns = headers

    df = df.rename(columns={df.columns[0]: "Tenor"})
    df["Tenor"] = convert_date_series(df["Tenor"])

    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Tenor"])
    rate_cols = df.columns[1:]
    df = df.dropna(subset=rate_cols, how="all").sort_values("Tenor").drop_duplicates("Tenor")

    df[rate_cols] = df[rate_cols].ffill()
    df = df.dropna(subset=rate_cols, how="any")

    df = df.set_index("Tenor")
    df.columns = [parse_tenor_label(c) for c in df.columns]
    return df

# US_TimeSeries formulas
# SPREADS = [("7Y", "8Y"), ("10Y", "12Y"), ("20Y", "25Y"), ("25Y", "30Y")]
# FLYS    = [("3Y", "4Y", "5Y"), ("5Y", "7Y", "10Y"), ("10Y", "12Y", "15Y"), ("20Y", "25Y", "30Y")]

def spread_name(a, b): return f"{a.replace('Y','y')}{b.replace('Y','y')}"
def fly_name(a, b, c): return f"{a.replace('Y','y')}{b.replace('Y','y')}{c.replace('Y','y')}"

def tenor_to_years(lbl: str) -> float | None:
    """
    Convert labels like '3M','6M','1Y','2Y','10Y' to years as float.
    Returns None if cannot parse.
    """
    if lbl is None:
        return None
    s = str(lbl).strip().upper()

    m = re.fullmatch(r"(\d+)\s*M", s)
    if m:
        return int(m.group(1)) / 12.0

    m = re.fullmatch(r"(\d+)\s*Y", s)
    if m:
        return float(int(m.group(1)))

    return None

def compute_spreads_flys(
    rates_df: pd.DataFrame,
    *,
    max_spread_gap_years: float = 10.0,
    scale_to_bps: bool = True,
):

    def tenor_to_years(lbl):
        s = str(lbl).upper().strip()
        if s.endswith("M"):
            return float(s[:-1]) / 12.0
        if s.endswith("Y"):
            return float(s[:-1])
        return None

    # parse & sort tenors
    ty = [(c, tenor_to_years(c)) for c in rates_df.columns]
    ty = [(c,y) for c,y in ty if y is not None]
    ty.sort(key=lambda x: x[1])

    cols = [c for c,_ in ty]
    yrs  = np.array([y for _,y in ty])

    out = {}

    n = len(cols)

    # ---- spreads ----
    for i in range(n):
        for j in range(i+1, n):
            if yrs[j] - yrs[i] <= max_spread_gap_years:
                name = f"spread_{cols[i].lower()}_{cols[j].lower()}"
                out[name] = rates_df[cols[j]] - rates_df[cols[i]]

    # ---- consecutive flies ONLY ----
    for i in range(n-2):
        cL, cM, cR = cols[i], cols[i+1], cols[i+2]
        name = f"fly_{cL.lower()}_{cM.lower()}_{cR.lower()}"
        out[name] = rates_df[cL] + rates_df[cR] - 2.0 * rates_df[cM]

    df = pd.DataFrame(out, index=rates_df.index)

    if scale_to_bps:
        df *= 100.0

    df = df.dropna(how="all").ffill().dropna(how="any")

    if df.shape[1] == 0:
        raise ValueError("No spreads/flys generated.")

    return df

def adf_pvalue_fast(x: pd.Series, maxlag=5) -> float:
    s = x.dropna()
    if len(s) < 200:
        return np.nan
    try:
        return adfuller(s.values, maxlag=maxlag, autolag=None, regression="c")[1]
    except Exception:
        return np.nan

def filter_mean_reverting(df: pd.DataFrame, alpha=0.05):
    pvals = {c: adf_pvalue_fast(df[c]) for c in df.columns}
    keep  = [c for c, p in pvals.items() if (p == p and p < alpha)]
    return df[keep].copy(), pd.Series(pvals).sort_values()

def main():
    xls = pd.ExcelFile(IN_PATH, engine="openpyxl")
    results = {}
    pvals_all = []

    for sh in xls.sheet_names:
        rates = clean_rates_sheet(xls, sh)
        try:
            sf = compute_spreads_flys(rates)
        except Exception:
            continue

        kept, pvals = filter_mean_reverting(sf, alpha=0.05)
        results[sh] = kept

        for col, pv in pvals.items():
            pvals_all.append({"Market": sh, "Series": col, "ADF_pvalue": pv})

    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        for sh, df in results.items():
            df.to_excel(writer, sheet_name=sh, index=True)  # Tenor is index (first col)
        pd.DataFrame(pvals_all).sort_values(["Market","ADF_pvalue"]).to_excel(
            writer, sheet_name="ADF_pvalues", index=False
        )

    print(f"Wrote {OUT_PATH}")
    print("Sheets written:", list(results.keys()))

if __name__ == "__main__":
    main()