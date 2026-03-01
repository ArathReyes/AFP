"""
MacroConditioning.py
--------------------
Pipeline for studying macro-factor conditioning of PCA-based
Relative Value signals across all rate and FX regions.

For each region this module:
  1. Loads (or recomputes) PCA residuals via RelativeValue2
  2. Runs a time-series OLS:  PnL_t = α + β * ΔFactor_{t-1} + ε
  3. Performs bucket (tercile) analysis of conditional Sharpe by factor level
  4. Computes rolling betas to assess parameter stability
  5. Recommends signal multipliers (2x / 1.5x / 1x) for significant factors

Outputs (saved to output_dir):
  regression_summary.csv        – β, t-stat, p-value, R² per region × factor
  bucket_summary.csv            – conditional Sharpe per tercile per region × factor
  multiplier_recommendations.csv – which factors to condition on and at what scale

Usage:
    from MacroConditioning import run_full_pipeline
    reg, bkt, mult, region_data = run_full_pipeline(regions=['Brazil','US','EU'])
"""

import os
import json
import pickle
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# ── Column rename map ─────────────────────────────────────────────────────────
MACRO_LABELS = {
    "VIX PX_LAST":    "VIX",
    "XAU PX_LAST":    "Gold",
    "CL1 PX_LAST":    "Oil",
    "DXY PX_LAST":    "DXY",
    "USGG10Y PX_LAST":"UST10Y",
}

# ── All regions available in config.json ─────────────────────────────────────
ALL_REGIONS = ["Brazil", "Chile", "Colombia", "US", "EU", "India",
               "China", "FX Brazil", "FX Mexico"]


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_macro(path: str = "macro_asset.csv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load macro_asset.csv. Returns:
        macro_levels  – daily closing levels, index = date
        macro_changes – 1-day log-changes (Δ), index = date
    """
    macro = pd.read_csv(path, index_col="DATE", parse_dates=True)
    macro.index = [d.date() for d in macro.index]
    macro.rename(columns=MACRO_LABELS, inplace=True)
    macro = macro.apply(pd.to_numeric, errors="coerce").ffill().dropna(how="all")
    changes = np.log(macro).diff()          # log-changes: more symmetric, handles oil spikes
    return macro, changes


def load_region(
    country: str,
    lookback: str = "5Y",
    base_date: dt.date = dt.date(2026, 1, 1),
    cache_dir: str = "cache",
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load (or recompute) a region's PnL, z-scores, and residuals.

    Results are cached in `cache_dir/<country>_data.pkl` so subsequent
    calls are near-instant. Delete the pickle to force a full recompute.

    Returns
    -------
    daily_pnl   pd.Series    – aggregate net P&L per day
    z_scores    pd.DataFrame – rolling z-scores of PCA residuals (one column per tenor)
    residuals   pd.DataFrame – raw PCA residuals (one column per tenor)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{country.replace(' ', '_')}_data.pkl")

    if os.path.exists(cache_file):
        print(f"  [{country}] Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"  [{country}] Computing PCA residuals (this may take a minute)…")
        from RelativeValue2 import RelativeValue

    rv = RelativeValue(country=country, lookback=lookback, base_date=base_date)
    rv.get_residuals()
    rv.update_tenors()

    pos = rv.get_weights(
        confidence=rv.confidence,
        buffer=rv.buffer,
        window=rv.window,
        vol_window=rv.vol_window,
        volatility_target=rv.vt,
        cap=rv.cap,
    )
    pnl_df, _ = rv.compute_pnl(pos=pos, tc=rv.tc, lag=1)
    daily_pnl = pnl_df.sum(axis=1).rename(country)

    z_scores = (
        (rv.residuals - rv.residuals.rolling(rv.window).mean())
        / rv.residuals.rolling(rv.window).std()
    )

    payload = (daily_pnl, z_scores, rv.residuals)
    with open(cache_file, "wb") as f:
        pickle.dump(payload, f)
    print(f"  [{country}] Cached to {cache_file}")
    return payload


# =============================================================================
# 2. TIME-SERIES REGRESSION
# =============================================================================

def run_ts_regression(
    pnl: pd.Series,
    macro_changes: pd.DataFrame,
    lag: int = 1,
) -> pd.DataFrame:
    """
    For each macro factor k run:
        PnL_t = α + β_k * ΔFactor_{k, t-lag} + ε_t

    Uses HC3 heteroskedasticity-robust standard errors.

    Returns DataFrame indexed by factor with columns:
        Beta, T-Stat, P-Value, R-Squared, N
    """
    results = []
    pnl_clean = pd.to_numeric(pnl, errors="coerce").dropna()

    for factor in macro_changes.columns:
        factor_lagged = macro_changes[factor].shift(lag)
        aligned = pd.concat([pnl_clean, factor_lagged], axis=1).dropna()
        aligned.columns = ["pnl", "factor"]

        if len(aligned) < 30:
            continue

        X = sm.add_constant(aligned["factor"])
        model = sm.OLS(aligned["pnl"], X).fit(cov_type="HC3")

        results.append(
            {
                "Factor":    factor,
                "Beta":      model.params.get("factor", np.nan),
                "T-Stat":    model.tvalues.get("factor", np.nan),
                "P-Value":   model.pvalues.get("factor", np.nan),
                "R-Squared": model.rsquared,
                "N":         int(len(aligned)),
            }
        )

    return pd.DataFrame(results).set_index("Factor")


# =============================================================================
# 3. BUCKET (TERCILE) CONDITIONAL SHARPE ANALYSIS
# =============================================================================

def bucket_analysis(
    pnl: pd.Series,
    macro_levels: pd.DataFrame,
    n_buckets: int = 3,
    min_obs_per_bucket: int = 30,
) -> pd.DataFrame:
    """
    Split the sample into `n_buckets` equal-frequency buckets based on the
    *lagged* level of each macro factor, then compute the annualised Sharpe
    ratio and average daily PnL within each bucket.

    This directly answers: "Does the strategy perform better when VIX is
    low / medium / high?" — the rates analog of conditional IC.

    Returns
    -------
    DataFrame indexed by Factor with columns:
        AvgPnL_Low / Mid / High
        Sharpe_Low / Mid / High
        Sharpe_Spread  (High minus Low Sharpe — primary signal)
    """
    labels = ["Low", "Mid", "High"] if n_buckets == 3 else [str(i) for i in range(n_buckets)]
    results = []
    pnl_clean = pd.to_numeric(pnl, errors="coerce").dropna()

    for factor in macro_levels.columns:
        factor_lag = macro_levels[factor].shift(1)
        aligned = pd.concat([pnl_clean, factor_lag], axis=1).dropna()
        aligned.columns = ["pnl", "factor"]

        if len(aligned) < n_buckets * min_obs_per_bucket:
            continue

        try:
            aligned["bucket"] = pd.qcut(
                aligned["factor"], q=n_buckets, labels=labels, duplicates="drop"
            )
        except ValueError:
            continue

        row = {"Factor": factor}
        sharpes = {}
        for lbl in labels:
            g = aligned.loc[aligned["bucket"] == lbl, "pnl"]
            if len(g) < 10:
                row[f"AvgPnL_{lbl}"] = np.nan
                row[f"Sharpe_{lbl}"] = np.nan
            else:
                ann_sharpe = (g.mean() / g.std()) * np.sqrt(252) if g.std() > 0 else np.nan
                row[f"AvgPnL_{lbl}"] = g.mean()
                row[f"Sharpe_{lbl}"] = ann_sharpe
                sharpes[lbl] = ann_sharpe

        if "High" in sharpes and "Low" in sharpes:
            row["Sharpe_Spread"] = sharpes["High"] - sharpes["Low"]
        else:
            row["Sharpe_Spread"] = np.nan

        results.append(row)

    return pd.DataFrame(results).set_index("Factor")


# =============================================================================
# 4. ROLLING BETA
# =============================================================================

def rolling_beta(
    pnl: pd.Series,
    macro_changes: pd.DataFrame,
    window: int = 63,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Compute a `window`-day rolling OLS beta of PnL on each macro factor change.
    Returns a DataFrame of shape (T, n_factors) aligned to pnl.index.
    """
    pnl_clean = pd.to_numeric(pnl, errors="coerce").dropna()
    all_betas = {}

    for factor in macro_changes.columns:
        factor_lag = macro_changes[factor].shift(lag)
        aligned = pd.concat([pnl_clean, factor_lag], axis=1).dropna()
        aligned.columns = ["pnl", "factor"]

        betas, dates = [], []
        for end in range(window, len(aligned) + 1):
            slice_ = aligned.iloc[end - window : end]
            if slice_["factor"].std() == 0:
                betas.append(np.nan)
            else:
                X = sm.add_constant(slice_["factor"])
                m = sm.OLS(slice_["pnl"], X).fit()
                betas.append(m.params.get("factor", np.nan))
            dates.append(aligned.index[end - 1])

        all_betas[factor] = pd.Series(betas, index=dates)

    return pd.DataFrame(all_betas).astype(float)


# =============================================================================
# 5. MULTIPLIER ASSIGNMENT
# =============================================================================

def assign_multipliers(
    regression_results: pd.DataFrame,
    t_threshold: float = 2.0,
    multiplier_map: dict | None = None,
) -> pd.DataFrame:
    """
    Given full-sample regression results, select factors that pass
    `t_threshold` (|t-stat| ≥ threshold) and bucket them by |β| magnitude.

    Bucket → Multiplier
    ───────────────────
    High   → 2.0   (most sensitive factor: amplify signal strongly)
    Mid    → 1.5   (moderate sensitivity)
    Low    → 1.0   (weakly significant but retained for completeness)

    The `Direction` column (+1 / -1) tells you whether to apply the multiplier
    when the factor RISES or FALLS. The rule:
        if (sign of β) × (sign of ΔFactor yesterday) = +1 → favorable → scale up
        otherwise → do not scale (keep multiplier at 1.0)

    Returns
    -------
    DataFrame indexed by Factor with columns:
        Beta, |Beta|, T-Stat, P-Value, R-Squared, Bucket, Multiplier, Direction
    """
    if multiplier_map is None:
        multiplier_map = {"High": 2.0, "Mid": 1.5, "Low": 1.0}

    sig = regression_results[
        regression_results["T-Stat"].abs() >= t_threshold
    ].copy()

    if sig.empty:
        return pd.DataFrame(columns=["Beta", "|Beta|", "T-Stat", "P-Value",
                                      "R-Squared", "Bucket", "Multiplier", "Direction"])

    sig["|Beta|"] = sig["Beta"].abs()
    sig["Direction"] = np.sign(sig["Beta"])

    if len(sig) >= 3:
        sig["Bucket"] = pd.qcut(
            sig["|Beta|"], q=3, labels=["Low", "Mid", "High"], duplicates="drop"
        )
    elif len(sig) == 2:
        sig["Bucket"] = pd.cut(
            sig["|Beta|"], bins=2, labels=["Low", "High"]
        )
    else:
        sig["Bucket"] = "High"

    sig["Multiplier"] = sig["Bucket"].map(multiplier_map)

    return sig[["Beta", "|Beta|", "T-Stat", "P-Value",
                "R-Squared", "Bucket", "Multiplier", "Direction"]]


# =============================================================================
# 6. FULL PIPELINE
# =============================================================================

def run_full_pipeline(
    regions: list[str] | None = None,
    macro_path: str = "macro_asset.csv",
    lookback: str = "5Y",
    base_date: dt.date = dt.date(2026, 1, 1),
    t_threshold: float = 2.0,
    rolling_window: int = 63,
    output_dir: str = "reports/macro_conditioning",
    cache_dir: str = "cache",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    End-to-end macro conditioning pipeline.

    Parameters
    ----------
    regions       : list of country names from config.json (default: all)
    macro_path    : path to macro_asset.csv
    lookback      : PCA lookback window string, e.g. '5Y'
    base_date     : RelativeValue2 base_date
    t_threshold   : |t-stat| cutoff for factor significance
    rolling_window: window (days) for rolling beta estimation
    output_dir    : where to save CSV outputs
    cache_dir     : where to cache heavy RelativeValue2 computations

    Returns
    -------
    reg_summary   : regression β/t-stat table for all region × factor pairs
    bkt_summary   : conditional Sharpe bucket table
    mult_summary  : multiplier recommendations (significant factors only)
    region_data   : dict keyed by country with raw series for plotting
    """
    if regions is None:
        regions = ALL_REGIONS

    os.makedirs(output_dir, exist_ok=True)

    macro_levels, macro_changes = load_macro(macro_path)

    all_reg, all_bkt, all_mult = [], [], []
    region_data: dict = {}

    for country in regions:
        print(f"\n{'='*60}")
        print(f"  {country}")
        print(f"{'='*60}")

        try:
            pnl, z_scores, residuals = load_region(
                country, lookback=lookback, base_date=base_date, cache_dir=cache_dir
            )

            # Align macro to region's trading dates
            macro_l = macro_levels.reindex(pnl.index).ffill()
            macro_c = macro_changes.reindex(pnl.index).ffill()

            # ── Regression ──────────────────────────────────────────────────
            reg = run_ts_regression(pnl, macro_c)
            reg_row = reg.copy()
            reg_row.insert(0, "Region", country)
            all_reg.append(reg_row.reset_index())

            # ── Bucket analysis ──────────────────────────────────────────────
            bkt = bucket_analysis(pnl, macro_l)
            bkt_row = bkt.copy()
            bkt_row.insert(0, "Region", country)
            all_bkt.append(bkt_row.reset_index())

            # ── Multiplier assignment ────────────────────────────────────────
            mult = assign_multipliers(reg, t_threshold=t_threshold)
            if not mult.empty:
                mult_row = mult.copy()
                mult_row.insert(0, "Region", country)
                all_mult.append(mult_row.reset_index())

            # ── Rolling betas ────────────────────────────────────────────────
            rb = rolling_beta(pnl, macro_c, window=rolling_window)

            region_data[country] = {
                "pnl":            pnl,
                "z_scores":       z_scores,
                "residuals":      residuals,
                "regression":     reg,
                "buckets":        bkt,
                "rolling_betas":  rb,
                "macro_levels":   macro_l,
                "macro_changes":  macro_c,
            }

            print(f"  Done. Significant factors (|t|≥{t_threshold}): "
                  f"{reg[reg['T-Stat'].abs() >= t_threshold].index.tolist()}")

        except Exception as exc:
            import traceback
            print(f"  ERROR processing {country}: {exc}")
            traceback.print_exc()

    # ── Compile & Save ───────────────────────────────────────────────────────
    reg_summary  = pd.concat(all_reg,  ignore_index=True) if all_reg  else pd.DataFrame()
    bkt_summary  = pd.concat(all_bkt,  ignore_index=True) if all_bkt  else pd.DataFrame()
    mult_summary = pd.concat(all_mult, ignore_index=True) if all_mult else pd.DataFrame()

    reg_summary.to_csv(f"{output_dir}/regression_summary.csv",  index=False)
    bkt_summary.to_csv(f"{output_dir}/bucket_summary.csv",      index=False)
    mult_summary.to_csv(f"{output_dir}/multiplier_recommendations.csv", index=False)

    print(f"\nAll results saved to: {output_dir}/")
    return reg_summary, bkt_summary, mult_summary, region_data


# =============================================================================
# 7. PLOTTING UTILITIES  (used by notebook)
# =============================================================================

def plot_beta_heatmap(reg_summary: pd.DataFrame, value_col: str = "Beta") -> "go.Figure":
    """
    Heatmap of β or T-Stat for every Region × Factor pair.
    """
    import plotly.graph_objects as go

    pivot = reg_summary.pivot(index="Region", columns="Factor", values=value_col)

    # Symmetric colour scale centred at 0
    abs_max = pivot.abs().max().max()

    text_vals = pivot.applymap(lambda v: f"{v:.3f}" if pd.notna(v) else "")

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=text_vals.values,
            texttemplate="%{text}",
            colorscale="RdBu",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            colorbar=dict(title=value_col),
        )
    )
    title_map = {
        "Beta":      "Factor Beta (PnL sensitivity to 1-day lagged macro change)",
        "T-Stat":    "T-Statistics (|t| ≥ 2 = statistically significant)",
        "R-Squared": "R-Squared of PnL ~ ΔFactor regression",
    }
    fig.update_layout(
        title=title_map.get(value_col, value_col),
        xaxis_title="Macro Factor",
        yaxis_title="Region",
        template="plotly_white",
        height=420,
        width=900,
    )
    return fig


def plot_significance_bars(reg_summary: pd.DataFrame, t_threshold: float = 2.0) -> "go.Figure":
    """
    Grouped bar chart of |T-Stat| per factor, faceted by region.
    A horizontal dashed line marks the significance threshold.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    regions  = reg_summary["Region"].unique()
    factors  = reg_summary["Factor"].unique()
    n_rows   = len(regions)

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=[f"{r}" for r in regions],
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    colours = {f: c for f, c in zip(
        factors, ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3"]
    )}

    for row_idx, region in enumerate(regions, 1):
        sub = reg_summary[reg_summary["Region"] == region]
        for _, r in sub.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[r["Factor"]],
                    y=[abs(r["T-Stat"])],
                    name=r["Factor"],
                    marker_color=colours.get(r["Factor"], "steelblue"),
                    showlegend=(row_idx == 1),
                ),
                row=row_idx, col=1,
            )
        # significance line
        fig.add_hline(
            y=t_threshold, line_dash="dash", line_color="red",
            annotation_text=f"|t|={t_threshold}", annotation_position="top right",
            row=row_idx, col=1,
        )

    fig.update_layout(
        title="Factor T-Statistics by Region",
        template="plotly_white",
        height=220 * n_rows,
        width=900,
        barmode="group",
    )
    return fig


def plot_bucket_sharpes(bkt_summary: pd.DataFrame) -> "go.Figure":
    """
    Grouped bar chart showing conditional Sharpe across Low / Mid / High
    factor level buckets, for every region × factor.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    regions = bkt_summary["Region"].unique()
    n_rows  = len(regions)

    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=list(regions),
        shared_xaxes=True,
        vertical_spacing=0.06,
    )

    bucket_cols   = {"Low": "#EF553B", "Mid": "#FFA15A", "High": "#00CC96"}

    for row_idx, region in enumerate(regions, 1):
        sub = bkt_summary[bkt_summary["Region"] == region].set_index("Factor")
        for lbl, colour in bucket_cols.items():
            col_name = f"Sharpe_{lbl}"
            if col_name not in sub.columns:
                continue
            fig.add_trace(
                go.Bar(
                    x=sub.index.tolist(),
                    y=sub[col_name].tolist(),
                    name=lbl,
                    marker_color=colour,
                    showlegend=(row_idx == 1),
                ),
                row=row_idx, col=1,
            )

    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.update_layout(
        title="Conditional Sharpe Ratio by Factor Bucket (Low / Mid / High Factor Level)",
        template="plotly_white",
        height=240 * n_rows,
        width=950,
        barmode="group",
        yaxis_title="Annualised Sharpe",
    )
    return fig


def plot_rolling_betas(country: str, region_data: dict, t_threshold: float = 2.0) -> "go.Figure":
    """
    Time-series of rolling betas for one region.
    Factors that were significant in the full-sample regression are highlighted.
    """
    import plotly.graph_objects as go

    rb   = region_data[country]["rolling_betas"]
    reg  = region_data[country]["regression"]
    sig  = set(reg[reg["T-Stat"].abs() >= t_threshold].index)

    fig = go.Figure()
    for factor in rb.columns:
        series = rb[factor].dropna()
        is_sig = factor in sig
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=factor,
                line=dict(width=2.5 if is_sig else 1.2,
                          dash="solid" if is_sig else "dot"),
                opacity=1.0 if is_sig else 0.45,
            )
        )

    fig.add_hline(y=0, line_color="black", line_dash="dash", line_width=0.8)
    fig.update_layout(
        title=f"Rolling Factor Betas — {country} (solid = significant in full sample)",
        xaxis_title="Date",
        yaxis_title="Beta",
        template="plotly_white",
        height=450,
        width=950,
    )
    return fig


def plot_multiplier_table(mult_summary: pd.DataFrame) -> "go.Figure":
    """
    Styled Plotly table of multiplier recommendations.
    Rows are colour-coded by Bucket (High → green, Mid → yellow, Low → white).
    """
    import plotly.graph_objects as go

    if mult_summary.empty:
        fig = go.Figure()
        fig.update_layout(title="No factors passed the significance threshold.")
        return fig

    colour_map = {"High": "rgba(0,200,100,0.25)",
                  "Mid":  "rgba(255,200,50,0.20)",
                  "Low":  "rgba(200,200,200,0.15)"}

    row_colors = [colour_map.get(str(b), "white") for b in mult_summary["Bucket"]]

    display_cols = ["Region", "Factor", "Beta", "T-Stat", "P-Value",
                    "R-Squared", "Bucket", "Multiplier", "Direction"]
    display_cols = [c for c in display_cols if c in mult_summary.columns]
    disp = mult_summary[display_cols].copy()
    for col in ["Beta", "T-Stat", "P-Value", "R-Squared", "Multiplier"]:
        if col in disp.columns:
            disp[col] = disp[col].round(4)

    fig = go.Figure(
        go.Table(
            header=dict(
                values=display_cols,
                fill_color="lightgray",
                align="center",
                font=dict(size=12, color="black", family="Arial Black"),
            ),
            cells=dict(
                values=[disp[c].tolist() for c in display_cols],
                fill_color=[row_colors] * len(display_cols),
                align="center",
                font=dict(size=11),
                height=28,
            ),
        )
    )
    fig.update_layout(
        title="Multiplier Recommendations (factors with |t| ≥ threshold)",
        template="plotly_white",
        height=max(300, 60 + 35 * len(mult_summary)),
        width=1000,
    )
    return fig
