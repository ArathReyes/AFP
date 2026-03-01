"""
Run Direct Weight-Based Backtest with CNN+Transformer Weight Generation

Optional: --tc-sweep runs transaction-cost stress test (Section 6) and produces:
  - One combined Alpha Decay plot (Sharpe vs TC, all markets) in Backtest_Results_SpreadsFlys/alpha_decay.png
  - One table (TC rate vs Sharpe per market) in Backtest_Results_SpreadsFlys/alpha_decay_sharpe_table.csv
  - Breakeven_TC_bps in backtest_summary.csv
"""

import os
import sys
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import torch

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from weight_based_backtest import WeightBasedBacktest, run_weight_based_backtest

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_slug(name: str) -> str:
    """
    Create a filesystem-safe, human-readable slug from a sheet or dataset name.
    """
    s = str(name).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "dataset"


def plot_alpha_decay(sharpes: pd.DataFrame, save_path: str) -> None:
    """Alpha Decay: Sharpe ratio vs transaction cost (bps). Colors match reference (US light blue, EU steel blue, LatAm greens, India orange-red, China dark red)."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(12, 8))

    # Match reference image: US light blue, EU darker steel blue, Mexico/Chile/Brazil light→dark green, India orange-red, China dark red/brown
    color_map = {
        "US": "#6baed6",      # light blue
        "EU": "#2171b5",      # steel blue
        "Mexico": "#74c476",  # light green
        "Chile": "#238b45",   # medium green
        "Brazil": "#005a32",  # dark green
        "India": "#e6550d",   # orange-red
        "China": "#a50f15",   # dark red-brown
    }
    # Fallback for any other sheet names
    other_colors = sns.color_palette("Greys_d", 6)
    for i, col in enumerate(sharpes.columns):
        if col not in color_map:
            color_map[col] = other_colors[min(i, 5)]

    # Index is already in bps (0, 0.025, 0.05, ...)
    for country in sharpes.columns:
        plt.plot(
            sharpes.index,
            sharpes[country],
            label=country,
            color=color_map.get(country, other_colors[0]),
            marker="o",
            markevery=max(1, len(sharpes) // 10),
            markersize=5,
            linewidth=2.5,
        )

    plt.axhline(y=0, color="black", linestyle="-", linewidth=1.5, alpha=0.8)
    plt.title("Alpha Decay", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Transaction Cost (bps)", fontsize=14)
    plt.ylabel("Sharpe Ratio", fontsize=14)
    plt.legend(title="Country", loc="lower left", borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Alpha decay plot saved to {save_path}")


def run_cnn_transformer_weight_model(data_path: str, results_tag: str = None, **kwargs) -> Dict[str, Any]:
    """Run CNN+Transformer weight generation model backtest on a single dataset (CSV).
    If results_tag is set, the main results plot is saved as weight_based_backtest_results_<results_tag>.png for LaTeX.
    """
    print("=" * 60)
    print(f"Running CNN+Transformer Weight Generation Model")
    print(f"Data source: {data_path}")
    print("=" * 60)

    # Model configuration
    model_config = {
        'filter_numbers': [16, 32],
        'attention_heads': 4,
        'hidden_units_factor': 2,
        'dropout': 0.25,
        'filter_size': 3,
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 60,
        'patience': 10,
        'weight_decay': 1e-5,
        'step_size': 10,
        'gamma': 0.5,
        'loss_function': 'portfolio_sharpe_classic',  # classic Sharpe (mean/std); use 'portfolio_sharpe' for diff Sharpe
        'weight_mode': 'tanh',  # Options: 'softmax' (long-only), 'tanh' (shorting)
        'look_back': 60,
        'total_notion': 10000,
        # 'transaction_cost': 0.001,
        'transaction_cost': 0.0,
        # 'excluded_curves': ['10y12y']  # List of curves to exclude from the backtest
    }

    # Run weight-based backtest (optionally with transaction-cost sweep for combined alpha-decay)
    results = run_weight_based_backtest(
        data_path,
        model_config,
        results_tag=results_tag,
        run_tc_sweep=kwargs.get("run_tc_sweep", False),
        tc_sweep_grid=kwargs.get("tc_sweep_grid"),
        save_sweep_outputs=kwargs.get("save_sweep_outputs", True),
        realistic_tc=kwargs.get("realistic_tc"),
    )

    return results

def main():
    """Run direct weight-based backtests over each *data* sheet in
    Rates_SpreadsFlys_MR.xlsx and save per-sheet results in clear folders.
    """
    parser = argparse.ArgumentParser(description="Run CNN+Transformer weight-based backtest on spreads/flies.")
    parser.add_argument("--tc-sweep", action="store_true", help="Run transaction-cost stress test and breakeven (Section 6) per market")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    set_random_seeds(42)

    # Simple relative paths from this script location
    # Excel with pre-computed spreads & flies
    excel_path = "../../data/Rates_SpreadsFlys_MR.xlsx"

    # Parent folder to hold all per-dataset backtest runs (under current folder)
    all_backtests_root = "./Backtest_Results_SpreadsFlys"
    os.makedirs(all_backtests_root, exist_ok=True)

    if not os.path.exists(excel_path):
        print(f"ERROR: Spreads & flies workbook not found at {excel_path}.")
        return

    print(f"Found spreads & flies workbook: {excel_path}")
    xls = pd.ExcelFile(excel_path)

    # Process all sheets in the workbook except the diagnostic sheet
    sheets = [sh for sh in xls.sheet_names if sh.lower() != "adf_pvalues"]
    if not sheets:
        print("No data sheets found in Rates_SpreadsFlys_MR.xlsx (excluding 'ADF_pvalues').")
        return

    all_metrics: Dict[str, Dict[str, Any]] = {}
    # When --tc-sweep: collect sweep_df per market for combined alpha-decay table and plot
    sweep_by_market: Dict[str, pd.DataFrame] = {}
    # Fixed TC grid in bps (0, 0.025, ..., 0.5 bps) for table and plot; backtest expects decimal so we pass bps/10000
    ALPHA_DECAY_TC_GRID_BPS = list(np.arange(0, 0.525, 0.025))
    # Realistic transaction cost per market (decimal, e.g. 0.0707e-4 = 0.0707 bps). Used for WithTC metrics.
    TCOSTS_BY_MARKET = {
        "US": 0.0707e-4,
        "EU": 0.0434e-4,
        "Mexico": 0.2182e-4,
        "Chile": 0.3953e-4,
        "Brazil": 0.1667e-4,
        "India": 0.1260e-4,
        "China": 0.1946e-4,
    }

    for sheet in sheets:
        print("\n" + "=" * 80)
        print(f"Preparing dataset from sheet: {sheet}")
        print("=" * 80)

        # Load sheet and keep only numeric columns (spreads & flies)
        df = pd.read_excel(xls, sheet_name=sheet, index_col=0)
        df = df.select_dtypes(include=[np.number]).copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(how="all").ffill().dropna(how="any")

        if df.shape[1] == 0:
            print(f"[SKIP] Sheet '{sheet}' has no usable numeric spread/fly series.")
            continue

        # Name the dataset/folder based on the nature of the data:
        # '<market>_spreads_flys_backtest', where <market> comes from the sheet name.
        sheet_slug = safe_slug(sheet)
        dataset_name = f"{sheet_slug}_spreads_flys_backtest"
        dataset_root = os.path.join(all_backtests_root, dataset_name)
        os.makedirs(dataset_root, exist_ok=True)

        # Save this sheet's time series as a CSV the backtest can consume
        csv_path = os.path.join(dataset_root, "spreads_flys_timeseries.csv")
        df.to_csv(csv_path)
        # When we change cwd into dataset_root we only need the filename
        csv_filename = os.path.basename(csv_path)

        # Ensure models and results live under this dataset folder by
        # temporarily changing cwd while running the backtest.
        models_dir = os.path.join(dataset_root, "models")
        os.makedirs(models_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"RUNNING WEIGHT-BASED BACKTEST FOR DATASET: {dataset_name}")
        print("=" * 80)
        print(f"CSV path: {csv_path}")
        print(f"Output root: {dataset_root}")

        old_cwd = os.getcwd()
        os.chdir(dataset_root)
        try:
            results_tag = None if sheet == "US" else sheet.replace(" ", "_")
            results = run_cnn_transformer_weight_model(
                csv_filename,
                results_tag=results_tag,
                run_tc_sweep=args.tc_sweep,
                tc_sweep_grid=[x / 10000 for x in ALPHA_DECAY_TC_GRID_BPS] if args.tc_sweep else None,
                save_sweep_outputs=False,  # no per-market CSV/plots; we build one table and one plot
                realistic_tc=TCOSTS_BY_MARKET.get(sheet, 0),
            )
        except Exception as e:
            print(f"Error during backtest for sheet '{sheet}': {e}")
            import traceback
            traceback.print_exc()
            results = {}
        finally:
            os.chdir(old_cwd)

        if "metrics" in results:
            metrics = results["metrics"]
            all_metrics[sheet] = dict(metrics)
            if args.tc_sweep and "transaction_cost_sweep" in results:
                tc = results["transaction_cost_sweep"]
                all_metrics[sheet]["breakeven_tc_bps"] = tc.get("breakeven_tc_bps")
                sweep_by_market[sheet] = tc["sweep_df"]
            if "metrics_with_tc" in results:
                all_metrics[sheet]["metrics_with_tc"] = results["metrics_with_tc"]
                all_metrics[sheet]["realistic_tc_bps"] = results.get("realistic_tc_bps")

            print(f"\nKey Results for {dataset_name}:")
            print(f"Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"Annualized Sharpe Ratio (classic): {metrics['annualized_sharpe_ratio']:.3f}")
            print(f"Diff Sharpe (training metric): {metrics['adjusted_diff_sharpe_ratio']:.3f}")
            print(f"Adjusted Ann. Return: {metrics['adjusted_annualized_return']:.4f}")
            print(f"Adjusted Ann. Volatility: {metrics['adjusted_annualized_volatility']:.4f}")
            print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"Final Equity: ${metrics['final_equity']:,.2f}")

    # Simple summary across all sheets
    if all_metrics:
        print("\n" + "=" * 80)
        print("SUMMARY OF WEIGHT-BASED BACKTESTS ACROSS SHEETS")
        print("=" * 80)
        for sheet, metrics in all_metrics.items():
            print(f"\nSheet: {sheet}")
            print(f"  Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"  Annualized Sharpe Ratio (classic): {metrics['annualized_sharpe_ratio']:.3f}")
            print(f"  Diff Sharpe (training metric): {metrics['adjusted_diff_sharpe_ratio']:.3f}")
            print(f"  Adjusted Ann. Return: {metrics['adjusted_annualized_return']:.4f}")
            print(f"  Adjusted Ann. Volatility: {metrics['adjusted_annualized_volatility']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"  Final Equity: ${metrics['final_equity']:,.2f}")

        # Save summary table (NoTC metrics + WithTC metrics when realistic TC is applied) per market
        summary_rows = []
        for sheet, m in all_metrics.items():
            mtc = m.get("metrics_with_tc") or {}
            row = {
                "Market": sheet,
                "Transaction_Cost_bps_Used": m.get("realistic_tc_bps"),
                "Sharpe_Ratio_NoTC": m["annualized_sharpe_ratio"],
                "Sortino_Ratio_NoTC": m.get("sortino_ratio"),
                "Total_Return_Pct_NoTC": m["total_return_pct"],
                "Max_Drawdown_Pct_NoTC": m["max_drawdown_pct"],
                "Win_Rate_Pct_NoTC": m["win_rate_pct"],
                "Profit_Factor_NoTC": m["profit_factor"],
                "Final_Equity_NoTC": m["final_equity"],
                "Sharpe_Ratio_WithTC": mtc.get("annualized_sharpe_ratio_raw") if mtc else None,
                "Sortino_Ratio_WithTC": mtc.get("sortino_ratio") if mtc else None,
                "Total_Return_Pct_WithTC": mtc.get("total_return_pct") if mtc else None,
                "Max_Drawdown_Pct_WithTC": mtc.get("max_drawdown_pct") if mtc else None,
                "Win_Rate_Pct_WithTC": mtc.get("win_rate_pct") if mtc else None,
                "Profit_Factor_WithTC": mtc.get("profit_factor") if mtc else None,
                "Final_Equity_WithTC": mtc.get("final_equity") if mtc else None,
                "Annual_Vol_Pct": m.get("annual_volatility_pct"),
                "Diff_Sharpe_Training_Metric": m["adjusted_diff_sharpe_ratio"],
                "Adjusted_Ann_Return": m["adjusted_annualized_return"],
                "Adjusted_Ann_Volatility": m["adjusted_annualized_volatility"],
                "Breakeven_TC_bps": m.get("breakeven_tc_bps"),
            }
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(all_backtests_root, "backtest_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

        # Combined alpha-decay table and plot (Section 6)
        if args.tc_sweep and sweep_by_market:
            # Table: index = transaction cost (bps), columns = country, values = Sharpe
            common_index_bps = ALPHA_DECAY_TC_GRID_BPS
            sharpes_df = pd.DataFrame(index=common_index_bps)
            for mkt in sweep_by_market:
                df = sweep_by_market[mkt]
                # Rows are in same order as tc_values passed to sweep
                sharpes_df[mkt] = df["annualized_sharpe_raw"].values[: len(common_index_bps)]
            # Legend order: US, EU, Mexico, Chile, Brazil, India, China (then any other markets)
            alpha_decay_legend_order = ["US", "EU", "Mexico", "Chile", "Brazil", "India", "China"]
            ordered = [c for c in alpha_decay_legend_order if c in sharpes_df.columns]
            for c in sharpes_df.columns:
                if c not in ordered:
                    ordered.append(c)
            sharpes_df = sharpes_df[ordered]
            sharpes_df.index.name = "Transaction_Cost_bps"
            table_path = os.path.join(all_backtests_root, "alpha_decay_sharpe_table.csv")
            sharpes_df.to_csv(table_path)
            print(f"Alpha decay table saved to: {table_path}")

            plot_path = os.path.join(all_backtests_root, "alpha_decay.png")
            plot_alpha_decay(sharpes_df, plot_path)
    else:
        print("No backtests completed successfully for any sheet.")

if __name__ == "__main__":
    main()
