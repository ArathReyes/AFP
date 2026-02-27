"""
Run Direct Weight-Based Backtest with CNN+Transformer Weight Generation

This script runs the CNN+Transformer model that generates portfolio weights
for yield spread allocation, using the new weight-based backtesting system
that directly applies model weights without Z-scores or position sizing.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from typing import Dict, Any
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

def run_cnn_transformer_weight_model(data_path: str, **kwargs) -> Dict[str, Any]:
    """Run CNN+Transformer weight generation model backtest on a single dataset (CSV)."""
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
        'loss_function': 'portfolio_sharpe',  # Use portfolio Sharpe ratio loss
        'weight_mode': 'tanh',  # Options: 'softmax' (long-only), 'tanh' (shorting)
        'look_back': 60,
        'total_notion': 10000,
        'transaction_cost': 0.001,
        'excluded_curves': ['10y12y']  # List of curves to exclude from the backtest
    }

    # Run weight-based backtest
    results = run_weight_based_backtest(data_path, model_config)

    return results

def main():
    """Run direct weight-based backtests over each *data* sheet in
    Rates_SpreadsFlys_MR.xlsx and save per-sheet results in clear folders.
    """
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

    # Exclude diagnostic sheet 'ADF_pvalues' - only run on data sheets
    sheets = [sh for sh in xls.sheet_names if sh.lower() != "adf_pvalues"]
    if not sheets:
        print("No data sheets found in Rates_SpreadsFlys_MR.xlsx (excluding 'ADF_pvalues').")
        return

    all_metrics: Dict[str, Dict[str, Any]] = {}

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
            # Use the local CSV filename so relative paths remain correct
            results = run_cnn_transformer_weight_model(csv_filename)
        except Exception as e:
            print(f"Error during backtest for sheet '{sheet}': {e}")
            import traceback
            traceback.print_exc()
            results = {}
        finally:
            os.chdir(old_cwd)

        if "metrics" in results:
            metrics = results["metrics"]
            all_metrics[sheet] = metrics

            print(f"\nKey Results for {dataset_name}:")
            print(f"Total Return: {metrics['total_return_pct']:.2f}%")
            print(f"Adjusted Diff Sharpe Ratio: {metrics['adjusted_diff_sharpe_ratio']:.3f}")
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
            print(f"  Adjusted Diff Sharpe Ratio: {metrics['adjusted_diff_sharpe_ratio']:.3f}")
            print(f"  Adjusted Ann. Return: {metrics['adjusted_annualized_return']:.4f}")
            print(f"  Adjusted Ann. Volatility: {metrics['adjusted_annualized_volatility']:.4f}")
            print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
            print(f"  Final Equity: ${metrics['final_equity']:,.2f}")
    else:
        print("No backtests completed successfully for any sheet.")

if __name__ == "__main__":
    main()
