"""
Weight-Based Backtesting System for CNN+Transformer Weight Generation Model

This backtesting system is specifically designed for models that generate portfolio weights.
It directly applies the trained weights to spread changes without using Z-scores or position sizing.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import os
import re
from datetime import datetime

from base_model import BaseTimeSeriesModel
from cnn_transformer_weight_model import CNNTransformerWeightModel

class WeightBasedBacktest:
    """
    Weight-based backtesting system for portfolio weight generation models.

    Data flow and Sharpe ratio:
    - Raw data: spread/fly levels (e.g. in bps), one column per series.
    - Sequences: X = rolling windows of levels (look_back days); y = next-day change in levels (bps).
    - Training: X and y are StandardScaler'd; model is trained to maximize classic Sharpe (mean/std) of
      portfolio_returns = weights @ y in scaled space.
    - Backtest reporting: We apply the same weights to UNSCALED y (y_test_unscaled, in bps), then
      convert to decimal (bps/10000) for PnL. So portfolio_returns and equity_curve are in real units.
    - Annualized Sharpe = (mean_daily_return / std_daily_return) * sqrt(252), with returns in decimal.
      This yields an interpretable Sharpe (typically in [-2, 3] for real strategies).
    """
    
    def __init__(self, data_path: str, model: BaseTimeSeriesModel, 
                 look_back: int = 16, total_notion: float = 10000,
                 transaction_cost: float = 0.001, excluded_curves: List[str] = None,
                 **model_kwargs):
        """
        Initialize weight-based backtesting system.
        
        Args:
            data_path: Path to the yield spread data CSV file
            model: Trained weight generation model
            look_back: Number of historical days to use for prediction
            total_notion: Total notional amount for portfolio
            transaction_cost: Transaction cost as fraction of position changes
            excluded_curves: List of curve names to exclude from the backtest
            **model_kwargs: Additional model parameters
        """
        self.data_path = data_path
        self.model = model
        self.look_back = look_back
        self.total_notion = total_notion
        self.transaction_cost = transaction_cost
        
        # Load and prepare data
        self.data = pd.read_csv(data_path, index_col=0)
        
        # Filter out excluded curves
        if excluded_curves is None:
            excluded_curves = []
        
        self.excluded_curves = excluded_curves
        if self.excluded_curves:
            print(f"\n{'='*60}")
            print(f"EXCLUDING CURVES FROM BACKTEST:")
            print(f"{'='*60}")
            for curve in self.excluded_curves:
                if curve in self.data.columns:
                    print(f"  - {curve}")
                else:
                    print(f"  - {curve} (WARNING: not found in data)")
            print(f"{'='*60}\n")
            
            # Drop excluded curves from data
            self.data = self.data.drop(columns=[c for c in self.excluded_curves if c in self.data.columns])
        
        self.curve_names = list(self.data.columns)
        self.n_features = len(self.curve_names)
        
        print(f"Loaded data: {self.data.shape}")
        print(f"Spread names: {self.curve_names}")
        
        # Prepare sequences for training and testing
        self.prepare_data()
        
        # Initialize tracking variables
        self.test_weights = None
        self.test_actual = None
        self.portfolio_returns = None
        self.cumulative_returns = None
        self.equity_curve = None
        
        print(f"Initialized weight-based backtest with {self.model.get_model_name()} model")
        print(f"Total notion: ${self.total_notion:,.0f}")
        
    def prepare_data(self) -> None:
        """Prepare train/validation/test splits for weight generation model."""
        # Create sequences
        X_seq, y = self.create_sequences(self.data)
        
        # Split data
        N = len(X_seq)
        val_size = int(N * 0.16)
        train_size = int(N * 0.64)
        test_size = N - train_size - val_size
        
        # Split sequences
        self.X_train = X_seq[:train_size]
        self.y_train = y[:train_size]
        self.X_val = X_seq[train_size:train_size+val_size]
        self.y_val = y[train_size:train_size+val_size]
        self.X_test = X_seq[train_size+val_size:]
        self.y_test = y[train_size+val_size:]
        
        # Store unscaled data for backtesting
        self.X_test_unscaled = self.X_test.copy()
        self.y_test_unscaled = self.y_test.copy()
        
        # Scale data for model training
        from sklearn.preprocessing import StandardScaler
        
        # Separate scalers for features (X) and targets (y)
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # Fit X scaler on training features (spread levels)
        n_train, lb, f = self.X_train.shape
        X_train_flat = self.X_train.reshape(n_train * lb, f)
        self.X_scaler.fit(X_train_flat)
        
        # Fit y scaler on training targets (returns)
        self.y_scaler.fit(self.y_train)
        
        # Transform features with X scaler
        def transform_X(X_seq_block):
            n, lb, f = X_seq_block.shape
            flat = X_seq_block.reshape(n * lb, f)
            flat_scaled = self.X_scaler.transform(flat)
            return flat_scaled.reshape(n, lb, f)
        
        # Transform targets with y scaler
        def transform_y(y_block):
            return self.y_scaler.transform(y_block)
        
        self.X_train = transform_X(self.X_train)
        self.X_val = transform_X(self.X_val)
        self.X_test = transform_X(self.X_test)
        self.y_train = transform_y(self.y_train)
        self.y_val = transform_y(self.y_val)
        self.y_test = transform_y(self.y_test)
        
        print(f"Data split - Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")
        print("Data scaled using StandardScaler")
        
    def create_sequences(self, df: pd.DataFrame) -> tuple:
        """Create sequences for time series prediction with returns as targets."""
        arr = df.to_numpy(dtype=np.float32)
        X_list, y_list = [], []
        
        for i in range(len(arr) - self.look_back):
            X_list.append(arr[i:i+self.look_back])
            
            # Calculate returns as differences between consecutive days
            current = arr[i+self.look_back]  # Current day values
            previous = arr[i+self.look_back-1]  # Previous day values
            
            # Calculate simple differences: (current - previous)
            # These differences are treated as returns directly
            returns = current - previous
            y_list.append(returns)
            
        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)
        return X, y
    
    def train_model(self) -> Dict[str, Any]:
        """Train the weight generation model."""
        print("Training weight generation model...")
        training_history = self.model.fit(
            self.X_train, self.y_train,
            self.X_val, self.y_val
        )
        print("Training completed!")
        return training_history
    
    def generate_weights(self) -> None:
        """Generate portfolio weights using the trained model."""
        print("Generating portfolio weights...")
        
        # Generate weights for test set
        self.test_weights = self.model.predict(self.X_test)
        self.test_actual = self.y_test
        
        print(f"Generated weights shape: {self.test_weights.shape}")
        print(f"Test actual shape: {self.test_actual.shape}")
        
        # Verify weights based on model's weight mode
        weight_sums = np.sum(self.test_weights, axis=1)
        weight_mins = np.min(self.test_weights, axis=1)
        weight_maxs = np.max(self.test_weights, axis=1)
        
        # Get weight mode from model
        weight_mode = getattr(self.model.model, 'weight_mode', 'softmax')
        print(f"Weight mode: {weight_mode}")
        print(f"Weight sums - Min: {weight_sums.min():.6f}, Max: {weight_sums.max():.6f}")
        print(f"Weight range - Min: {weight_mins.min():.6f}, Max: {weight_maxs.max():.6f}")
        
        if weight_mode == "tanh":
            print(f"Number of negative weights: {np.sum(self.test_weights < 0)} / {self.test_weights.size}")
            print(f"Percentage of negative weights: {np.sum(self.test_weights < 0) / self.test_weights.size * 100:.1f}%")
        else:
            print("Long-only portfolio (all weights positive)")
        
    def _n_legs_per_instrument(self) -> np.ndarray:
        """Return number of rate legs per instrument: 2 for spreads, 3 for flies.
        Uses column naming from Rates_SpreadsFlys_MR.xlsx (spread_* and fly_* from process_rates_data.py),
        with fallback to counting tenor-like tokens (e.g. 7y, 10y) in the name.
        """
        n_legs = np.ones(len(self.curve_names), dtype=np.float64) * 2  # default spread
        for i, name in enumerate(self.curve_names):
            s = str(name).strip().lower()
            # Match Excel column names from process_rates_data.py: spread_7y_8y, fly_3y_4y_5y
            if s.startswith("fly_"):
                n_legs[i] = 3
            elif s.startswith("spread_"):
                n_legs[i] = 2
            else:
                # Fallback: count tenor-like tokens (e.g. 7y8y, 3y4y5y or other conventions)
                tenors = re.findall(r"\d+y", s)
                n = len(tenors)
                n_legs[i] = 3 if n >= 3 else 2
        return n_legs

    def calculate_portfolio_returns(self, apply_transaction_cost: bool = True) -> None:
        """
        Calculate portfolio returns by applying weights to spread changes.

        Uses UNSCALED (raw) next-day spread changes so that returns and Sharpe are
        in real units (e.g. bps → decimal for PnL). Training still uses scaled y
        internally; only backtest reporting is in real units.
        """
        print("Calculating portfolio returns...")

        # Use UNSCALED spread changes so portfolio return is in raw units (bps from data)
        returns_bps = self.y_test_unscaled  # shape (n_test, n_features), in bps
        portfolio_returns_bps = np.sum(self.test_weights * returns_bps, axis=1)  # (n_test,) in bps

        print(f"Spread changes (unscaled) - Mean: {np.mean(returns_bps):.4f} bps, Std: {np.std(returns_bps):.4f} bps")
        print(f"Portfolio returns (bps) - Mean: {np.mean(portfolio_returns_bps):.4f}, Std: {np.std(portfolio_returns_bps):.4f}")

        # Convert bps to decimal for PnL: 1 bps = 1e-4
        self.portfolio_returns_bps = portfolio_returns_bps
        self.portfolio_returns = portfolio_returns_bps / 10000.0  # decimal (e.g. 10 bps -> 0.001)

        # Rate-weighted turnover: spread = 2 legs, fly = 3 legs (cost per rate is same)
        weight_changes = np.abs(np.diff(self.test_weights, axis=0, prepend=0))
        n_legs = self._n_legs_per_instrument()  # (n_features,)
        self._daily_turnover = np.sum(weight_changes * n_legs[np.newaxis, :], axis=1)
        self._gross_portfolio_returns = self.portfolio_returns.copy()

        if apply_transaction_cost:
            transaction_costs = self._daily_turnover * self.transaction_cost
            self.portfolio_returns = self.portfolio_returns - transaction_costs
            self.portfolio_returns_bps = self.portfolio_returns_bps - transaction_costs * 10000.0

        self.cumulative_returns = np.cumsum(self.portfolio_returns)
        self.equity_curve = self.total_notion * (1.0 + self.cumulative_returns)

        print(f"Portfolio returns (decimal) - Mean: {np.mean(self.portfolio_returns):.6f}, Std: {np.std(self.portfolio_returns):.6f}")
        print(f"Daily return range (decimal): [{np.min(self.portfolio_returns):.6f}, {np.max(self.portfolio_returns):.6f}]")
        
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.

        Portfolio returns are in decimal (real units: bps/10000). So mean and std
        are in decimal; annualized Sharpe = (mean/std)*sqrt(252) is in the usual range.
        """
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated yet. Run calculate_portfolio_returns first.")

        # Portfolio returns are in DECIMAL (real units: spread change in bps / 10000)
        mean_daily = np.mean(self.portfolio_returns)
        std_daily = np.std(self.portfolio_returns)
        adjusted_annualized_return = mean_daily * 252
        adjusted_annualized_volatility = std_daily * np.sqrt(252)

        lambda_risk = 0.5
        adjusted_diff_sharpe_ratio = adjusted_annualized_return - lambda_risk * adjusted_annualized_volatility

        # Raw Sharpe (can be huge if portfolio vol is tiny)
        if std_daily > 1e-12:
            annualized_sharpe_ratio_raw = (mean_daily / std_daily) * np.sqrt(252)
        else:
            annualized_sharpe_ratio_raw = np.nan

        std_annual_raw = std_daily * np.sqrt(252)
        # Apply low-vol cap only for EU (near-perfect hedge -> tiny std -> huge raw Sharpe). Other markets use raw.
        market = getattr(self, '_market_name', None)
        LOW_VOL_THRESHOLD = 0.02   # 2% annual vol
        MIN_ANNUAL_VOL_FLOOR = 0.05  # 5% floor when capping EU
        if market == 'EU' and std_annual_raw < LOW_VOL_THRESHOLD and std_annual_raw > 1e-12:
            std_annual_used = max(std_annual_raw, MIN_ANNUAL_VOL_FLOOR)
            annualized_sharpe_ratio = (mean_daily * 252) / std_annual_used
            print(f"  [Sharpe] EU: raw annual vol = {std_annual_raw*100:.2f}% (below {LOW_VOL_THRESHOLD*100:.0f}%); using floor {MIN_ANNUAL_VOL_FLOOR*100:.0f}% for reported Sharpe.")
            print(f"  [Sharpe] Unadjusted Sharpe = {annualized_sharpe_ratio_raw:.2f} -> Reported (capped) Sharpe = {annualized_sharpe_ratio:.2f}")
        else:
            annualized_sharpe_ratio = annualized_sharpe_ratio_raw

        if std_daily <= 1e-12:
            annualized_sharpe_ratio = np.nan

        total_return = (self.equity_curve[-1] / self.total_notion - 1) * 100
        
        # Maximum drawdown
        peak = np.maximum.accumulate(self.equity_curve)
        drawdown = (self.equity_curve - peak) / peak
        max_drawdown = np.min(drawdown) * 100
        
        # Additional metrics
        win_rate = np.mean(self.portfolio_returns > 0) * 100
        positive_returns = self.portfolio_returns[self.portfolio_returns > 0]
        negative_returns = self.portfolio_returns[self.portfolio_returns < 0]
        if len(negative_returns) > 0:
            profit_factor = np.sum(positive_returns) / abs(np.sum(negative_returns))
        else:
            profit_factor = float('inf') if len(positive_returns) > 0 else 0.0

        # Sortino ratio: annualized return / annualized downside deviation (std of negative returns only)
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_std_daily = 0.0
        if len(downside_returns) > 0:
            downside_std_daily = np.sqrt(np.mean(downside_returns ** 2))
            if downside_std_daily > 1e-12:
                sortino_ratio = (mean_daily / downside_std_daily) * np.sqrt(252)
            else:
                sortino_ratio = np.nan
        else:
            sortino_ratio = np.nan
        # Apply same EU low-vol cap to Sortino for consistency
        if market == 'EU' and not np.isnan(sortino_ratio) and downside_std_daily > 1e-12:
            downside_annual_raw = downside_std_daily * np.sqrt(252)
            if downside_annual_raw < LOW_VOL_THRESHOLD:
                sortino_ratio = (mean_daily * 252) / max(downside_annual_raw, MIN_ANNUAL_VOL_FLOOR)
        
        # Weight statistics
        weight_stats = {}
        for i, name in enumerate(self.curve_names):
            weight_stats[f'{name}_mean'] = np.mean(self.test_weights[:, i])
            weight_stats[f'{name}_std'] = np.std(self.test_weights[:, i])
            weight_stats[f'{name}_min'] = np.min(self.test_weights[:, i])
            weight_stats[f'{name}_max'] = np.max(self.test_weights[:, i])
            weight_stats[f'{name}_short_pct'] = np.mean(self.test_weights[:, i] < 0) * 100
        
        metrics = {
            'total_return_pct': total_return,
            'adjusted_annualized_return': adjusted_annualized_return,
            'adjusted_annualized_volatility': adjusted_annualized_volatility,
            'adjusted_diff_sharpe_ratio': adjusted_diff_sharpe_ratio,
            'annualized_sharpe_ratio': annualized_sharpe_ratio,
            'annualized_sharpe_ratio_raw': annualized_sharpe_ratio_raw,
            'sortino_ratio': sortino_ratio,
            'annual_volatility_pct': std_annual_raw * 100,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'final_equity': self.equity_curve[-1],
            'num_trades': len(self.portfolio_returns),
            'weight_stats': weight_stats
        }
        
        return metrics

    @staticmethod
    def _metrics_from_returns(returns: np.ndarray, total_notion: float = 10000.0, market: Optional[str] = None) -> Dict[str, Any]:
        """Compute performance metrics from a daily returns series (decimal). Used by TC sweep and for metrics_with_tc.
        Applies low-vol cap only when market=='EU' so alpha-decay table/plot show sensible Sharpe for EU.
        """
        mean_daily = np.mean(returns)
        std_daily = np.std(returns)
        LOW_VOL_THRESHOLD = 0.02
        MIN_ANNUAL_VOL_FLOOR = 0.05
        if std_daily <= 1e-12:
            sharpe_raw = np.nan
            vol_pct = 0.0
        else:
            vol_pct = std_daily * np.sqrt(252) * 100
            std_annual_raw = std_daily * np.sqrt(252)
            sharpe_uncapped = (mean_daily / std_daily) * np.sqrt(252)
            if market == 'EU' and std_annual_raw < LOW_VOL_THRESHOLD:
                std_annual_used = max(std_annual_raw, MIN_ANNUAL_VOL_FLOOR)
                sharpe_raw = (mean_daily * 252) / std_annual_used
            else:
                sharpe_raw = sharpe_uncapped
        # Sortino: annualized return / annualized downside deviation
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.sqrt(np.mean(downside ** 2))
            if downside_std > 1e-12:
                sortino = (mean_daily / downside_std) * np.sqrt(252)
                if market == 'EU':
                    dd_annual = downside_std * np.sqrt(252)
                    if dd_annual < LOW_VOL_THRESHOLD:
                        sortino = (mean_daily * 252) / max(dd_annual, MIN_ANNUAL_VOL_FLOOR)
            else:
                sortino = np.nan
        else:
            sortino = np.nan
        cum = np.cumsum(returns)
        equity = total_notion * (1.0 + cum)
        total_return_pct = (equity[-1] / total_notion - 1) * 100
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.clip(peak, 1e-12, None)
        max_drawdown_pct = np.min(drawdown) * 100
        win_rate_pct = np.mean(returns > 0) * 100
        pos = returns[returns > 0].sum()
        neg = returns[returns < 0].sum()
        profit_factor = pos / abs(neg) if neg != 0 else (float('inf') if pos > 0 else 0.0)
        return {
            'total_return_pct': total_return_pct,
            'annualized_sharpe_ratio_raw': sharpe_raw,
            'sortino_ratio': sortino,
            'annual_volatility_pct': vol_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'win_rate_pct': win_rate_pct,
            'profit_factor': profit_factor,
            'final_equity': equity[-1],
        }

    def run_transaction_cost_sweep(
        self,
        tc_values: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        save_outputs: bool = True,
        market: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Stress-test transaction costs and compute breakeven (Section 6).
        R_net_t = R_t - psi * Turnover_t; breakeven psi such that sum(R_net) = 0.
        When save_outputs is True, saves sweep CSV and plots; when False, only returns sweep_df and breakeven (for combined alpha-decay table/plot).
        """
        if not hasattr(self, '_gross_portfolio_returns') or self._gross_portfolio_returns is None:
            raise ValueError("Run backtest first (e.g. run_backtest(apply_transaction_cost=False)).")
        gross = self._gross_portfolio_returns
        turnover = self._daily_turnover
        total_turnover = np.sum(turnover)
        sum_gross = np.sum(gross)

        # Exact breakeven from Section 6: sum(R_t) = psi_BE * sum(Turnover_t) => psi_BE = sum(R_t) / sum(Turnover_t)
        breakeven_tc = None
        breakeven_tc_bps = None
        if total_turnover > 1e-12:
            breakeven_tc = sum_gross / total_turnover
            breakeven_tc_bps = breakeven_tc * 10000

        if tc_values is None:
            # Grid from 0 up to past breakeven (or 0.01 if no breakeven)
            max_tc = breakeven_tc * 1.2 if breakeven_tc is not None and breakeven_tc > 0 else 0.01
            tc_values = list(np.linspace(0, min(max_tc, 0.01), 80))
        rows = []
        for tc in tc_values:
            net_returns = gross - turnover * tc
            m = self._metrics_from_returns(net_returns, self.total_notion, market=market)
            rows.append({
                'transaction_cost': tc,
                'tc_bps': tc * 10000,
                'total_return_pct': m['total_return_pct'],
                'annualized_sharpe_raw': m['annualized_sharpe_ratio_raw'],
                'max_drawdown_pct': m['max_drawdown_pct'],
                'win_rate_pct': m['win_rate_pct'],
                'profit_factor': m['profit_factor'],
                'final_equity': m['final_equity'],
            })
        sweep_df = pd.DataFrame(rows)

        result = {
            'sweep_df': sweep_df,
            'breakeven_tc': breakeven_tc,
            'breakeven_tc_bps': breakeven_tc_bps,
        }
        if not save_outputs:
            return result

        os.makedirs('results', exist_ok=True)
        csv_path = save_path if save_path and save_path.endswith('.csv') else os.path.join('results', 'transaction_cost_sweep.csv')
        sweep_df.to_csv(csv_path, index=False)
        print(f"Transaction cost sweep saved to {csv_path}")

        # ---- Plot 1: P&L decrease to breakeven (standalone figure) ----
        fig1, ax1 = plt.subplots(1, 1, figsize=(9, 5))
        tc_bps = sweep_df['tc_bps'].values
        ret_pct = sweep_df['total_return_pct'].values
        ax1.plot(tc_bps, ret_pct, 'b-o', markersize=3, label='Net total return (%)')
        ax1.axhline(0, color='gray', linestyle='--')
        if breakeven_tc_bps is not None and breakeven_tc_bps > 0:
            ax1.axvline(breakeven_tc_bps, color='red', linestyle=':', alpha=0.9, label=f'Breakeven = {breakeven_tc_bps:.1f} bps')
        ax1.set_xlabel('Transaction cost (bps, per rate)')
        ax1.set_ylabel('Total return (%)')
        ax1.set_title('P&L changing with transaction cost rate')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        pnl_plot_path = csv_path.replace('.csv', '_pnl_to_breakeven.png') if save_path and save_path.endswith('.csv') else os.path.join('results', 'transaction_cost_pnl_to_breakeven.png')
        plt.savefig(pnl_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"P&L-to-breakeven plot saved to {pnl_plot_path}")

        # ---- Plot 2: Two-panel sweep (Total Return + Sharpe vs TC) ----
        fig2, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1_2, ax2_2 = axes
        ax1_2.plot(tc_bps, ret_pct, 'b-o', markersize=4)
        ax1_2.axhline(0, color='gray', linestyle='--')
        if breakeven_tc_bps is not None:
            ax1_2.axvline(breakeven_tc_bps, color='red', linestyle=':', alpha=0.8, label=f'Breakeven ≈ {breakeven_tc_bps:.1f} bps')
        ax1_2.set_ylabel('Total Return (%)')
        ax1_2.set_title('Transaction cost stress test: Total return vs. TC')
        ax1_2.legend(loc='best')
        ax1_2.grid(True, alpha=0.3)
        ax2_2.plot(tc_bps, sweep_df['annualized_sharpe_raw'], 'g-o', markersize=4)
        ax2_2.axhline(0, color='gray', linestyle='--')
        ax2_2.set_xlabel('Transaction cost (bps, per rate)')
        ax2_2.set_ylabel('Annualized Sharpe (raw)')
        ax2_2.set_title('Transaction cost stress test: Sharpe vs. TC')
        ax2_2.grid(True, alpha=0.3)
        plt.tight_layout()
        sweep_plot_path = csv_path.replace('.csv', '.png') if save_path and save_path.endswith('.csv') else os.path.join('results', 'transaction_cost_sweep.png')
        plt.savefig(sweep_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Transaction cost sweep plot saved to {sweep_plot_path}")

        print("\nTransaction cost sweep summary (Section 6 breakeven):")
        if breakeven_tc_bps is not None:
            print(f"  Breakeven TC (per rate, decimal): {breakeven_tc:.6f}  =>  {breakeven_tc_bps:.1f} bps")
        else:
            print("  Breakeven TC: strategy unprofitable at tc=0 or zero turnover")
        result['csv_path'] = csv_path
        result['plot_path'] = sweep_plot_path
        result['pnl_to_breakeven_plot_path'] = pnl_plot_path
        return result

    def run_backtest(self, apply_transaction_cost: bool = True) -> Dict[str, Any]:
        """Run the complete weight-based backtest."""
        print("="*80)
        print("WEIGHT-BASED BACKTEST")
        print("="*80)
        
        # Train model
        training_history = self.train_model()
        
        # Plot training loss evolution
        print("\nPlotting training loss evolution...")
        self.plot_training_loss(training_history)
        
        # Generate weights
        self.generate_weights()
        
        # Calculate portfolio returns
        self.calculate_portfolio_returns(apply_transaction_cost)
        
        # Run diagnostic analysis
        print("\nRunning diagnostic analysis...")
        self.analyze_spread_changes_and_weights()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics()
        
        # Print results
        self.print_results(metrics)
        
        return {
            'metrics': metrics,
            'training_history': training_history,
            'portfolio_returns': self.portfolio_returns,
            'equity_curve': self.equity_curve,
            'test_weights': self.test_weights,
            'test_actual': self.test_actual
        }
    
    def print_results(self, metrics: Dict[str, Any]) -> None:
        """Print backtest results."""
        print("\n" + "="*80)
        print("WEIGHT-BASED BACKTEST RESULTS")
        print("="*80)
        
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"Annualized Return (scaled units): {metrics['adjusted_annualized_return']:.4f}")
        print(f"Annualized Volatility (scaled units): {metrics['adjusted_annualized_volatility']:.4f}")
        print(f"Diff Sharpe (training metric, scaled units): {metrics['adjusted_diff_sharpe_ratio']:.3f}")
        print(f"Annualized Sharpe Ratio (classic mean/std): {metrics['annualized_sharpe_ratio']:.3f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {metrics['win_rate_pct']:.1f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Final Equity: ${metrics['final_equity']:,.2f}")
        
        print("\n" + "="*50)
        print("DEBUGGING INFORMATION")
        print("="*50)
        print(f"Number of trading days: {len(self.portfolio_returns)}")
        print(f"Portfolio returns range: [{np.min(self.portfolio_returns):.6f}, {np.max(self.portfolio_returns):.6f}]")
        print(f"Portfolio returns mean: {np.mean(self.portfolio_returns):.6f}")
        print(f"Portfolio returns std: {np.std(self.portfolio_returns):.6f}")
        print(f"Total notion: ${self.total_notion:,.0f}")
        print(f"Final cumulative return: {self.cumulative_returns[-1]:.6f}")
        
        print("\n" + "="*60)
        print("WEIGHT STATISTICS (Signed Weights with Shorting)")
        print("="*60)
        for name in self.curve_names:
            mean_weight = metrics['weight_stats'][f'{name}_mean']
            std_weight = metrics['weight_stats'][f'{name}_std']
            min_weight = metrics['weight_stats'][f'{name}_min']
            max_weight = metrics['weight_stats'][f'{name}_max']
            short_pct = metrics['weight_stats'][f'{name}_short_pct']
            print(f"{name:>8}: Mean={mean_weight:.3f}, Std={std_weight:.3f}, Range=[{min_weight:.3f}, {max_weight:.3f}], Short%={short_pct:.1f}%")
    
    def plot_training_loss(self, training_history: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """Plot training loss evolution to monitor Sharpe ratio optimization."""
        if 'train_loss' not in training_history:
            print("No training history available for plotting.")
            return
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Training Loss Evolution (Negative Sharpe Ratio)', fontsize=16)
        
        train_losses = training_history['train_loss']
        val_losses = training_history.get('val_loss', [])
        epochs = range(1, len(train_losses) + 1)
        
        # Plot: Training and validation loss (Negative Sharpe Ratio)
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        if val_losses:
            ax.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (Negative Sharpe Ratio)')
        ax.set_title('Training Loss Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add annotations for best performance
        if val_losses:
            best_val_epoch = np.argmin(val_losses) + 1
            best_val_loss = min(val_losses)
            ax.annotate(f'Best Val: {best_val_loss:.4f}\n(Epoch {best_val_epoch})',
                           xy=(best_val_epoch, best_val_loss), xytext=(10, 10),
                           textcoords='offset points', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        # Default save path if not provided
        if save_path is None:
            save_path = "results/training_loss_evolution.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training loss plot saved to: {save_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("TRAINING LOSS ANALYSIS")
        print("="*60)
        print(f"Initial Training Loss: {train_losses[0]:.6f} (Sharpe: {-train_losses[0]:.6f})")
        print(f"Final Training Loss: {train_losses[-1]:.6f} (Sharpe: {-train_losses[-1]:.6f})")
        print(f"Loss Improvement: {train_losses[0] - train_losses[-1]:.6f}")
        
        if val_losses:
            print(f"Initial Validation Loss: {val_losses[0]:.6f} (Sharpe: {-val_losses[0]:.6f})")
            print(f"Best Validation Loss: {min(val_losses):.6f} (Sharpe: {-min(val_losses):.6f})")
            print(f"Final Validation Loss: {val_losses[-1]:.6f} (Sharpe: {-val_losses[-1]:.6f})")
        
        # Check if model is learning
        if train_losses[-1] < train_losses[0]:
            print("[PASS] Model is learning: Training loss decreased")
        else:
            print("[FAIL] Model may not be learning: Training loss increased or stayed same")
        
        if val_losses and min(val_losses) < val_losses[0]:
            print("[PASS] Generalization: Validation loss improved")
        elif val_losses:
            print("[FAIL] Potential overfitting: Validation loss did not improve")
        
        plt.show()

    def analyze_spread_changes_and_weights(self) -> None:
        """Analyze spread changes and weights to diagnose potential issues."""
        if self.test_weights is None or self.y_test_unscaled is None:
            print("Cannot analyze: weights or spread changes not available.")
            return
        
        print("\n" + "="*80)
        print("DIAGNOSTIC ANALYSIS: SPREAD CHANGES AND WEIGHTS")
        print("="*80)
        
        # Analyze spread differences (targets)
        spread_differences = self.y_test_unscaled
        print(f"Spread Differences Analysis:")
        print(f"  Shape: {spread_differences.shape}")
        print(f"  Mean: {np.mean(spread_differences):.6f} bps")
        print(f"  Std: {np.std(spread_differences):.6f} bps")
        print(f"  Min: {np.min(spread_differences):.6f} bps")
        print(f"  Max: {np.max(spread_differences):.6f} bps")
        print(f"  Contains NaN: {np.isnan(spread_differences).any()}")
        print(f"  Contains Inf: {np.isinf(spread_differences).any()}")
        
        # Check for extreme values (in basis points)
        extreme_threshold = 5.0  # 5 basis points is a reasonable threshold for daily changes
        extreme_count = np.sum(np.abs(spread_differences) > extreme_threshold)
        print(f"  Extreme differences (>5 bps): {extreme_count} / {spread_differences.size} ({extreme_count/spread_differences.size*100:.1f}%)")
        
        # Analyze weights
        print(f"\nWeights Analysis:")
        print(f"  Shape: {self.test_weights.shape}")
        print(f"  Mean: {np.mean(self.test_weights):.6f}")
        print(f"  Std: {np.std(self.test_weights):.6f}")
        print(f"  Min: {np.min(self.test_weights):.6f}")
        print(f"  Max: {np.max(self.test_weights):.6f}")
        print(f"  Contains NaN: {np.isnan(self.test_weights).any()}")
        print(f"  Contains Inf: {np.isinf(self.test_weights).any()}")
        
        # Check weight sums (should be close to 1 for softmax)
        weight_sums = np.sum(self.test_weights, axis=1)
        print(f"  Weight sums - Mean: {np.mean(weight_sums):.6f}, Std: {np.std(weight_sums):.6f}")
        print(f"  Weight sums - Min: {np.min(weight_sums):.6f}, Max: {np.max(weight_sums):.6f}")
        
        # Analyze portfolio returns calculation
        if self.portfolio_returns is not None:
            print(f"\nPortfolio Returns Analysis:")
            print(f"  Shape: {self.portfolio_returns.shape}")
            print(f"  Mean: {np.mean(self.portfolio_returns):.6f}")
            print(f"  Std: {np.std(self.portfolio_returns):.6f}")
            print(f"  Min: {np.min(self.portfolio_returns):.6f}")
            print(f"  Max: {np.max(self.portfolio_returns):.6f}")
            
            # Check for extreme portfolio returns (in basis points)
            extreme_portfolio_threshold = 0.001  # 10 bps in decimal (10/10000)
            extreme_portfolio_count = np.sum(np.abs(self.portfolio_returns) > extreme_portfolio_threshold)
            print(f"  Extreme portfolio returns (>10 bps): {extreme_portfolio_count} / {len(self.portfolio_returns)} ({extreme_portfolio_count/len(self.portfolio_returns)*100:.1f}%)")
            
            # Show worst days
            worst_days_idx = np.argsort(self.portfolio_returns)[:5]
            best_days_idx = np.argsort(self.portfolio_returns)[-5:]
            
            print(f"\n  Worst 5 days:")
            for i, idx in enumerate(worst_days_idx):
                print(f"    Day {idx}: Return={self.portfolio_returns[idx]:.6f}")
                print(f"      Weights: {self.test_weights[idx]}")
                print(f"      Spread differences: {spread_differences[idx]}")
            
            print(f"\n  Best 5 days:")
            for i, idx in enumerate(best_days_idx):
                print(f"    Day {idx}: Return={self.portfolio_returns[idx]:.6f}")
                print(f"      Weights: {self.test_weights[idx]}")
                print(f"      Spread differences: {spread_differences[idx]}")
        
        # Check for potential issues
        print(f"\n" + "="*60)
        print("POTENTIAL ISSUES DETECTED:")
        print("="*60)
        
        issues_found = False
        
        if np.isnan(spread_differences).any() or np.isinf(spread_differences).any():
            print("[ERROR] Spread differences contain NaN or Inf values!")
            issues_found = True
        
        if np.isnan(self.test_weights).any() or np.isinf(self.test_weights).any():
            print("[ERROR] Weights contain NaN or Inf values!")
            issues_found = True
        
        if np.abs(np.mean(weight_sums) - 1.0) > 0.01:
            print(f"[WARN] Weight sums deviate significantly from 1.0 (mean: {np.mean(weight_sums):.6f})")
            issues_found = True
        
        if extreme_count > spread_differences.size * 0.05:  # More than 5% extreme values
            print(f"[WARN] Too many extreme spread differences ({extreme_count/spread_differences.size*100:.1f}%)")
            issues_found = True
        
        if self.portfolio_returns is not None and np.abs(np.mean(self.portfolio_returns)) > 0.0001:  # 1 bps mean in decimal
            print(f"[WARN] Portfolio returns have high mean ({np.mean(self.portfolio_returns):.6f} decimal, {np.mean(self.portfolio_returns)*10000:.2f} bps), suggesting bias")
            issues_found = True
        
        if not issues_found:
            print("[PASS] No obvious issues detected in the data.")
        
        # Additional analysis: Rolling Sharpe ratio performance
        print(f"\n" + "="*60)
        print("ROLLING SHARPE RATIO ANALYSIS (60-day window)")
        print("="*60)
        
        if self.portfolio_returns is not None and len(self.portfolio_returns) >= 60:
            # Use portfolio returns directly for Sharpe calculation
            rolling_returns = pd.Series(self.portfolio_returns).rolling(60)
            rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
            
            # Remove NaN values (first 59 days)
            valid_sharpe = rolling_sharpe.dropna()
            
            if len(valid_sharpe) > 0:
                print(f"Rolling Sharpe Statistics:")
                print(f"  Mean: {valid_sharpe.mean():.3f}")
                print(f"  Std: {valid_sharpe.std():.3f}")
                print(f"  Min: {valid_sharpe.min():.3f}")
                print(f"  Max: {valid_sharpe.max():.3f}")
                print(f"  Median: {valid_sharpe.median():.3f}")
                
                # Performance categories
                positive_sharpe_pct = (valid_sharpe > 0).mean() * 100
                good_sharpe_pct = (valid_sharpe > 1).mean() * 100
                excellent_sharpe_pct = (valid_sharpe > 2).mean() * 100
                
                print(f"\nRolling Sharpe Performance:")
                print(f"  Positive Sharpe (>0): {positive_sharpe_pct:.1f}% of time")
                print(f"  Good Sharpe (>1): {good_sharpe_pct:.1f}% of time")
                print(f"  Excellent Sharpe (>2): {excellent_sharpe_pct:.1f}% of time")
                
                # Expected vs Actual
                print(f"\nTraining vs Out-of-Sample Comparison:")
                print(f"  Training Sharpe (best): +201.8255 (unrealistically high)")
                print(f"  Out-of-sample mean Sharpe: {valid_sharpe.mean():.3f}")
                print(f"  Performance gap: {201.8255 - valid_sharpe.mean():.1f}")
                
                if valid_sharpe.mean() > 0.5:
                    print("[PASS] Model shows decent Sharpe performance out-of-sample")
                elif valid_sharpe.mean() > 0:
                    print("[WARN] Model shows weak but positive Sharpe performance")
                else:
                    print("[FAIL] Model shows poor Sharpe performance out-of-sample")
            else:
                print("[WARN] Not enough data for rolling Sharpe analysis")
        else:
            print("[ERROR] Insufficient portfolio returns for rolling Sharpe analysis")

    def plot_combined_spreads(self, save_path: Optional[str] = None) -> None:
        """Plot all 8 spreads and flies combined in one plot for comparison."""
        if self.y_test_unscaled is None:
            raise ValueError("Backtest not run yet. Run run_backtest first.")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Get the original raw data for the test period
        N = len(self.data)
        val_size = int((N - self.look_back) * 0.16)
        train_size = int((N - self.look_back) * 0.64)
        test_start_idx = train_size + val_size + self.look_back
        
        # Raw data for test period (the actual spread/fly levels)
        raw_test_data = self.data.iloc[test_start_idx:test_start_idx + len(self.y_test_unscaled)].values
        
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle('All Spreads and Flies Comparison (Out-of-Sample Period)', fontsize=16)
        
        # Define a color palette for better visualization
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.curve_names)))
        
        for i, name in enumerate(self.curve_names):
            raw_values = raw_test_data[:, i]
            ax.plot(raw_values, label=name, linewidth=1.5, alpha=0.8, color=colors[i])
        
        ax.set_title('All Spreads and Flies Values Over Time', fontsize=14)
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Level (bps)', fontsize=12)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Default save path if not provided
        if save_path is None:
            save_path = "results/combined_spreads_flies.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined spreads/flies plot saved to: {save_path}")
        
        plt.show()
    
    def plot_individual_spreads(self, save_path: Optional[str] = None) -> None:
        """Plot all 8 individual spreads and flies from out-of-sample period."""
        if self.y_test_unscaled is None:
            raise ValueError("Backtest not run yet. Run run_backtest first.")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Get the original raw data for the test period
        N = len(self.data)
        val_size = int((N - self.look_back) * 0.16)
        train_size = int((N - self.look_back) * 0.64)
        test_start_idx = train_size + val_size + self.look_back
        
        # Raw data for test period (the actual spread/fly levels)
        raw_test_data = self.data.iloc[test_start_idx:test_start_idx + len(self.y_test_unscaled)].values
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle('Individual Spreads and Flies (Out-of-Sample Period)', fontsize=16)
        
        for i, name in enumerate(self.curve_names):
            if i >= len(self.curve_names):
                break
                
            row = i // 2
            col = i % 2
            
            if row < 4:  # Make sure we don't exceed subplot bounds
                # Plot raw spread/fly levels
                raw_values = raw_test_data[:, i]
                axes[row, col].plot(raw_values, 'b-', linewidth=1.5, alpha=0.8)
                axes[row, col].set_title(f'{name}', fontsize=14)
                axes[row, col].set_xlabel('Trading Days')
                axes[row, col].set_ylabel('Level (bps)')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add statistics box
                mean_val = np.mean(raw_values)
                std_val = np.std(raw_values)
                min_val = np.min(raw_values)
                max_val = np.max(raw_values)
                
                stats_text = f'Mean: {mean_val:.1f} bps\nStd: {std_val:.1f} bps\nRange: [{min_val:.1f}, {max_val:.1f}]'
                axes[row, col].text(0.02, 0.98, stats_text,
                    transform=axes[row, col].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=10)
        
        plt.tight_layout()
        
        # Default save path if not provided
        if save_path is None:
            save_path = "results/individual_spreads_flies.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual spreads/flies plot saved to: {save_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("INDIVIDUAL SPREADS/FLIES SUMMARY")
        print("="*60)
        
        for i, name in enumerate(self.curve_names):
            if i >= len(raw_test_data[0]):
                break
                
            raw_values = raw_test_data[:, i]
            differences = self.y_test_unscaled[:, i]
            
            print(f"\n{name}:")
            print(f"  Level - Mean: {np.mean(raw_values):.2f} bps, Std: {np.std(raw_values):.2f} bps")
            print(f"  Range: [{np.min(raw_values):.1f}, {np.max(raw_values):.1f}] bps")
            print(f"  Daily Changes - Mean: {np.mean(differences):.3f} bps, Std: {np.std(differences):.3f} bps")
            print(f"  Daily Range: [{np.min(differences):.2f}, {np.max(differences):.2f}] bps")
        
        plt.show()

    def plot_results(self, save_path: Optional[str] = None, omit_legends_for_many_curves: bool = False) -> None:
        """Plot comprehensive backtest results (4 subplots: Portfolio Value, All Spreads and Flies, Daily Returns, Returns Distribution).
        If omit_legends_for_many_curves is True, the 'All Spreads and Flies Comparison' is drawn without legend (e.g. China, India).
        Rolling Sharpe Ratio and Portfolio Weight Evolution subplots are commented out.
        """
        if self.portfolio_returns is None:
            raise ValueError("Backtest not run yet. Run run_backtest first.")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Weight-Based Backtest Results - {self.model.get_model_name()}', fontsize=16)
        
        # 1. Portfolio value curve with performance metrics
        axes[0, 0].plot(self.equity_curve)
        axes[0, 0].set_title('Portfolio Value Curve')
        axes[0, 0].set_xlabel('Trading Days')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Calculate and display key performance metrics
        # Use portfolio returns directly for display metrics (adjusted units)
        adjusted_annualized_return = np.mean(self.portfolio_returns) * 252  # Adjusted annualized return
        adjusted_annualized_volatility = np.std(self.portfolio_returns) * np.sqrt(252)  # Adjusted annualized volatility
        # Adjusted Diff Sharpe ratio calculation consistent with training
        lambda_risk = 0.5  # Same risk aversion parameter as training
        adjusted_diff_sharpe_ratio = adjusted_annualized_return - lambda_risk * adjusted_annualized_volatility

        # Add performance metrics text box in upper left corner
        metrics_text = f'Differentiable Sharpe Ratio: {adjusted_diff_sharpe_ratio:.3f}\n'
        metrics_text += f'Ann. Return: {adjusted_annualized_return:.4f}\n'
        metrics_text += f'Ann. Volatility: {adjusted_annualized_volatility:.4f}'

        # axes[0, 0].text(0.02, 0.98, metrics_text, transform=axes[0, 0].transAxes,
        #                verticalalignment='top', horizontalalignment='left',
        #                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
        #                fontsize=11, fontweight='bold')
        
        # 2. All Spreads and Flies Comparison (Weighted Contributions)
        # Get the original raw data for the test period
        N = len(self.data)
        val_size = int((N - self.look_back) * 0.16)
        train_size = int((N - self.look_back) * 0.64)
        test_start_idx = train_size + val_size + self.look_back
        
        # Raw data for test period (the actual spread/fly levels)
        raw_test_data = self.data.iloc[test_start_idx:test_start_idx + len(self.y_test_unscaled)].values
        
        # Define a color palette for better visualization
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.curve_names)))
        
        # Calculate weighted contribution for each spread/fly
        for i, name in enumerate(self.curve_names):
            # Get spread changes (returns) for this instrument
            spread_changes = self.y_test_unscaled[:, i]
            # Get weights for this instrument
            weights = self.test_weights[:, i]
            # Calculate weighted returns for this instrument
            weighted_returns = weights * spread_changes
            cumulative_weighted_bps = np.cumsum(weighted_returns)
            portfolio_contribution = self.total_notion * (cumulative_weighted_bps / 10000.0)
            
            axes[0, 1].plot(portfolio_contribution, label=name, linewidth=1.5, alpha=0.8, color=colors[i])
        
        axes[0, 1].set_title('All Spreads and Flies Comparison (Weighted Contributions)')
        axes[0, 1].set_xlabel('Trading Days')
        axes[0, 1].set_ylabel('Cumulative Weighted Contribution ($)')
        if not omit_legends_for_many_curves:
            axes[0, 1].legend(loc='best', fontsize=8, framealpha=0.9)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # 3. Daily returns
        axes[1, 0].plot(self.portfolio_returns)
        axes[1, 0].set_title('Daily Portfolio Returns')
        axes[1, 0].set_xlabel('Trading Days')
        axes[1, 0].set_ylabel('Daily Return')
        axes[1, 0].grid(True)
        
        # 4. Returns distribution
        axes[1, 1].hist(self.portfolio_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Returns Distribution')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        # 5. Rolling Sharpe ratio (commented out per user request)
        # window = 60
        # rolling_returns = pd.Series(self.portfolio_returns).rolling(window)
        # rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
        # axes[2, 0].plot(rolling_sharpe)
        # axes[2, 0].set_title(f'Rolling Sharpe Ratio ({window}-day window)')
        # ...
        
        # 6. Portfolio Weight Evolution (commented out per user request)
        # for i, name in enumerate(self.curve_names):
        #     axes[2, 1].plot(self.test_weights[:, i], label=name, alpha=0.7)
        # axes[2, 1].set_title('Portfolio Weight Evolution')
        # ...
        
        plt.tight_layout()
        
        # Default save path if not provided
        if save_path is None:
            save_path = "results/weight_based_backtest_results.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to: {save_path}")
        
        plt.show()
    
    def plot_weight_analysis(self, save_path: Optional[str] = None) -> None:
        """Plot detailed weight analysis."""
        if self.test_weights is None:
            raise ValueError("Weights not generated yet. Run generate_weights first.")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Weight Analysis', fontsize=16)
        
        # 1. Average weights
        avg_weights = np.mean(self.test_weights, axis=0)
        bars = axes[0, 0].bar(self.curve_names, avg_weights)
        axes[0, 0].set_title('Average Portfolio Weights')
        axes[0, 0].set_ylabel('Weight')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, weight in zip(bars, avg_weights):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{weight:.3f}', ha='center', va='bottom')
        
        # 2. Weight correlation matrix
        weight_corr = np.corrcoef(self.test_weights.T)
        im = axes[0, 1].imshow(weight_corr, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_title('Weight Correlation Matrix')
        axes[0, 1].set_xticks(range(len(self.curve_names)))
        axes[0, 1].set_yticks(range(len(self.curve_names)))
        axes[0, 1].set_xticklabels(self.curve_names, rotation=45)
        axes[0, 1].set_yticklabels(self.curve_names)
        
        # Add correlation values
        for i in range(len(self.curve_names)):
            for j in range(len(self.curve_names)):
                axes[0, 1].text(j, i, f'{weight_corr[i, j]:.2f}',
                               ha='center', va='center', color='black')
        
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. Weight volatility
        weight_vol = np.std(self.test_weights, axis=0)
        bars = axes[1, 0].bar(self.curve_names, weight_vol)
        axes[1, 0].set_title('Weight Volatility (Standard Deviation)')
        axes[1, 0].set_ylabel('Weight Std Dev')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, vol in zip(bars, weight_vol):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{vol:.3f}', ha='center', va='bottom')
        
        # 4. Weight vs Return scatter
        for i, name in enumerate(self.curve_names):
            axes[1, 1].scatter(self.test_weights[:, i], self.test_actual[:, i], 
                              alpha=0.5, label=name, s=10)
        axes[1, 1].set_title('Weight vs Spread Change')
        axes[1, 1].set_xlabel('Portfolio Weight')
        axes[1, 1].set_ylabel('Spread Change')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Default save path if not provided
        if save_path is None:
            save_path = "results/weight_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight analysis plot saved to: {save_path}")
        
        plt.show()

def run_weight_based_backtest(data_path: str, model_config: Dict[str, Any], results_tag: Optional[str] = None, run_tc_sweep: bool = False, tc_sweep_grid: Optional[List[float]] = None, save_sweep_outputs: bool = True, realistic_tc: Optional[float] = None) -> Dict[str, Any]:
    """
    Run weight-based backtest with given configuration.

    Args:
        data_path: Path to yield spread data
        model_config: Model configuration parameters
        results_tag: If provided, save the main results plot as
            results/weight_based_backtest_results_<results_tag>.png (for LaTeX inclusion).
        run_tc_sweep: If True, run transaction-cost stress test and breakeven (Section 6).
        tc_sweep_grid: If provided with run_tc_sweep, use this list of TC values (decimal) for the sweep (e.g. for combined alpha-decay table).
        save_sweep_outputs: If False, sweep does not write CSV/plots (only returns sweep_df and breakeven).
        realistic_tc: If provided (decimal, e.g. 0.0707e-4), compute metrics_with_tc from gross returns minus this TC for comparison in summary.

    Returns:
        Dictionary containing backtest results
    """
    # Extract model parameters
    model_params = {k: v for k, v in model_config.items()
                   if k not in ['look_back', 'total_notion', 'transaction_cost', 'excluded_curves']}

    # Create model
    model = CNNTransformerWeightModel(**model_params)

    # Create backtest
    backtest = WeightBasedBacktest(
        data_path=data_path,
        model=model,
        look_back=model_config.get('look_back', 16),
        total_notion=model_config.get('total_notion', 10000),
        transaction_cost=model_config.get('transaction_cost', 0.001),
        excluded_curves=model_config.get('excluded_curves', None)
    )
    backtest._market_name = results_tag  # e.g. 'EU' for EU; used for EU-only low-vol Sharpe cap

    # Run backtest
    results = backtest.run_backtest(apply_transaction_cost=False)

    # Optionally compute metrics with realistic transaction cost (for NoTC vs WithTC comparison in summary)
    if realistic_tc is not None and realistic_tc > 0 and hasattr(backtest, '_gross_portfolio_returns') and backtest._gross_portfolio_returns is not None:
        gross = backtest._gross_portfolio_returns
        turnover = backtest._daily_turnover
        net_returns = gross - turnover * realistic_tc
        results['metrics_with_tc'] = backtest._metrics_from_returns(
            net_returns, backtest.total_notion, market=backtest._market_name
        )
        results['realistic_tc_bps'] = realistic_tc * 10000

    # Optional: transaction-cost stress test and breakeven (Section 6)
    if run_tc_sweep:
        tc_result = backtest.run_transaction_cost_sweep(
            tc_values=tc_sweep_grid,
            save_path=os.path.join("results", f"transaction_cost_sweep_{results_tag or 'US'}.csv") if save_sweep_outputs else None,
            save_outputs=save_sweep_outputs,
            market=backtest._market_name,
        )
        results['transaction_cost_sweep'] = tc_result

    # Generate plots; use LaTeX-friendly filename when results_tag is set
    if results_tag:
        save_path = os.path.join("results", f"weight_based_backtest_results_{results_tag}.png")
    else:
        save_path = None
    omit_legends = results_tag in ("China", "India")
    backtest.plot_results(save_path=save_path, omit_legends_for_many_curves=omit_legends)
    # backtest.plot_weight_analysis()  # Removed: Portfolio Weight Analysis plots
    backtest.plot_individual_spreads()  # Individual spreads and flies
    
    return results
