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
from datetime import datetime

from base_model import BaseTimeSeriesModel
from cnn_transformer_weight_model import CNNTransformerWeightModel

class WeightBasedBacktest:
    """
    Weight-based backtesting system for portfolio weight generation models.
    
    This system directly applies model-generated weights to spread changes,
    without using Z-scores or position sizing calculations.
    
    Key features:
    - Direct weight application to spread changes
    - Portfolio return calculation: sum(weight_i * spread_change_i)
    - Sharpe ratio calculation from portfolio returns
    - No Z-score or position sizing logic
    - Comprehensive performance analysis and plotting
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
        
    def calculate_portfolio_returns(self, apply_transaction_cost: bool = True) -> None:
        """
        Calculate portfolio returns by directly applying weights to spread changes.
        
        Args:
            apply_transaction_cost: Whether to apply transaction costs
        """
        print("Calculating portfolio returns...")
        
        # CRITICAL: Use scaled returns to match training data scale
        # During training, the model sees scaled targets, so we must use scaled targets for testing too
        returns = self.y_test  # Use scaled returns, not unscaled!
        
        print(f"Returns - Mean: {np.mean(returns):.6f}, Std: {np.std(returns):.6f}")
        print(f"Returns - Min: {np.min(returns):.6f}, Max: {np.max(returns):.6f}")
        
        # Portfolio return = sum(weight_i * return_i) for each day
        # This gives us the weighted average of returns
        self.portfolio_returns = np.sum(self.test_weights * returns, axis=1)
        
        print(f"Portfolio returns - Mean: {np.mean(self.portfolio_returns):.6f}, Std: {np.std(self.portfolio_returns):.6f}")
        print(f"Portfolio returns - Min: {np.min(self.portfolio_returns):.6f}, Max: {np.max(self.portfolio_returns):.6f}")
        
        # Apply transaction costs if requested
        if apply_transaction_cost:
            # Calculate weight changes (turnover)
            weight_changes = np.abs(np.diff(self.test_weights, axis=0, prepend=0))
            total_turnover = np.sum(weight_changes, axis=1)
            transaction_costs = total_turnover * self.transaction_cost
            
            # Net returns after transaction costs
            self.portfolio_returns = self.portfolio_returns - transaction_costs
        
        # Calculate cumulative returns 
        # Portfolio returns are differences, use them directly as returns
        self.cumulative_returns = np.cumsum(self.portfolio_returns)
        self.equity_curve = self.total_notion * (1 + self.cumulative_returns)
        
        print(f"Using returns directly (differences as returns)")
        print(f"Daily return range: [{np.min(self.portfolio_returns):.6f}, {np.max(self.portfolio_returns):.6f}]")
        
        print(f"Portfolio returns calculated - Shape: {self.portfolio_returns.shape}")
        print(f"Mean daily return: {np.mean(self.portfolio_returns):.6f}")
        print(f"Std daily return: {np.std(self.portfolio_returns):.6f}")
        
        # Print portfolio returns for review
        print(f"\n" + "="*60)
        print("PORTFOLIO RETURNS FOR REVIEW")
        print("="*60)
        print("First 20 portfolio returns:")
        for i in range(min(20, len(self.portfolio_returns))):
            print(f"Day {i:3d}: {self.portfolio_returns[i]:10.6f}")
        
        if len(self.portfolio_returns) > 20:
            print("...")
            print("Last 10 portfolio returns:")
            for i in range(max(0, len(self.portfolio_returns)-10), len(self.portfolio_returns)):
                print(f"Day {i:3d}: {self.portfolio_returns[i]:10.6f}")
        
        print(f"\nPortfolio returns statistics:")
        print(f"Min: {np.min(self.portfolio_returns):10.6f}")
        print(f"Max: {np.max(self.portfolio_returns):10.6f}")
        print(f"Mean: {np.mean(self.portfolio_returns):10.6f}")
        print(f"Std: {np.std(self.portfolio_returns):10.6f}")
        print(f"Median: {np.median(self.portfolio_returns):10.6f}")
        
        # Show distribution
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print(f"\nPercentiles:")
        for p in percentiles:
            val = np.percentile(self.portfolio_returns, p)
            print(f"{p:2d}th: {val:10.6f}")
        print("="*60)
        
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        if self.portfolio_returns is None:
            raise ValueError("Portfolio returns not calculated yet. Run calculate_portfolio_returns first.")
        
        # Basic metrics
        total_return = (self.equity_curve[-1] / self.total_notion - 1) * 100
        
        # Use portfolio returns directly for performance metrics
        # Returns (differences) are in different units - present as "adjusted" metrics
        adjusted_annualized_return = np.mean(self.portfolio_returns) * 252  # No percentage conversion
        adjusted_annualized_volatility = np.std(self.portfolio_returns) * np.sqrt(252)  # No percentage conversion
        
        # Adjusted Diff Sharpe ratio (consistent with training calculation)
        # Using formula: (mean return - lambda * volatility)
        lambda_risk = 0.5  # Same risk aversion parameter as training
        adjusted_diff_sharpe_ratio = adjusted_annualized_return - lambda_risk * adjusted_annualized_volatility
        
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
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'final_equity': self.equity_curve[-1],
            'num_trades': len(self.portfolio_returns),
            'weight_stats': weight_stats
        }
        
        return metrics
    
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
        print(f"Annualized Return: {metrics['adjusted_annualized_return']:.4f}")
        print(f"Annualized Volatility: {metrics['adjusted_annualized_volatility']:.4f}")
        print(f"Differentiable Sharpe Ratio: {metrics['adjusted_diff_sharpe_ratio']:.3f}")
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
            extreme_portfolio_threshold = 10.0  # 10 basis points daily return is reasonable
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
        
        if self.portfolio_returns is not None and np.abs(np.mean(self.portfolio_returns)) > 1.0:  # 1 basis point mean
            print(f"[WARN] Portfolio returns have high mean ({np.mean(self.portfolio_returns):.6f} bps), suggesting bias")
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

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """Plot comprehensive backtest results."""
        if self.portfolio_returns is None:
            raise ValueError("Backtest not run yet. Run run_backtest first.")
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
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
            # Calculate cumulative weighted contribution
            cumulative_weighted = np.cumsum(weighted_returns)
            # Scale to portfolio notion to show $ contribution
            portfolio_contribution = self.total_notion * cumulative_weighted
            
            axes[0, 1].plot(portfolio_contribution, label=name, linewidth=1.5, alpha=0.8, color=colors[i])
        
        axes[0, 1].set_title('All Spreads and Flies Comparison (Weighted Contributions)')
        axes[0, 1].set_xlabel('Trading Days')
        axes[0, 1].set_ylabel('Cumulative Weighted Contribution ($)')
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
        
        # 5. Rolling Sharpe ratio
        window = 60  # Match the lookback window used for training
        # Use portfolio returns directly for Sharpe calculation
        rolling_returns = pd.Series(self.portfolio_returns).rolling(window)
        rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
        axes[2, 0].plot(rolling_sharpe)
        axes[2, 0].set_title(f'Rolling Sharpe Ratio ({window}-day window)')
        axes[2, 0].set_xlabel('Trading Days')
        axes[2, 0].set_ylabel('Sharpe Ratio')
        axes[2, 0].grid(True)
        
        # Add horizontal line at Sharpe = 0 for reference
        axes[2, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Sharpe = 0')
        axes[2, 0].axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        axes[2, 0].legend()
        
        # 6. Weight evolution
        for i, name in enumerate(self.curve_names):
            axes[2, 1].plot(self.test_weights[:, i], label=name, alpha=0.7)
        axes[2, 1].set_title('Portfolio Weight Evolution')
        axes[2, 1].set_xlabel('Trading Days')
        axes[2, 1].set_ylabel('Weight')
        axes[2, 1].legend(fontsize=8)
        axes[2, 1].grid(True)
        
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

def run_weight_based_backtest(data_path: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run weight-based backtest with given configuration.
    
    Args:
        data_path: Path to yield spread data
        model_config: Model configuration parameters
        
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
    
    # Run backtest
    results = backtest.run_backtest(apply_transaction_cost=False)
    
    # Generate plots (will be saved to results/ folder automatically)
    backtest.plot_results()  # Includes combined spreads as second subplot
    # backtest.plot_weight_analysis()  # Removed: Portfolio Weight Analysis plots
    backtest.plot_individual_spreads()  # Individual spreads and flies
    
    return results
