import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def get_hurst_exponent(time_series, max_lag=20):
    lags = range(2, max_lag)
    # Ensure input is float array to prevent type errors
    time_series = np.array(time_series, dtype=float)
    
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def get_half_life(time_series):
    # Prepare DataFrame
    df_ou = pd.DataFrame(time_series.copy())
    df_ou.columns = ['y']
    df_ou['y_lag'] = df_ou['y'].shift(1)
    df_ou['delta_y'] = df_ou['y'] - df_ou['y_lag']
    df_ou = df_ou.dropna()

    if df_ou.empty:
        return np.nan

    # Regression: delta_y ~ alpha + beta * y_lag
    X = sm.add_constant(df_ou['y_lag'])
    
    # --- The FIX is implicit here because we force numeric in the main loop ---
    model = sm.OLS(df_ou['delta_y'], X)
    results = model.fit()
    
    beta = results.params['y_lag']
    
    if beta >= 0:
        return np.inf
        
    half_life = -np.log(2) / beta
    return half_life

def analyze_stationarity(residuals):
    metrics = []

    for col in residuals.columns:
        # --- THE FIX: Force Numeric Conversion ---
        # 1. Coerce errors to NaN (turns strings/bad data into NaN)
        # 2. Drop NaNs to ensure clean input for statsmodels
        series = pd.to_numeric(residuals[col], errors='coerce').dropna()
        
        # Check if we have enough data left after cleaning
        if len(series) < 30:
            print(f"Skipping {col}: Not enough valid data points ({len(series)})")
            continue
            
        try:
            # 1. Augmented Dickey-Fuller
            adf_result = adfuller(series)
            adf_stat = adf_result[0]
            p_value = adf_result[1]
            
            # 2. Hurst Exponent
            hurst = get_hurst_exponent(series.values)
            
            # 3. Half-Life
            hl = get_half_life(series)
            
            metrics.append({
                'Series': col,
                'ADF Stat': round(adf_stat, 3),
                'P-Value': round(p_value, 4),
                'Is Stationary (95%)': p_value < 0.05,
                'Hurst Exp': round(hurst, 3),
                'Half-Life (Days)': round(hl, 1)
            })
        except Exception as e:
            print(f"Error processing {col}: {e}")

    return pd.DataFrame(metrics).set_index('Series')


def strategy_results(returns, risk_free_rate=0.0, freq=252):
    """
    Computes performance metrics for a trading strategy.
    
    Parameters:
    - returns (pd.Series): Series of daily percentage returns (e.g., 0.01 for 1%).
    - risk_free_rate (float): Annualized risk-free rate (default 0.0).
    - freq (int): Frequency of data (252 for daily, 52 for weekly).
    
    Returns:
    - pd.Series: A formatted series of metrics.
    """
    if returns.empty:
        return pd.Series()

    # 1. Basic Stats
    total_return = (1 + returns).prod() - 1
    mean_return = returns.mean()
    volatility = returns.std()
    
    # 2. Annualized Metrics
    # CAGR (Geometric Mean) is preferred over simple arithmetic mean for long periods
    ann_return = (1 + total_return) ** (freq / len(returns)) - 1
    ann_vol = volatility * np.sqrt(freq)
    
    # 3. Sharpe Ratio
    # Note: If Rf is 0, this simplifies to Information Ratio vs 0
    sharpe_ratio = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0
    
    # 4. Sortino Ratio (Downside Risk only)
    # Only considers volatility of negative returns
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(freq)
    sortino_ratio = (ann_return - risk_free_rate) / downside_vol if downside_vol != 0 else 0
    
    # 5. Max Drawdown
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    max_dd = drawdown.min()
    
    # 6. Calmar Ratio (Return / Max Drawdown)
    calmar_ratio = ann_return / abs(max_dd) if max_dd != 0 else 0
    
    # 7. Win Rate & Trade Stats
    win_rate = len(returns[returns > 0]) / len(returns)
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    profit_factor = abs(returns[returns > 0].sum() / returns[returns < 0].sum())
    
    # 8. Skewness & Kurtosis (Tail Risk)
    skew = returns.skew()
    kurt = returns.kurtosis()

    # Compile Results
    stats = {
        'Total Return': f"{total_return:.2%}",
        'Ann. Return (CAGR)': f"{ann_return:.2%}",
        'Ann. Volatility': f"{ann_vol:.2%}",
        'Sharpe Ratio': round(sharpe_ratio, 2),
        'Sortino Ratio': round(sortino_ratio, 2),
        'Max Drawdown': f"{max_dd:.2%}",
        'Calmar Ratio': round(calmar_ratio, 2),
        'Win Rate': f"{win_rate:.2%}",
        'Profit Factor': round(profit_factor, 2),
        'Skewness': round(skew, 2),
        'Kurtosis': round(kurt, 2)
    }
    
    return pd.Series(stats)