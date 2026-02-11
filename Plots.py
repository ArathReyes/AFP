import numpy as np
import pandas as pd
from scipy.stats import norm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def plot_heatmap(data: pd.DataFrame, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.astype(float), annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_bar(data: pd.Series, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=data.index.astype(str), y=data.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_drawdown(pnl_series, title='Strategy Drawdown'):
    """
    Computes and plots the drawdown with strict type checking to avoid 'isfinite' errors.
    """
    # --- 1. Data Cleaning (The Fix) ---
    # Ensure input is a Series (handle DataFrame input by taking first column)
    if isinstance(pnl_series, pd.DataFrame):
        pnl_series = pnl_series.iloc[:, 0]
        
    # Force numeric type, turning non-parseable strings to NaN
    pnl_series = pd.to_numeric(pnl_series, errors='coerce')
    
    # Drop NaNs and Infinite values which break plotting
    pnl_series = pnl_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    if pnl_series.empty:
        print(f"Error: {title} - Input series is empty after cleaning.")
        return pd.Series(dtype=float)

    # --- 2. Calculation ---
    # Assumes pnl_series are Returns (e.g. 0.01). 
    # If using Dollar P&L, change to: wealth_index = pnl_series.cumsum()
    wealth_index = (1 + pnl_series).cumprod()
    
    hwm = wealth_index.cummax()
    drawdown = (wealth_index / hwm) - 1
    
    # --- 3. Plotting ---
    plt.figure(figsize=(10, 6))
    
    # Fill "Underwater" Area
    plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    plt.plot(drawdown.index, drawdown, color='darkred', linewidth=1.5, label='Drawdown')
    
    # Formatting
    plt.title(title, fontsize=14, weight='bold')
    plt.ylabel('Drawdown', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.8)
    
    # Y-axis percentage format
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.legend(loc='lower left')
    plt.tight_layout()
    
    plt.show()
    
    return drawdown



def plot_time_series(df, title, x_label, y_label, template="plotly_white"):
    """
    Generates an interactive Plotly line chart for a dataframe with multiple time series.
    
    Parameters:
    - df: The dataframe containing time series (Index should be datetime).
    - title: The main title of the chart.
    - x_label: Label for the X-axis.
    - y_label: Label for the Y-axis.
    - template: Plotly theme (default: 'plotly_white').
    
    Returns:
    - fig: The Plotly figure object.
    """
    
    # px.line automatically uses the DataFrame index as the x-axis
    # and treats every column as a separate line.
    fig = px.line(df, title=title, template=template)

    # Update the axis labels and layout settings
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title="Series Name",
        hovermode="x unified" # Shows all values for a specific x-point on hover
    )

    return fig

