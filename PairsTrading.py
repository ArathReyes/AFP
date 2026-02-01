from scipy.stats import norm
import pandas as pd
from itertools import combinations
import statsmodels.tsa.stattools as ts
from Stats import analyze_stationarity
from Dates import bump_date
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def check_cointegration(X: pd.Series, Y: pd.Series) -> bool:
    """
    Performs the Engle-Granger two-step cointegration test on Series X and Y.
    
    Args:
        X (pd.Series): The first time series.
        Y (pd.Series): The second time series.
        
    Returns:
        bool: True if cointegrated (at 95% confidence), False otherwise.
    """
    X.dropna(inplace=True)
    Y.dropna(inplace=True)
    index_ = X.index.intersection(Y.index)
    X, Y = X[index_], Y[index_]

    score, p_value, _ = ts.coint(Y, X)
    return p_value < 0.05


def suitable_pairs(data: pd.DataFrame):

    pairs = combinations(data.columns, 2)
    spreads = pd.DataFrame()
    cointegration = pd.Series()

    for s,l in pairs:
        pair_str = s+'/'+l
        spreads[pair_str] = data[l] - data[s]
        cointegration[pair_str] = check_cointegration(data[s], data[l])

    res = analyze_stationarity(spreads)
    res['Cointegrated'] = cointegration

    return res[['ADF Stat', 'P-Value', 'Is Stationary (95%)', 'Cointegrated', 'Hurst Exp', 'Half-Life (Days)']]

def statistical_dislocation_signal(data: pd.DataFrame, window : float = 63):
    from itertools import combinations
    tenors = data.columns
    pairs = combinations(tenors, 2)
    z_scores = pd.DataFrame(index = tenors, columns=tenors)
    spreads = pd.DataFrame(index = tenors, columns=tenors)
    for s,l in pairs:
        spread = (data[l] - data[s]).dropna().tail(window)
        spreads.at[s, l]  = spread.iloc[-1]
        z_scores.at[s, l] = ((spread - spread.mean())/ spread.std()).iloc[-1]
    return spreads, z_scores

def spread_bollinger_bands(data: pd.DataFrame, long: str, short: str, rolling_window: float = 63, sigma_factor: float = 2):
    bands = pd.DataFrame(columns=['Second Lower', 'Lower', 'Spread', 'Upper', 'Second Upper'])
    bands['Spread'] = (data[long] - data[short]).dropna()
    mus = bands['Spread'].rolling(window=rolling_window).mean()
    stds = bands['Spread'].rolling(window=rolling_window).std()
    bands['Upper'] =  mus + sigma_factor*stds
    bands['Second Upper'] =  mus + 2*sigma_factor*stds
    bands['Lower'] =  mus - sigma_factor*stds
    bands['Second Lower'] =  mus - 2*sigma_factor*stds

    return bands


def plot_spreads_dislocations(data: pd.DataFrame, window: float = 63, confidence: float = 0.75):
    """
    Displays 'spreads' values in a table, colored by 'z_scores' logic.
    
    Green: z_score > threshold
    Red:   z_score < -threshold
    """
    spreads, z_scores = statistical_dislocation_signal(data, window)
    threshold = norm.ppf(confidence)
    # 1. Prepare Header
    headers = ['Tenor'] + list(spreads.columns)
    
    # 2. Prepare Data Columns
    # Start with the index column (the row names)
    cell_values = [list(spreads.index)]
    cell_colors = [['#f0f0f0'] * len(spreads.index)] # Light gray for index column
    
    # Iterate through each column in the dataframe
    for col_name in spreads.columns:
        col_display_values = []
        col_bg_colors = []
        
        # Iterate through each row (index) for the current column
        for row_idx in spreads.index:
            
            # Retrieve values safely
            spread_val = spreads.at[row_idx, col_name]
            z_val = z_scores.at[row_idx, col_name]
            
            # Check for NaNs (empty parts of the matrix)
            if pd.isna(spread_val) or pd.isna(z_val):
                col_display_values.append("")
                col_bg_colors.append("white")
            else:
                # --- DISPLAY LOGIC: Show the Spread ---
                col_display_values.append(f"{spread_val:.4f}")
                
                # --- COLOR LOGIC: Check the Z-Score ---
                if z_val > threshold:
                    # Green for statistical buy signal
                    col_bg_colors.append("rgba(144, 238, 144, 0.6)") 
                elif z_val < -threshold:
                    # Red for statistical sell signal
                    col_bg_colors.append("rgba(255, 99, 71, 0.6)")
                else:
                    # White for no signal
                    col_bg_colors.append("white")
        
        # Append this column's data to the main lists
        cell_values.append(col_display_values)
        cell_colors.append(col_bg_colors)

    # 3. Create Figure
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color='paleturquoise',
            align='center',
            font=dict(color='black', size=12, weight='bold')
        ),
        cells=dict(
            values=cell_values,
            fill_color=cell_colors,
            align='center',
            font=dict(color='black', size=11),
            height=30
        )
    )])
    
    fig.update_layout(
        title_text=f"Spread Z-Score Dislocations (Window: {window} | Confidence {confidence*100}%)",
        width=1000,
        height=600
    )
    
    return fig

def plot_spread_bollinger(data: pd.DataFrame, long_tenor: str, short_tenor: str, window: int = 63, sigma_factor: float = 2.0):
    """
    Calculates spread Bollinger Bands and returns a Plotly figure.
    
    Args:
        data (pd.DataFrame): The dataframe containing prices/rates for the tenors.
        long_tenor (str): Column name for the long leg.
        short_tenor (str): Column name for the short leg.
        window (int): Rolling window size.
        sigma_factor (float): The multiplier for the standard deviation (Z-score).
    """
    # 1. Calculation
    # Ensure window is an integer
    window = int(window)
    
    # Calculate Spread and drop initial NaNs to align plot
    spread_series = (data[long_tenor] - data[short_tenor]).dropna()
    
    # Calculate Rolling Stats
    mus = spread_series.rolling(window=window).mean()
    stds = spread_series.rolling(window=window).std()
    
    # Create DataFrame for bands
    bands = pd.DataFrame(index=spread_series.index)
    bands['Spread'] = spread_series
    bands['Mean'] = mus
    
    # Inner Bands (Based on sigma_factor)
    bands['Upper'] = mus + (sigma_factor * stds)
    bands['Lower'] = mus - (sigma_factor * stds)
    
    # Outer Bands (2x sigma_factor - Extreme dislocation)
    bands['Upper 2'] = mus + (2 * sigma_factor * stds)
    bands['Lower 2'] = mus - (2 * sigma_factor * stds)
    
    # Drop NaNs created by the rolling window for a clean plot
    bands = bands.dropna()

    # 2. Plotting
    fig = go.Figure()

    # -- Layer 1: Extreme Bands (Outer) --
    # We color these red/gray to show extreme limits
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands['Upper 2'],
        mode='lines', line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dot'),
        name=f'Upper ({2*sigma_factor}σ)', legendgroup='outer'
    ))
    
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands['Lower 2'],
        mode='lines', line=dict(color='rgba(150, 150, 150, 0.5)', width=1, dash='dot'),
        name=f'Lower ({2*sigma_factor}σ)', legendgroup='outer'
    ))

    # -- Layer 2: Inner Bands (Standard) --
    # We fill the area between Upper and Lower to show the "Normal" range
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands['Upper'],
        mode='lines', line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
        name=f'Upper ({sigma_factor}σ)', legendgroup='inner'
    ))

    fig.add_trace(go.Scatter(
        x=bands.index, y=bands['Lower'],
        mode='lines', line=dict(color='rgba(100, 100, 100, 0.3)', width=1),
        fill='tonexty', # Fills to the trace before it (Upper)
        fillcolor='rgba(200, 200, 200, 0.1)', # Very light gray fill
        name=f'Lower ({sigma_factor}σ)', legendgroup='inner'
    ))

    # -- Layer 3: Moving Average --
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands['Mean'],
        mode='lines', line=dict(color='orange', width=1.5, dash='dash'),
        name=f'{window}d MA'
    ))

    # -- Layer 4: The Spread (Main Signal) --
    fig.add_trace(go.Scatter(
        x=bands.index, y=bands['Spread'],
        mode='lines', line=dict(color='blue', width=2),
        name='Spread'
    ))

    # 3. Layout Styling
    fig.update_layout(
        title=f"Spread Bollinger Bands: {long_tenor} - {short_tenor} (Window: {window}, σ: {sigma_factor})",
        xaxis_title="Date",
        yaxis_title="Spread Level",
        hovermode="x unified", # Shows all values for a specific date on hover
        template="plotly_white",
        width=1000,
        height=600
    )

    return fig


def plot_pairs_dashboard(
    res: pd.DataFrame,
    data: pd.DataFrame,
    long_tenor: str,
    short_tenor: str,
    window: int = 63,
    confidence: float = 0.75,
    sigma_factor: float = 2.0,
):
    """
    Combine the summary table (res), dislocation table, and Bollinger time-series into one Plotly figure.
    """
    # Clean up the res table for display
    res_display = res.copy()
    res_display = res_display.reset_index()
    res_display = res_display.rename(columns={res_display.columns[0]: "Pair"})
    for col in res_display.select_dtypes(include=["float", "int"]).columns:
        res_display[col] = res_display[col].round(4)

    res_table = go.Table(
        header=dict(
            values=list(res_display.columns),
            fill_color="lightgray",
            align="center",
            font=dict(color="black", size=12, family="Arial Black"),
        ),
        cells=dict(
            values=[res_display[c].astype(str).tolist() for c in res_display.columns],
            align="center",
            height=26,
        ),
    )

    dislocations_fig = plot_spreads_dislocations(data, window=window, confidence=confidence)
    bollinger_fig = plot_spread_bollinger(
        data, long_tenor=long_tenor, short_tenor=short_tenor, window=window, sigma_factor=sigma_factor
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "table"}, {"type": "table"}], [{"type": "xy", "colspan": 2}, None]],
        column_widths=[0.45, 0.55],
        row_heights=[0.45, 0.55],
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
        subplot_titles=(
            "Pairs Stationarity",
            dislocations_fig.layout.title.text or "Spread Z-Scores",
            bollinger_fig.layout.title.text or "Spread Bollinger Bands",
        ),
    )

    fig.add_trace(res_table, row=1, col=1)
    for trace in dislocations_fig.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in bollinger_fig.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        title_text="Pairs Dashboard",
        height=950,
        width=1200,
        template="plotly_white",
        showlegend=True,
    )

    fig.update_xaxes(title_text=bollinger_fig.layout.xaxis.title.text or "Date", row=2, col=1)
    fig.update_yaxes(title_text=bollinger_fig.layout.yaxis.title.text or "Spread Level", row=2, col=1)

    return fig



def plot_pairs_dashboard(
    type_of_spreads: pd.DataFrame,
    long_tenor: str,
    short_tenor: str,
    window: int = 63,
    stationarity_lookback : str = '2Y',
    confidence: float = 0.75,
    sigma_factor: float = 2.0,
):
    """
    Combine the summary table (res), dislocation table, and Bollinger time-series into one Plotly figure.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    def get_rates_data():
        swap = pd.read_excel('data/MX_Rates.xlsx', index_col=0, sheet_name='Swap', parse_dates=True)
        swap.index = [d.date() for d in swap.index]
        bonds = pd.read_excel('data/MX_Rates.xlsx', index_col=0, sheet_name='Bonds', parse_dates=True)
        bonds.index = [d.date() for d in bonds.index]
        tenors = swap.columns.intersection(bonds.columns)
        ASW = (bonds[tenors] - swap[tenors])*100
        ASW.dropna(inplace = True, how = 'all')
        return bonds, swap, ASW
    bonds, swap, ASW = get_rates_data()
    if type_of_spreads.upper() == 'ASW':
        data = ASW
    elif type_of_spreads.upper() == 'BOND':
        data = bonds
    elif type_of_spreads.upper() == 'SWAP':
        data = swap
    else:
        raise Exception(f'Type of spread ({type_of_spreads}) invalid. Only available for bonds, swaps and ASW.')
    # Filter data
    end = data.index[-1]
    start = bump_date(end, '-' + stationarity_lookback)
    cointegration_data = data[(data.index > start) & (data.index <= end) ]
    # Run cointegration test
    res = suitable_pairs(cointegration_data)
    res = res[(res['Cointegrated']) & (res['Half-Life (Days)'] <= 10)].sort_values('Half-Life (Days)')
    # Clean up the res table for display
    res_display = res.copy()
    res_display = res_display.reset_index()
    res_display = res_display.rename(columns={res_display.columns[0]: "Pair"})
    for col in res_display.select_dtypes(include=["float", "int"]).columns:
        res_display[col] = res_display[col].round(4)

    res_table = go.Table(
        header=dict(
            values=list(res_display.columns),
            fill_color="lightgray",
            align="center",
            font=dict(color="black", size=12, family="Arial Black"),
        ),
        cells=dict(
            values=[res_display[c].astype(str).tolist() for c in res_display.columns],
            align="center",
            height=26,
        ),
    )

    dislocations_fig = plot_spreads_dislocations(data, window=window, confidence=confidence)
    bollinger_fig = plot_spread_bollinger(
        data, long_tenor=long_tenor, short_tenor=short_tenor, window=window, sigma_factor=sigma_factor
    )

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "table"}, {"type": "table"}], [{"type": "xy", "colspan": 2}, None]],
        column_widths=[0.45, 0.55],
        row_heights=[0.45, 0.55],
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
        subplot_titles=(
            "Pairs Stationarity",
            dislocations_fig.layout.title.text or "Spread Z-Scores",
            bollinger_fig.layout.title.text or "Spread Bollinger Bands",
        ),
    )

    fig.add_trace(res_table, row=1, col=1)
    for trace in dislocations_fig.data:
        fig.add_trace(trace, row=1, col=2)
    for trace in bollinger_fig.data:
        fig.add_trace(trace, row=2, col=1)

    fig.update_layout(
        title_text="Rates Pairs",
        height=950,
        width=1200,
        template="plotly_white",
        showlegend=True,
    )

    fig.update_xaxes(title_text=bollinger_fig.layout.xaxis.title.text or "Date", row=2, col=1)
    fig.update_yaxes(title_text=bollinger_fig.layout.yaxis.title.text or "Spread Level", row=2, col=1)

    return fig