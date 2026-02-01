import json
import numpy as np
import pandas as pd
import datetime as dt
import quantstats
import scipy.stats as stats
from scipy.stats import norm

import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
from Dates import from_excel_date, bump_date
from PCA import yield_curve_decomposition
from Stats import analyze_stationarity

class RelativeValue:

    def __init__(self, country: str , lookback: str = '5Y', base_date: dt.date = dt.date(2026, 1, 1)):
        with open('config.json') as f:
            config = json.load(f)[country]
        self.country = country
        self.lookback = lookback
        self.base_date = base_date
        self.tenors = config["Tenors"]
        self.tradeble_tenors = config["Tradable"]
        self.tc = config["Transaction Cost"]
        self.window = config["Window"]
        self.vol_window = config["Volatility Window"]
        self.confidence = config["Confidence"]
        self.threshold = norm.ppf(self.confidence)
        self.buffer = config["Buffer"]
        self.vt = config["Volatility Target"]
        self.cap = config["Cap"]
        file = 'FX Forwards' if 'fx' in country.lower() else 'Rates'
        self.rates = pd.read_excel(f'data/{file}.xlsx', index_col=0, sheet_name=config["Curve Name"])[self.tenors]
        self.rates.index.rename('Date', inplace=True)
        self.rates.drop('Ticker',inplace=True)
        if 'fx' not in country.lower():
            self.rates.index = self.rates.index.map(from_excel_date) 
        self.rates.index = [d.date() for d in self.rates.index]
        self.rates.dropna(inplace=True)
        if 'fx' not in country.lower():
            self.rates /= 100
        self.rolling_volatilities = self.rates.diff().rolling(self.vol_window).std() * np.sqrt(252)


    def get_residuals(self):
        """
        
        """
        print(f"="*100)
        print(f"Getting PCA residuals ...")

        start_date = bump_date(self.rates.index[1], self.lookback)
        dates = self.rates.loc[start_date:].index
        self.residuals = pd.DataFrame(index=dates, columns= self.rates.columns)
        self.explained_variance = pd.DataFrame(index=dates, columns=['Level', 'Slope', 'Curvature'])
        self.pca_scores = pd.DataFrame(index=dates, columns=['Level', 'Slope', 'Curvature'])
        for i, d in enumerate(dates):
            start = bump_date(d, '-' + self.lookback)
            yield_curve = self.rates[(self.rates.index >= start) & (self.rates.index <=d)].dropna()
            res = yield_curve_decomposition(yield_curve)
            self.residuals.loc[d] = res['residuals'].iloc[-1]
            self.explained_variance.loc[d] = res['explained_variance']
            self.pca_scores.loc[d] = res['scores'].loc[d].values
        print(f"Residuals successfully computed")
        print(f"="*100)


    def get_weights(self,
                    confidence: float,
                    buffer: float,
                    window: float,
                    vol_window: float,
                    volatility_target: float,
                    cap: float) -> pd.DataFrame:
        rolling_volatilities = self.rates.diff().rolling(vol_window).std() * np.sqrt(252)
        z_scores = (self.residuals - self.residuals.rolling(window).mean()) / self.residuals.rolling(window).std()
        z_scores.dropna(inplace=True)
        
        dates = z_scores.index
        pos = pd.DataFrame(index=dates, columns=self.residuals.columns)
        threshold = norm.ppf(confidence)

        for i, d in enumerate(dates):
            rebalancing_toggle = (z_scores.loc[d].abs() > threshold).astype(int)
            normalized_pos = (- z_scores.loc[d] / rolling_volatilities.loc[d]) * rebalancing_toggle
            total_gross_exposure = normalized_pos.abs().sum()
            scale_factor = np.clip(total_gross_exposure, 1e-6, None)
            normalized_pos = normalized_pos.div(scale_factor, axis=0)
            if i == 0:
                pos.loc[d] = normalized_pos
            elif i > 0:
                d_1 = dates[i-1]
                rebalancing_toggle *= ((normalized_pos - pos.loc[d_1]).abs() > buffer).astype(int)
                pos.loc[d] = normalized_pos + (pos.loc[d_1] * (1 - rebalancing_toggle))
            if abs(pos.loc[d].sum()) < 1:
                normalize_factor = max(pos.loc[d].abs().sum(), 1e-4)
                pos.loc[d] = pos.loc[d].div(normalize_factor, axis = 'rows')
        
        pos = (pos* volatility_target/rolling_volatilities).clip(-cap, cap)
        return pos

    def compute_pnl(self,
                    pos: pd.DataFrame,
                    tc: float = 0.2e-4,
                    lag: int = 1) -> pd.DataFrame:
        pnl = pos.shift(lag) * self.rates.diff()  - pos.diff().abs() * tc
        pnl.dropna(inplace=True)
        IL = 100*(1 + pnl.sum(axis=1)).cumprod()
        return pnl, IL
    
    def update_tenors(self):
        """
        """
        print("Updating tradable tenors...")
        self.residuals = self.residuals[self.tradeble_tenors]
        self.rates = self.rates[self.tradeble_tenors]

    def stress_rebalacing(self,
                          buffer_list: list = [0.00, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15,  0.175, 0.20, 0.225, 0.25],
                          confidence_list : list = [0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825, 0.85, 0.875, 0.90]):
        print("Rebalancing Stress test...")
        sharpes = pd.DataFrame(index=buffer_list, columns=confidence_list)
        for b in buffer_list:
            for c in confidence_list:
                pos  = self.get_weights(confidence=c, buffer=b, window=self.window,
                                        vol_window=self.vol_window, volatility_target=self.vt, cap=self.cap)
                pnl, _ = self.compute_pnl(pos, tc=self.tc, lag=1) 
                pnl = pnl.sum(axis=1)
                sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(252)
                sharpes.loc[b,c] = sharpe_ratio
        return sharpes
    

    def stress_windows(self,
                       window_list : list = [1, 2, 3, 4, 6, 9, 12, 18, 24],
                       vol_window_list: list = [1, 2, 3, 4, 6, 9, 12, 18, 24]):
        print(f'Window Stress test...')
        sharpes = pd.DataFrame(index=vol_window_list, columns=window_list)

        for vw in vol_window_list:
            rv_ = self.rates.diff().rolling(21*vw).std() * np.sqrt(252)
            for w in window_list:
                pos  = self.get_weights(window=21*w, vol_window= 21*vw, confidence=0.5, buffer=0.0, volatility_target=self.vt, cap=self.cap)
                pnl, IL = self.compute_pnl(pos, tc=self.tc, lag=1) 
                pnl = pnl.sum(axis=1)
                sharpe_ratio = (pnl.mean() / pnl.std()) * np.sqrt(252)
                sharpes.loc[vw, w] = sharpe_ratio
        return sharpes
    
    def stress_signal_persistance(self,
                                  lags : list = [-15, -14, -13, -12, -11, -10,
                                                 -9, -8, -7, -6, -5, -4, -3, -2,
                                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10, 11, 12, 13, 14, 15]):
        print(f'Signal Persistance test...')
        sharpes = pd.Series(index=lags)
        pos  = self.get_weights(window = self.window,
                                vol_window=self.vol_window,
                                confidence=0.5, buffer= 0.0,
                                volatility_target=self.vt, cap=self.cap)
        for lag in lags:
                pnl, _ = self.compute_pnl(pos, tc=0.0, lag=lag) 
                pnl = pnl.sum(axis=1)
                sharpes[lag] = (pnl.mean() / pnl.std()) * np.sqrt(252)

        return sharpes
    
    @staticmethod
    def plot_dashboard_full(residuals, lagged_sharpes, rebalancing_sharpes, window_sharpes):
        """
        Creates a 5-Row Dashboard:
        Row 1: Table + Half-Life
        Row 2: Lagged Sharpes
        Row 3: Performance Surface (Toggle 2D/3D)
        Row 4: Window Sharpes (Toggle 2D/3D)
        Row 5: QQ Plots of Residuals (Dropdown Selection)
        """
        report_df = analyze_stationarity(residuals)
        # --- 1. PRE-CALCULATION FOR QQ PLOTS ---
        # We pre-calculate standardized QQ data for all columns to use in the dropdown
        qq_data = {}
        for col in residuals.columns:
            # Clean data
            data = pd.to_numeric(residuals[col], errors='coerce').dropna()
            if len(data) > 0:
                # Standardize data (Z-score) so we can compare to a standard normal line
                data_std = (data - data.mean()) / data.std()
                # Get theoretical quantiles (osm) and ordered responses (osr)
                (osm, osr), _ = stats.probplot(data_std, dist="norm", plot=None)
                qq_data[col] = {'theoretical': osm, 'sample': osr}

        # Get the first column to display initially
        initial_col = residuals.columns[0]
        initial_qq = qq_data[initial_col]

        # --- 2. LAYOUT SETUP ---
        fig = make_subplots(
            rows=5, cols=2,
            column_widths=[0.55, 0.45],
            specs=[
                [{"type": "table"}, {"type": "xy"}],      # Row 1
                [{"type": "xy", "colspan": 2}, None],     # Row 2
                [{"type": "xy", "colspan": 2}, None],     # Row 3 (2D/3D)
                [{"type": "xy", "colspan": 2}, None],     # Row 4 (2D/3D)
                [{"type": "xy", "colspan": 2}, None]      # Row 5 (QQ Plot)
            ],
            subplot_titles=(
                "Mean Reversion Metrics", "Half-Life (Days)", 
                "Signal Persistance", 
                "Buffer/Threshold Stress Test", 
                "Windows Stress Test",
                f"QQ Plot vs Normal Dist"
            ),
            vertical_spacing=0.05
        )

        # --- ROW 1: Table & Half-Life ---
        # (Trace 0)
        display_df = report_df.copy().reset_index()
        fill_colors = []
        if 'Is Stationary (95%)' in display_df.columns:
            for stationary in display_df['Is Stationary (95%)']:
                fill_colors.append('rgba(235, 255, 235, 1)' if stationary else 'rgba(255, 235, 235, 1)')
        else:
            fill_colors = ['white'] * len(display_df)

        fig.add_trace(go.Table(
            header=dict(values=list(display_df.columns), fill_color='paleturquoise', align='center'),
            cells=dict(values=[display_df[k].tolist() for k in display_df.columns],
                    fill_color=[fill_colors] * len(display_df.columns), align='center',
                    format=[None, ".3f", ".4f", None, ".3f", ".1f"])
        ), row=1, col=1)

        # (Trace 1)
        hl_colors = ['crimson' if hl > 20 else 'cornflowerblue' for hl in report_df['Half-Life (Days)']]
        fig.add_trace(go.Bar(
            x=report_df.index, y=report_df['Half-Life (Days)'],
            marker_color=hl_colors, name="Half-Life"
        ), row=1, col=2)

        # --- ROW 2: Sharpe Ratios ---
        # (Trace 2)
        sharpe_colors = ['forestgreen' if s > 0 else 'firebrick' for s in lagged_sharpes.values]
        fig.add_trace(go.Bar(
            x=lagged_sharpes.index, y=lagged_sharpes.values,
            marker_color=sharpe_colors, name="Sharpe",
            text=[f"{v:.2f}" for v in lagged_sharpes.values], textposition='auto'
        ), row=2, col=1)

        # --- ROW 3: HEATMAP 1 ---
        # (Trace 3: 2D)
        fig.add_trace(go.Heatmap(
            z=rebalancing_sharpes.values, x=rebalancing_sharpes.columns, y=rebalancing_sharpes.index,
            colorscale='Viridis', colorbar=dict(title="Sharpe", len=0.15, y=0.53, x=1.02),
            hovertemplate='<b>Conf</b>: %{x}<br><b>Buffer</b>: %{y}<br><b>Sharpe</b>: %{z:.2f}<extra></extra>',
            visible=True
        ), row=3, col=1)
        
        # (Trace 4: 3D)
        fig.add_trace(go.Surface(
            z=rebalancing_sharpes.values, x=rebalancing_sharpes.columns, y=rebalancing_sharpes.index,
            colorscale='Viridis', colorbar=dict(title="Sharpe", len=0.15, y=0.53, x=1.02),
            hovertemplate='<b>Conf</b>: %{x}<br><b>Buffer</b>: %{y}<br><b>Sharpe</b>: %{z:.2f}<extra></extra>',
            visible=False, scene='scene'
        ))

        # --- ROW 4: HEATMAP 2 ---
        # (Trace 5: 2D)
        fig.add_trace(go.Heatmap(
            z=window_sharpes.values, x=window_sharpes.columns, y=window_sharpes.index,
            colorscale='Viridis', colorbar=dict(title="Sharpe", len=0.15, y=0.30, x=1.02),
            hovertemplate='<b>Conf</b>: %{x}<br><b>Buffer</b>: %{y}<br><b>Sharpe</b>: %{z:.2f}<extra></extra>',
            visible=True
        ), row=4, col=1)

        # (Trace 6: 3D)
        fig.add_trace(go.Surface(
            z=window_sharpes.values, x=window_sharpes.columns, y=window_sharpes.index,
            colorscale='Viridis', colorbar=dict(title="Sharpe", len=0.15, y=0.30, x=1.02),
            hovertemplate='<b>Conf</b>: %{x}<br><b>Buffer</b>: %{y}<br><b>Sharpe</b>: %{z:.2f}<extra></extra>',
            visible=False, scene='scene2'
        ))

        # --- ROW 5: QQ PLOT (Dynamic) ---
        # (Trace 7: The Scatter Points)
        fig.add_trace(go.Scatter(
            x=initial_qq['theoretical'], y=initial_qq['sample'],
            mode='markers', marker=dict(color='blue', size=4),
            name='Residuals'
        ), row=5, col=1)

        # (Trace 8: The Reference Line y=x)
        # We create a line from min to max of the theoretical range
        min_val, max_val = min(initial_qq['theoretical']), max(initial_qq['theoretical'])
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(color='red', dash='dash'),
            name='Normal Dist'
        ), row=5, col=1)

        # --- 3. LAYOUT & MENUS ---
        
        # SCENES CONFIG (Hidden initially)
        fig.update_layout(
            scene=dict(
                domain=dict(x=[0, 1], y=[0.42, 0.58]), # Row 3 position
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
            ),
            scene2=dict(
                domain=dict(x=[0, 1], y=[0.20, 0.36]), # Row 4 position
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5))
            )
        )

        # LABELS
        fig.update_yaxes(title_text="Days", row=1, col=2)
        fig.update_yaxes(title_text="Sharpe", row=2, col=1)
        
        fig.update_xaxes(title_text="Confidence", row=3, col=1)
        fig.update_yaxes(title_text="Buffer", row=3, col=1)
        
        fig.update_xaxes(title_text="Confidence", row=4, col=1)
        fig.update_yaxes(title_text="Buffer", row=4, col=1)

        fig.update_xaxes(title_text="Theoretical Quantiles (Normal)", row=5, col=1)
        fig.update_yaxes(title_text="Sample Quantiles (Standardized)", row=5, col=1)

        # --- DEFINING BUTTONS ---

        # 1. Dropdown for QQ Plots (Row 5)
        # We use 'restyle' to update ONLY traces 7 and 8
        qq_buttons = []
        for col in residuals.columns:
            if col not in qq_data: continue
            
            # Calculate line coordinates for this specific series
            t_min = min(qq_data[col]['theoretical'])
            t_max = max(qq_data[col]['theoretical'])
            
            qq_buttons.append(dict(
                label=str(col),
                method="restyle",
                args=[
                    {
                        'x': [qq_data[col]['theoretical'], [t_min, t_max]], 
                        'y': [qq_data[col]['sample'], [t_min, t_max]]
                    }, 
                    [7, 8] # Indices of traces to update (Scatter, Line)
                ]
            ))

        # 2. Update Menus Configuration
        fig.update_layout(
            updatemenus=[
                # ROW 3 TOGGLE
                dict(
                    type="buttons", direction="left", x=0.05, y=0.61, xanchor="left",
                    buttons=[
                        dict(label="2D", method="update", args=[
                            # Vis: Table, HL, Shp, HM1(ON), Srf1(OFF), HM2(ON), Srf2(OFF), QQ1(ON), QQ2(ON)
                            {"visible": [True, True, True, True, False, True, False, True, True]}, 
                            {"xaxis3.visible": True, "yaxis3.visible": True, "scene.visible": False}
                        ]),
                        dict(label="3D", method="update", args=[
                            # Vis: Table, HL, Shp, HM1(OFF), Srf1(ON), HM2(ON), Srf2(OFF), QQ1(ON), QQ2(ON)
                            {"visible": [True, True, True, False, True, True, False, True, True]}, 
                            {"xaxis3.visible": False, "yaxis3.visible": False, "scene.visible": True, 
                            "scene.xaxis.visible": True, "scene.yaxis.visible": True, "scene.zaxis.visible": True}
                        ])
                    ]
                ),
                # ROW 4 TOGGLE
                dict(
                    type="buttons", direction="left", x=0.05, y=0.39, xanchor="left",
                    buttons=[
                        dict(label="2D", method="update", args=[
                            {"visible": [True, True, True, True, False, True, False, True, True]},
                            {"xaxis4.visible": True, "yaxis4.visible": True, "scene2.visible": False}
                        ]),
                        dict(label="3D", method="update", args=[
                            {"visible": [True, True, True, True, False, False, True, True, True]},
                            {"xaxis4.visible": False, "yaxis4.visible": False, "scene2.visible": True,
                            "scene2.xaxis.visible": True, "scene2.yaxis.visible": True, "scene2.zaxis.visible": True}
                        ])
                    ]
                ),
                # ROW 5 DROPDOWN (QQ Selection)
                dict(
                    type="dropdown", direction="down", x=0.05, y=0.17, xanchor="left",
                    active=0,
                    buttons=qq_buttons
                )
            ]
        )

        fig.update_layout(title_text="Stress Test", height=2200, template="plotly_white", showlegend=False)

        return fig
    
    def stress_test(self):
        print("="*100)
        print("STRESS TEST")
        print("-"*100)

        self.update_tenors()
        rebalancing_sharpes = self.stress_rebalacing()
        window_sharpes = self.stress_windows()
        lagged_sharpes = self.stress_signal_persistance()

        RelativeValue.plot_dashboard_full(self.residuals, lagged_sharpes, rebalancing_sharpes, window_sharpes).write_html(f"{self.country}_stress_test.html")
        print("="*100)


    
    def backtest(self,
                    confidence: float = None,
                    buffer: float = None,
                    window: float = None,
                    vol_window: float = None,
                    volatility_target: float = None,
                    cap: float = None,
                    tc: float = None,
                    lag: int = 1):
        
        if confidence is None:
            confidence = self.confidence
        if buffer is None:
            buffer = self.buffer
        if window is None:
            window = self.window
        if vol_window is None:
            vol_window = self.vol_window
        if volatility_target is None:
            volatility_target = self.vt
        if cap is None:
            cap = self.cap
        if tc is None:
            tc = self.tc
        print("="*100)
        print("BACKTEST")
        print("-"*100)
        print(f"  Parameters | Confidence: {confidence} | Buffer: {buffer} | Window: {window} | Volatility Window: {vol_window} | Volatility Target: {vol_window}")
        pos = self.get_weights(confidence=confidence, buffer=buffer,
                               window=window, vol_window=vol_window,
                               volatility_target=volatility_target,
                               cap=cap)
        self.pnl_backtest, IL = self.compute_pnl(pos=pos, tc=tc, lag=lag)
        IL.index = pd.to_datetime(IL.index)
        quantstats.reports.html(IL.astype(float), output=f'{self.country}_Backtest.html', title = f'{self.country} Relative Value Strategy')
        print("="*100)
