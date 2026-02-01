import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

def yield_curve_decomposition(df_yields: pd.DataFrame, n_components: int = 3) -> dict:
    """
    Decomposes a yield curve history into its first 3 Principal Components 
    (Level, Slope, Curvature).

    Parameters:
    -----------
    df_yields : pd.DataFrame
        Index: Dates (datetime)
        Columns: Tenors (e.g., 1M, 2Y, 10Y)
        Values: Yields (float)

    Returns:
    --------
    dict containing:
        'scores': pd.DataFrame (Time series of the factors)
        'loadings': pd.DataFrame (Eigenvectors - weights for each tenor)
        'residuals': pd.DataFrame (Original Data - Reconstructed Data)
        'explained_variance': list (Ratio of variance explained by each PC)
    """
    # 1. Drop missing rows to ensure PCA stability
    clean_data = df_yields.dropna()
    
    # 2. Initialize and Fit PCA (3 Components)
    # sklearn automatically handles centering (mean subtraction)
    pca = PCA(n_components=n_components)
    
    # 3. Calculate Scores (The time series of the factors)
    # Shape: (n_dates, 3)
    scores_matrix = pca.fit_transform(clean_data)
    
    scores_df = pd.DataFrame(
        scores_matrix,
        index=clean_data.index,
        columns=['Level', 'Slope', 'Curvature']
    )

    # 4. Extract Loadings (The "Shape" of the factors)
    # pca.components_ shape is (3, n_tenors), so we transpose it
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=clean_data.columns,
        columns=['Level', 'Slope', 'Curvature']
    )

    # 5. Calculate Residuals
    # Inverse transform reconstructs the data from the 3 components
    reconstructed_matrix = pca.inverse_transform(scores_matrix)
    reconstructed_matrix = pd.DataFrame(
        reconstructed_matrix,
        index=clean_data.index,
        columns=clean_data.columns
    )
    
    residuals_df = pd.DataFrame(
        clean_data.values - reconstructed_matrix,
        index=clean_data.index,
        columns=clean_data.columns
    )

    return {
        'scores': scores_df,
        'loadings': loadings_df,
        'residuals': residuals_df,
        'explained_variance': pca.explained_variance_ratio_,
        'fitted_yields': reconstructed_matrix
    }