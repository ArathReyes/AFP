"""
CNN+Transformer model implementation for time series prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, List
import math
import os

from base_model import BaseTimeSeriesModel, ModelFactory

def compute_sharpe_ratio_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                             holding_days: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Sharpe ratio loss for yield spread predictions.
    
    This function treats yield spread predictions as trading signals and calculates
    returns based on the accuracy of directional predictions.
    
    Args:
        predictions: Model predictions of shape (batch_size, num_spreads)
        targets: Actual returns of shape (batch_size, num_spreads)
        holding_days: Number of holding days (default 1)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Negative Sharpe ratio (to minimize)
    """
    # Method: Treat yield spread predictions as trading signals
    # If we predict spread will increase, we "go long" the spread
    # If we predict spread will decrease, we "go short" the spread
    
    # Calculate trading returns based on prediction accuracy
    actual_direction = torch.sign(targets)  # +1 if spread increased, -1 if decreased
    predicted_direction = torch.sign(predictions)  # +1 if we predict increase, -1 if decrease
    
    # Trading return = predicted_direction * actual_change
    # If we predict correctly: positive return
    # If we predict incorrectly: negative return
    trading_returns_bps = predicted_direction * targets  # Returns in basis points
    
    # Average returns across all spreads for each sample
    returns_bps = torch.mean(trading_returns_bps, dim=1)  # (batch_size,) as returns
    
    # Use returns directly (differences are treated as returns)
    returns = returns_bps  # No conversion needed
    
    # Compute Diff Sharpe ratio: (mean return - lambda * volatility)
    mean_return = torch.mean(returns) * 252  # Annualized return
    std_return = torch.std(returns) * torch.sqrt(torch.tensor(252.0))  # Annualized volatility
    
    # Diff Sharpe ratio with risk aversion parameter lambda
    lambda_risk = 0.1  # Risk aversion parameter - can be adjusted
    diff_sharpe_ratio = mean_return - lambda_risk * std_return
    
    # Return negative Diff Sharpe ratio for minimization
    return -diff_sharpe_ratio

def compute_trading_sharpe_loss(predictions: torch.Tensor, targets: torch.Tensor,
                               holding_days: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Sharpe ratio loss based on actual trading strategy returns.
    
    This function simulates a realistic trading strategy where:
    1. Yield spread predictions are used to make trading decisions
    2. Returns are calculated based on the actual yield spread changes
    3. Sharpe ratio is computed from the trading returns
    
    Args:
        predictions: Model predictions of shape (batch_size, num_spreads)
        targets: Actual returns of shape (batch_size, num_spreads)
        holding_days: Number of holding days (default 1)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Negative Sharpe ratio (to minimize)
    """
    # Simulate trading strategy:
    # 1. Use predictions as position sizes (normalized to [-1, 1])
    # 2. Calculate returns as: position_size * actual_change
    
    # Normalize predictions to position sizes
    # Use tanh to bound between -1 and 1
    position_sizes = torch.tanh(predictions)  # (batch_size, num_spreads)
    
    # Calculate trading returns
    # Return = position_size * actual_change
    trading_returns_bps = position_sizes * targets  # (batch_size, num_spreads) in basis points
    
    # Average returns across all spreads for each sample
    returns_bps = torch.mean(trading_returns_bps, dim=1)  # (batch_size,) as returns
    
    # Use returns directly (differences are treated as returns)
    returns = returns_bps  # No conversion needed
    
    # Compute Diff Sharpe ratio: (mean return - lambda * volatility)
    mean_return = torch.mean(returns) * 252  # Annualized return
    std_return = torch.std(returns) * torch.sqrt(torch.tensor(252.0))  # Annualized volatility
    
    # Diff Sharpe ratio with risk aversion parameter lambda
    lambda_risk = 0.1  # Risk aversion parameter - can be adjusted
    diff_sharpe_ratio = mean_return - lambda_risk * std_return
    
    # Return negative Diff Sharpe ratio for minimization
    return -diff_sharpe_ratio

def compute_portfolio_sharpe_loss(weights: torch.Tensor, targets: torch.Tensor,
                                 holding_days: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute portfolio Sharpe ratio loss for pre-normalized portfolio weights.
    
    Args:
        weights: Portfolio weights of shape (batch_size, num_spreads) - ALREADY NORMALIZED
        targets: Actual returns of shape (batch_size, num_spreads)
        holding_days: Number of holding days (default 1)
        eps: Small epsilon to avoid division by zero
        
    Returns:
        Negative Sharpe ratio (to minimize)
    """
    # Calculate portfolio returns (weights are already normalized by the model)
    # targets are now properly scaled returns (standardized units)
    portfolio_returns = torch.sum(weights * targets, dim=1)  # (batch_size,) in standardized units
    
    # Note: Sharpe ratio is scale-invariant (mean/std), so standardized units are fine
    
    # Compute Diff Sharpe ratio: (mean return - lambda * volatility)
    mean_return = torch.mean(portfolio_returns) * 252  # Annualized return
    std_return = torch.std(portfolio_returns) * torch.sqrt(torch.tensor(252.0))  # Annualized volatility
    
    # Diff Sharpe ratio with risk aversion parameter lambda
    lambda_risk = 0.5  # Risk aversion parameter - can be adjusted
    diff_sharpe_ratio = mean_return - lambda_risk * std_return
    
    # Return negative Diff Sharpe ratio for minimization
    return -diff_sharpe_ratio

class PositionalEncoding(nn.Module):
    """Injects positional information into the input sequence."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CNN_Block(nn.Module):
    """
    1D Convolutional block with a proper residual connection.
    """
    def __init__(self, in_filters: int, out_filters: int, normalization: bool = True, filter_size: int = 2):
        super(CNN_Block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=filter_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=out_filters, out_channels=out_filters, kernel_size=filter_size, padding='same')
        self.relu = nn.ReLU(inplace=True)
        self.normalization1 = nn.BatchNorm1d(out_filters)
        self.normalization2 = nn.BatchNorm1d(out_filters)

        # 1x1 convolution for the residual connection if channels change
        self.shortcut = nn.Conv1d(in_filters, out_filters, kernel_size=1) if in_filters != out_filters else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (N, C_in, T)
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.normalization1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.normalization2(out)

        out += residual # Add residual
        out = self.relu(out)

        return out

class CNNTransformerTimeSeries(nn.Module):
    """
    Hybrid CNN-Transformer model for multi-variate time series forecasting.
    """
    def __init__(
        self,
        input_features: int,
        output_features: int,
        lookback: int = 30,
        filter_numbers: List[int] = [16, 32], # Gradual increase
        attention_heads: int = 4,
        hidden_units_factor: int = 2,
        dropout: float = 0.25,
        filter_size: int = 3, # A slightly larger kernel might capture more info
    ):
        super(CNNTransformerTimeSeries, self).__init__()

        all_filters = [input_features] + filter_numbers
        self.convBlocks = nn.ModuleList()
        for i in range(len(all_filters)-1):
            self.convBlocks.append(
                CNN_Block(all_filters[i], all_filters[i+1], filter_size=filter_size)
            )

        final_cnn_output_dim = all_filters[-1]

        # Positional Encoding Layer
        self.pos_encoder = PositionalEncoding(d_model=final_cnn_output_dim, dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=final_cnn_output_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_units_factor * final_cnn_output_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2) # Using 2 layers
        self.linear = nn.Linear(final_cnn_output_dim, output_features)

    def forward(self, x):
        # Input x: (N, T, C_in)
        x = x.permute(0, 2, 1) # -> (N, C_in, T)

        for block in self.convBlocks:
            x = block(x) # (N, C_out, T)
        x = x.permute(2, 0, 1) # -> (T, N, C_out)
        x = self.pos_encoder(x) # Apply positional encoding
        x = self.transformer_encoder(x) # (T, N, C_out)
        prediction = self.linear(x[-1,:,:]) # (N, output_features)
        return prediction

class CNNTransformerWeightGenerator(nn.Module):
    """
    CNN-Transformer model that generates portfolio weights for yield spreads.
    This model takes yield spread time series and outputs allocation weights.
    """
    def __init__(
        self,
        input_features: int,
        lookback: int = 30,
        filter_numbers: List[int] = [16, 32],
        attention_heads: int = 4,
        hidden_units_factor: int = 2,
        dropout: float = 0.25,
        filter_size: int = 3,
        weight_mode: str = "softmax",  # "softmax", "tanh", or "linear"
    ):
        super(CNNTransformerWeightGenerator, self).__init__()
        
        self.weight_mode = weight_mode
        self.input_features = input_features

        all_filters = [input_features] + filter_numbers
        self.convBlocks = nn.ModuleList()
        for i in range(len(all_filters)-1):
            self.convBlocks.append(
                CNN_Block(all_filters[i], all_filters[i+1], filter_size=filter_size)
            )

        final_cnn_output_dim = all_filters[-1]

        # Positional Encoding Layer
        self.pos_encoder = PositionalEncoding(d_model=final_cnn_output_dim, dropout=dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=final_cnn_output_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_units_factor * final_cnn_output_dim,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        
        # Output layer: generates weights for each yield spread
        self.weight_linear = nn.Linear(final_cnn_output_dim, input_features)
        
        # Weight normalization options
        if weight_mode == "softmax":
            # Always positive, sums to 1 (current approach)
            self.weight_norm = nn.Softmax(dim=1)
        elif weight_mode == "tanh":
            # Can be negative, bounded between -1 and 1
            self.weight_norm = nn.Tanh()
        elif weight_mode == "linear":
            # Can be any value, unbounded (no normalization)
            self.weight_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown weight_mode: {weight_mode}")

    def forward(self, x):
        # Input x: (N, T, C_in) - batch of yield spread time series
        x = x.permute(0, 2, 1) # -> (N, C_in, T)

        # CNN feature extraction
        for block in self.convBlocks:
            x = block(x) # (N, C_out, T)
        
        # Transformer processing
        x = x.permute(2, 0, 1) # -> (T, N, C_out)
        x = self.pos_encoder(x) # Apply positional encoding
        x = self.transformer_encoder(x) # (T, N, C_out)
        
        # Generate weights from final time step
        weights = self.weight_linear(x[-1,:,:]) # (N, input_features)
        weights = self.weight_norm(weights)  # Apply chosen normalization
        
        return weights

class CNNTransformerModel(BaseTimeSeriesModel):
    """
    CNN+Transformer model implementation for time series prediction.

    This model uses a hybrid CNN-Transformer architecture with convolutional blocks
    for feature extraction and transformer layers for sequence modeling.
    """

    def __init__(self, filter_numbers: List[int] = [16, 32], attention_heads: int = 4,
                 hidden_units_factor: int = 2, dropout: float = 0.25, filter_size: int = 3,
                 batch_size: int = 64, learning_rate: float = 0.001,
                 num_epochs: int = 60, patience: int = 10, device: Optional[str] = None,
                 weight_decay: float = 1e-5, step_size: int = 10, gamma: float = 0.5,
                 loss_function: str = "sharpe_ratio"):
        """
        Initialize CNN+Transformer model.

        Args:
            filter_numbers: List of filter numbers for CNN blocks
            attention_heads: Number of attention heads in transformer
            hidden_units_factor: Factor for feedforward dimension in transformer
            dropout: Dropout rate
            filter_size: Kernel size for CNN blocks
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            weight_decay: Weight decay for regularization
            step_size: Step size for learning rate scheduler
            gamma: Gamma for learning rate scheduler
            loss_function: Loss function to use ('sharpe_ratio', 'trading_sharpe', 'portfolio_sharpe', or 'mse')
        """
        super().__init__(
            filter_numbers=filter_numbers,
            attention_heads=attention_heads,
            hidden_units_factor=hidden_units_factor,
            dropout=dropout,
            filter_size=filter_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
            weight_decay=weight_decay,
            step_size=step_size,
            gamma=gamma
        )

        self.filter_numbers = filter_numbers
        self.attention_heads = attention_heads
        self.hidden_units_factor = hidden_units_factor
        self.dropout = dropout
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.loss_function = loss_function

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model will be initialized during fit()
        self.model = None
        self.input_size = None
        self.output_size = None
        self.lookback = None

    def _initialize_model(self, input_size: int, output_size: int, lookback: int):
        """Initialize the CNN+Transformer model with given dimensions."""
        self.input_size = input_size
        self.output_size = output_size
        self.lookback = lookback

        self.model = CNNTransformerTimeSeries(
            input_features=input_size,
            output_features=output_size,
            lookback=lookback,
            filter_numbers=self.filter_numbers,
            attention_heads=self.attention_heads,
            hidden_units_factor=self.hidden_units_factor,
            dropout=self.dropout,
            filter_size=self.filter_size
        ).to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the CNN+Transformer model.

        Args:
            X_train: Training features of shape (n_samples, look_back, n_features)
            y_train: Training targets of shape (n_samples, n_features)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary containing training metrics
        """
        # Initialize model if not done already
        if self.model is None:
            self._initialize_model(X_train.shape[2], y_train.shape[1], X_train.shape[1])

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Loss and optimizer - Using Sharpe ratio loss instead of MSE
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)

        # Training loop with early stopping
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs_trained': 0
        }

        print(f"Starting CNN+Transformer training on {self.device}...")

        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch)

                # Use selected loss function
                if self.loss_function == "sharpe_ratio":
                    loss = compute_sharpe_ratio_loss(output, y_batch)
                elif self.loss_function == "trading_sharpe":
                    loss = compute_trading_sharpe_loss(output, y_batch)
                elif self.loss_function == "portfolio_sharpe":
                    loss = compute_portfolio_sharpe_loss(output, y_batch)
                elif self.loss_function == "mse":
                    loss = nn.MSELoss()(output, y_batch)
                else:
                    raise ValueError(f"Unknown loss function: {self.loss_function}")

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)
            training_history['train_loss'].append(train_loss)

            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        output = self.model(X_batch)

                        # Use selected loss function for validation too
                        if self.loss_function == "sharpe_ratio":
                            loss = compute_sharpe_ratio_loss(output, y_batch)
                        elif self.loss_function == "trading_sharpe":
                            loss = compute_trading_sharpe_loss(output, y_batch)
                        elif self.loss_function == "portfolio_sharpe":
                            loss = compute_portfolio_sharpe_loss(output, y_batch)
                        elif self.loss_function == "mse":
                            loss = nn.MSELoss()(output, y_batch)
                        else:
                            raise ValueError(f"Unknown loss function: {self.loss_function}")

                        val_loss += loss.item() * X_batch.size(0)
                val_loss /= len(val_loader.dataset)
                training_history['val_loss'].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation set, use training loss for early stopping
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_model_state = self.model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

            # Step scheduler
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                val_str = f", Val Loss={val_loss:.6f}" if val_loss is not None else ""
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}{val_str}, LR={lr:.6f}")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_history['epochs_trained'] = epoch + 1
        training_history['best_val_loss'] = best_val_loss

        self.is_trained = True

        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
        return training_history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model with batch processing for BatchNorm consistency.

        Args:
            X: Input features of shape (n_samples, look_back, n_features)

        Returns:
            Predictions of shape (n_samples, n_features)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.model.eval()
        
        # Use same batch size as training for BatchNorm consistency
        batch_size = self.batch_size  # 64
        n_samples = X.shape[0]
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X[i:end_idx]
                X_tensor = torch.tensor(X_batch, dtype=torch.float32)
                
                batch_predictions = self.model(X_tensor.to(self.device)).cpu().numpy()
                all_predictions.append(batch_predictions)
        
        predictions = np.vstack(all_predictions)
        print(f"Predictions generated using batch size {batch_size} for BatchNorm consistency")
        return predictions

    def get_model_name(self) -> str:
        """Return the model name."""
        return "CNNTransformer"

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.

        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'lookback': self.lookback,
            'model_params': self.model_params,
            'is_trained': self.is_trained
        }

        torch.save(model_data, filepath)
        print(f"CNN+Transformer model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.

        Args:
            filepath: Path from where to load the model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = torch.load(filepath, map_location=self.device)

        # Update model parameters
        self.model_params.update(model_data['model_params'])
        self.input_size = model_data['input_size']
        self.output_size = model_data['output_size']
        self.lookback = model_data['lookback']
        self.is_trained = model_data['is_trained']

        # Initialize and load model
        self._initialize_model(self.input_size, self.output_size, self.lookback)
        self.model.load_state_dict(model_data['model_state_dict'])

        print(f"CNN+Transformer model loaded from {filepath}")

# class CNNTransformerWeightModel(BaseTimeSeriesModel):
#     """
#     CNN+Transformer model that generates portfolio weights for yield spreads.
#     This model learns to allocate weights across different yield spreads.
#     """
#
#     def __init__(self, filter_numbers: List[int] = [16, 32], attention_heads: int = 4,
#                  hidden_units_factor: int = 2, dropout: float = 0.25, filter_size: int = 3,
#                  batch_size: int = 64, learning_rate: float = 0.001,
#                  num_epochs: int = 60, patience: int = 10, device: Optional[str] = None,
#                  weight_decay: float = 1e-5, step_size: int = 10, gamma: float = 0.5,
#                  loss_function: str = "portfolio_sharpe", weight_mode: str = "softmax"):
#         """
#         Initialize CNN+Transformer weight generation model.
#
#         Args:
#             filter_numbers: List of filter numbers for CNN blocks
#             attention_heads: Number of attention heads in transformer
#             hidden_units_factor: Factor for feedforward dimension in transformer
#             dropout: Dropout rate
#             filter_size: Kernel size for CNN blocks
#             batch_size: Training batch size
#             learning_rate: Learning rate for optimizer
#             num_epochs: Maximum number of training epochs
#             patience: Early stopping patience
#             device: Device to use ('cuda', 'cpu', or None for auto-detect)
#             weight_decay: Weight decay for regularization
#             step_size: Step size for learning rate scheduler
#             gamma: Gamma for learning rate scheduler
#             loss_function: Loss function to use ('portfolio_sharpe', 'trading_sharpe', or 'mse')
#         """
#         super().__init__(
#             filter_numbers=filter_numbers,
#             attention_heads=attention_heads,
#             hidden_units_factor=hidden_units_factor,
#             dropout=dropout,
#             filter_size=filter_size,
#             batch_size=batch_size,
#             learning_rate=learning_rate,
#             num_epochs=num_epochs,
#             patience=patience,
#             device=device,
#             weight_decay=weight_decay,
#             step_size=step_size,
#             gamma=gamma
#         )
#
#         self.filter_numbers = filter_numbers
#         self.attention_heads = attention_heads
#         self.hidden_units_factor = hidden_units_factor
#         self.dropout = dropout
#         self.filter_size = filter_size
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.num_epochs = num_epochs
#         self.patience = patience
#         self.weight_decay = weight_decay
#         self.step_size = step_size
#         self.gamma = gamma
#         self.loss_function = loss_function
#
#         # Set device
#         if device is None:
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         else:
#             self.device = torch.device(device)
#
#         # Store weight mode for later use
#         self.weight_mode = weight_mode
#
#         # Model will be initialized during fit()
#         self.model = None
#         self.input_size = None
#         self.output_size = None
#         self.lookback = None
#
#     def _initialize_model(self, input_size: int, output_size: int, lookback: int):
#         """Initialize the CNN+Transformer weight generation model with given dimensions."""
#         self.input_size = input_size
#         self.output_size = output_size
#         self.lookback = lookback
#
#         self.model = CNNTransformerWeightGenerator(
#             input_features=input_size,
#             lookback=lookback,
#             filter_numbers=self.filter_numbers,
#             attention_heads=self.attention_heads,
#             hidden_units_factor=self.hidden_units_factor,
#             dropout=self.dropout,
#             filter_size=self.filter_size,
#             weight_mode=self.weight_mode
#         ).to(self.device)
#
#     def fit(self, X_train: np.ndarray, y_train: np.ndarray,
#             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
#         """
#         Train the CNN+Transformer weight generation model.
#
#         Args:
#             X_train: Training features of shape (n_samples, look_back, n_features)
#             y_train: Training targets of shape (n_samples, n_features) - yield spread changes
#             X_val: Validation features (optional)
#             y_val: Validation targets (optional)
#
#         Returns:
#             Dictionary containing training metrics
#         """
#         # Initialize model if not done already
#         if self.model is None:
#             self._initialize_model(X_train.shape[2], y_train.shape[1], X_train.shape[1])
#
#         # Convert to tensors
#         X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#         y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
#
#         # Create data loaders
#         train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#         train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
#
#         val_loader = None
#         if X_val is not None and y_val is not None:
#             X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
#             y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
#             val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
#             val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
#
#         # Loss and optimizer
#         optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size, gamma=self.gamma)
#
#         # Training loop with early stopping
#         best_val_loss = float('inf')
#         epochs_no_improve = 0
#         best_model_state = None
#         training_history = {
#             'train_loss': [],
#             'val_loss': [],
#             'epochs_trained': 0
#         }
#
#         print(f"Starting CNN+Transformer weight generation training on {self.device}...")
#
#         for epoch in range(self.num_epochs):
#             # Training
#             self.model.train()
#             train_loss = 0
#             for X_batch, y_batch in train_loader:
#                 X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
#                 optimizer.zero_grad()
#
#                 # Generate portfolio weights
#                 weights = self.model(X_batch)  # (batch_size, n_features)
#
#                 # Use selected loss function
#                 if self.loss_function == "portfolio_sharpe":
#                     loss = compute_portfolio_sharpe_loss(weights, y_batch)
#                 elif self.loss_function == "trading_sharpe":
#                     loss = compute_trading_sharpe_loss(weights, y_batch)
#                 elif self.loss_function == "mse":
#                     loss = nn.MSELoss()(weights, y_batch)
#                 else:
#                     raise ValueError(f"Unknown loss function: {self.loss_function}")
#
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item() * X_batch.size(0)
#             train_loss /= len(train_loader.dataset)
#             training_history['train_loss'].append(train_loss)
#
#             # Validation
#             val_loss = None
#             if val_loader is not None:
#                 self.model.eval()
#                 val_loss = 0
#                 with torch.no_grad():
#                     for X_batch, y_batch in val_loader:
#                         X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
#                         weights = self.model(X_batch)
#
#                         # Use selected loss function for validation too
#                         if self.loss_function == "portfolio_sharpe":
#                             loss = compute_portfolio_sharpe_loss(weights, y_batch)
#                         elif self.loss_function == "trading_sharpe":
#                             loss = compute_trading_sharpe_loss(weights, y_batch)
#                         elif self.loss_function == "mse":
#                             loss = nn.MSELoss()(weights, y_batch)
#                         else:
#                             raise ValueError(f"Unknown loss function: {self.loss_function}")
#
#                         val_loss += loss.item() * X_batch.size(0)
#                 val_loss /= len(val_loader.dataset)
#                 training_history['val_loss'].append(val_loss)
#
#                 # Early stopping check
#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     best_model_state = self.model.state_dict()
#                     epochs_no_improve = 0
#                 else:
#                     epochs_no_improve += 1
#                     if epochs_no_improve >= self.patience:
#                         print(f"Early stopping at epoch {epoch+1}")
#                         break
#             else:
#                 # No validation set, use training loss for early stopping
#                 if train_loss < best_val_loss:
#                     best_val_loss = train_loss
#                     best_model_state = self.model.state_dict()
#                     epochs_no_improve = 0
#                 else:
#                     epochs_no_improve += 1
#                     if epochs_no_improve >= self.patience:
#                         print(f"Early stopping at epoch {epoch+1}")
#                         break
#
#             # Step scheduler
#             scheduler.step()
#
#             if (epoch + 1) % 10 == 0:
#                 val_str = f", Val Loss={val_loss:.6f}" if val_loss is not None else ""
#                 lr = scheduler.get_last_lr()[0]
#                 print(f"Epoch {epoch+1}: Train Loss={train_loss:.6f}{val_str}, LR={lr:.6f}")
#
#         # Restore best model
#         if best_model_state is not None:
#             self.model.load_state_dict(best_model_state)
#
#         training_history['epochs_trained'] = epoch + 1
#         training_history['best_val_loss'] = best_val_loss
#
#         self.is_trained = True
#
#         print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
#         return training_history
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Generate portfolio weights using the trained model.
#
#         Args:
#             X: Input features of shape (n_samples, look_back, n_features)
#
#         Returns:
#             Portfolio weights of shape (n_samples, n_features)
#         """
#         if not self.is_trained:
#             raise ValueError("Model must be trained before making predictions")
#
#         self.model.eval()
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#
#         with torch.no_grad():
#             weights = self.model(X_tensor.to(self.device)).cpu().numpy()
#
#         return weights
#
#     def get_model_name(self) -> str:
#         """Return the model name."""
#         return "CNNTransformerWeight"
#
#     def save_model(self, filepath: str) -> None:
#         """
#         Save the trained model to disk.
#
#         Args:
#             filepath: Path where to save the model
#         """
#         if not self.is_trained:
#             raise ValueError("Cannot save untrained model")
#
#         model_data = {
#             'model_state_dict': self.model.state_dict(),
#             'input_size': self.input_size,
#             'output_size': self.output_size,
#             'lookback': self.lookback,
#             'model_params': self.model_params,
#             'is_trained': self.is_trained
#         }
#
#         torch.save(model_data, filepath)
#         print(f"CNN+Transformer weight model saved to {filepath}")
#
#     def load_model(self, filepath: str) -> None:
#         """
#         Load a trained model from disk.
#
#         Args:
#             filepath: Path from where to load the model
#         """
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"Model file not found: {filepath}")
#
#         model_data = torch.load(filepath, map_location=self.device)
#
#         # Update model parameters
#         self.model_params.update(model_data['model_params'])
#         self.input_size = model_data['input_size']
#         self.output_size = model_data['output_size']
#         self.lookback = model_data['lookback']
#         self.is_trained = model_data['is_trained']
#
#         # Initialize and load model
#         self._initialize_model(self.input_size, self.output_size, self.lookback)
#         self.model.load_state_dict(model_data['model_state_dict'])
#
#         print(f"CNN+Transformer weight model loaded from {filepath}")

# Register the CNN+Transformer models with the factory
# ModelFactory.register_model("CNNTransformer", CNNTransformerModel)
# ModelFactory.register_model("CNNTransformerWeight", CNNTransformerWeightModel)
