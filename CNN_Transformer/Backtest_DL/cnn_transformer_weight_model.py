"""
CNN+Transformer Weight Generation Model for Yield Spread Portfolio Allocation
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
from cnn_transformer_model import (
    PositionalEncoding, CNN_Block, 
    compute_portfolio_sharpe_loss, compute_trading_sharpe_loss
)

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
    ):
        super(CNNTransformerWeightGenerator, self).__init__()

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
        # Default: softmax for long-only portfolio (weights sum to 1)
        # Can be changed to tanh for signed weights (shorting capability)
        self.weight_mode = "softmax"  # Options: "softmax", "tanh"

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
        
        # Apply weight mode
        if self.weight_mode == "softmax":
            # Long-only portfolio (weights sum to 1)
            weights = torch.softmax(weights, dim=1)
        elif self.weight_mode == "tanh":
            # Signed weights for shorting (range: [-1, +1])
            weights = torch.tanh(weights)
        else:
            raise ValueError(f"Unknown weight_mode: {self.weight_mode}")
        
        return weights

class CNNTransformerWeightModel(BaseTimeSeriesModel):
    """
    CNN+Transformer model that generates portfolio weights for yield spreads.
    This model learns to allocate weights across different yield spreads.
    """
    
    def __init__(self, filter_numbers: List[int] = [16, 32], attention_heads: int = 4,
                 hidden_units_factor: int = 2, dropout: float = 0.25, filter_size: int = 3,
                 batch_size: int = 64, learning_rate: float = 0.001, 
                 num_epochs: int = 60, patience: int = 10, device: Optional[str] = None,
                 weight_decay: float = 1e-5, step_size: int = 10, gamma: float = 0.5,
                 loss_function: str = "portfolio_sharpe", weight_mode: str = "softmax"):
        """
        Initialize CNN+Transformer weight generation model.
        
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
            loss_function: Loss function to use ('portfolio_sharpe', 'trading_sharpe', or 'mse')
            weight_mode: Weight generation mode ('softmax' for long-only, 'tanh' for shorting)
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
        self.weight_mode = weight_mode
        
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
        """Initialize the CNN+Transformer weight generation model with given dimensions."""
        self.input_size = input_size
        self.output_size = output_size
        self.lookback = lookback
        
        self.model = CNNTransformerWeightGenerator(
            input_features=input_size,
            lookback=lookback,
            filter_numbers=self.filter_numbers,
            attention_heads=self.attention_heads,
            hidden_units_factor=self.hidden_units_factor,
            dropout=self.dropout,
            filter_size=self.filter_size
        ).to(self.device)
        
        # Set weight mode in the model
        self.model.weight_mode = self.weight_mode
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the CNN+Transformer weight generation model.
        
        Args:
            X_train: Training features of shape (n_samples, look_back, n_features)
            y_train: Training targets of shape (n_samples, n_features) - yield spread changes
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
        
        # Loss and optimizer
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
        
        print(f"Starting CNN+Transformer weight generation training on {self.device}...")
        
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                
                # Generate portfolio weights
                weights = self.model(X_batch)  # (batch_size, n_features)
                
                # Use selected loss function
                if self.loss_function == "portfolio_sharpe":
                    loss = compute_portfolio_sharpe_loss(weights, y_batch)
                elif self.loss_function == "trading_sharpe":
                    loss = compute_trading_sharpe_loss(weights, y_batch)
                elif self.loss_function == "mse":
                    loss = nn.MSELoss()(weights, y_batch)
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
                        weights = self.model(X_batch)
                        
                        # Use selected loss function for validation too
                        if self.loss_function == "portfolio_sharpe":
                            loss = compute_portfolio_sharpe_loss(weights, y_batch)
                        elif self.loss_function == "trading_sharpe":
                            loss = compute_trading_sharpe_loss(weights, y_batch)
                        elif self.loss_function == "mse":
                            loss = nn.MSELoss()(weights, y_batch)
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
        Generate portfolio weights using the trained model with batch processing for BatchNorm consistency.
        
        Args:
            X: Input features of shape (n_samples, look_back, n_features)
            
        Returns:
            Portfolio weights of shape (n_samples, n_features)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        # Use same batch size as training for BatchNorm consistency
        batch_size = self.batch_size  # 64
        n_samples = X.shape[0]
        all_weights = []
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                end_idx = min(i + batch_size, n_samples)
                X_batch = X[i:end_idx]
                X_tensor = torch.tensor(X_batch, dtype=torch.float32)
                
                batch_weights = self.model(X_tensor.to(self.device)).cpu().numpy()
                all_weights.append(batch_weights)
        
        weights = np.vstack(all_weights)
        print(f"Weights generated using batch size {batch_size} for BatchNorm consistency")
        return weights
    
    def get_model_name(self) -> str:
        """Return the model name."""
        return "CNNTransformerWeight"
    
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
        print(f"CNN+Transformer weight model saved to {filepath}")
    
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
        
        print(f"CNN+Transformer weight model loaded from {filepath}")

# Register the CNN+Transformer weight model with the factory
ModelFactory.register_model("CNNTransformerWeight", CNNTransformerWeightModel)

