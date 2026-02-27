"""
Abstract base class for time series prediction models
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any

class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series prediction models.
    
    This class defines the interface that all models must implement
    to be compatible with the backtesting system.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the model with any required parameters.
        
        Args:
            **kwargs: Model-specific parameters
        """
        self.is_trained = False
        self.model_params = kwargs
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features of shape (n_samples, look_back, n_features)
            y_train: Training targets of shape (n_samples, n_features)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Dictionary containing training metrics and information
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Input features of shape (n_samples, look_back, n_features)
            
        Returns:
            Predictions of shape (n_samples, n_features)
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return the name of the model for identification.
        
        Returns:
            String name of the model
        """
        pass
    
    def get_model_params(self) -> Dict[str, Any]:
        """
        Return the model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return self.model_params.copy()
    
    def is_model_trained(self) -> bool:
        """
        Check if the model has been trained.
        
        Returns:
            True if model is trained, False otherwise
        """
        return self.is_trained
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        raise NotImplementedError("Model saving not implemented for this model type")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path from where to load the model
        """
        raise NotImplementedError("Model loading not implemented for this model type")

class ModelFactory:
    """
    Factory class for creating model instances.
    """
    
    _models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class):
        """
        Register a model class with the factory.
        
        Args:
            name: Name to register the model under
            model_class: The model class to register
        """
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, name: str, **kwargs) -> BaseTimeSeriesModel:
        """
        Create a model instance by name.
        
        Args:
            name: Name of the model to create
            **kwargs: Parameters to pass to the model constructor
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model name is not registered
        """
        if name not in cls._models:
            available_models = list(cls._models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available_models}")
        
        return cls._models[name](**kwargs)
    
    @classmethod
    def list_models(cls) -> list:
        """
        List all registered models.
        
        Returns:
            List of registered model names
        """
        return list(cls._models.keys())
