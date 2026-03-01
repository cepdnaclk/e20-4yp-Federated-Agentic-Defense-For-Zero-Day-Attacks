"""
Utility functions for federated learning model weight handling.

This module provides functions to extract, serialize, and deserialize
model weights for federated aggregation. It handles both PyTorch (Autoencoder)
and XGBoost models, converting them to numpy arrays for Flower compatibility.

Key Functions:
    - autoencoder_weights_to_numpy: Extract PyTorch weights as numpy arrays
    - numpy_to_autoencoder_weights: Load numpy arrays into PyTorch model
    - xgboost_to_numpy: Serialize XGBoost model to numpy-compatible format
    - numpy_to_xgboost: Deserialize numpy arrays back to XGBoost model
"""

import logging
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

logger = logging.getLogger(__name__)


def autoencoder_weights_to_numpy(
    model: torch.nn.Module,
) -> List[np.ndarray]:
    """
    Extracts PyTorch model weights as a list of numpy arrays.
    
    This function converts all trainable parameters from a PyTorch model
    into numpy arrays suitable for federated aggregation. The order of
    parameters is preserved to ensure correct reconstruction.
    
    Args:
        model: PyTorch model (AnomalyAutoencoder).
    
    Returns:
        List of numpy arrays, one per parameter tensor.
        Order: [encoder_weights, encoder_biases, ..., decoder_weights, ...]
    
    Example:
        >>> from agents.models.autoencoder import AnomalyAutoencoder
        >>> model = AnomalyAutoencoder(input_dim=40, latent_dim=8)
        >>> weights = autoencoder_weights_to_numpy(model)
        >>> len(weights)  # Number of parameter tensors
        20
    
    Note:
        Non-trainable parameters (e.g., batch norm running stats) are
        included for complete state restoration.
    """
    weights = []
    for name, param in model.state_dict().items():
        weights.append(param.cpu().numpy().copy())
        logger.debug(f"Extracted parameter: {name}, shape: {param.shape}")
    
    return weights


def numpy_to_autoencoder_weights(
    model: torch.nn.Module,
    weights: List[np.ndarray],
    strict: bool = True,
) -> torch.nn.Module:
    """
    Loads numpy arrays into a PyTorch model's parameters.
    
    This function reconstructs model state from federated-aggregated
    numpy arrays. The arrays must be in the same order as extracted
    by autoencoder_weights_to_numpy.
    
    Args:
        model: PyTorch model to update.
        weights: List of numpy arrays from federated aggregation.
        strict: If True, requires exact match of parameter count.
    
    Returns:
        Updated PyTorch model.
    
    Raises:
        ValueError: If weight count doesn't match model parameters.
    
    Example:
        >>> model = AnomalyAutoencoder(input_dim=40, latent_dim=8)
        >>> updated_model = numpy_to_autoencoder_weights(model, aggregated_weights)
    """
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    
    if len(weights) != len(keys):
        msg = f"Weight count mismatch: got {len(weights)}, expected {len(keys)}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)
    
    # Update state dict with new weights
    new_state_dict = {}
    for i, (key, weight) in enumerate(zip(keys, weights)):
        # Convert to tensor with same dtype as original
        original_dtype = state_dict[key].dtype
        new_state_dict[key] = torch.tensor(weight, dtype=original_dtype)
        logger.debug(f"Updated parameter: {key}")
    
    model.load_state_dict(new_state_dict, strict=strict)
    logger.info(f"Loaded {len(weights)} parameters into autoencoder")
    
    return model


def xgboost_to_numpy(
    model: xgb.XGBClassifier,
    label_encoder: Optional[Any] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Serializes an XGBoost model to numpy-compatible format.
    
    XGBoost models are tree-based and don't have traditional "weights"
    like neural networks. Instead, we serialize the model to JSON and
    encode it as a numpy array of bytes for federated transfer.
    
    Args:
        model: Trained XGBClassifier.
        label_encoder: Optional sklearn LabelEncoder for classes.
    
    Returns:
        Tuple of:
            - numpy array containing serialized model bytes
            - metadata dict with model config
    
    Example:
        >>> from xgboost import XGBClassifier
        >>> model = XGBClassifier()
        >>> model.fit(X_train, y_train)
        >>> serialized, metadata = xgboost_to_numpy(model)
    
    Note:
        The serialized format preserves the complete model structure
        including all trees, feature importances, and hyperparameters.
    """
    # Save model to JSON string
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        model.save_model(temp_path)
        with open(temp_path, 'r') as f:
            model_json = f.read()
    finally:
        Path(temp_path).unlink(missing_ok=True)
    
    # Convert JSON string to bytes, then to numpy array
    model_bytes = model_json.encode('utf-8')
    serialized = np.frombuffer(model_bytes, dtype=np.uint8).copy()
    
    # Build metadata
    metadata = {
        "n_estimators": int(model.n_estimators) if hasattr(model, 'n_estimators') else None,
        "max_depth": int(model.max_depth) if hasattr(model, 'max_depth') else None,
        "learning_rate": float(model.learning_rate) if hasattr(model, 'learning_rate') else None,
        "n_classes": int(model.n_classes_) if hasattr(model, 'n_classes_') else None,
        "serialization_format": "json_bytes",
    }
    
    # Include label encoder classes if provided
    if label_encoder is not None:
        metadata["classes"] = label_encoder.classes_.tolist()
    
    logger.info(f"Serialized XGBoost model: {len(serialized)} bytes")
    
    return serialized, metadata


def numpy_to_xgboost(
    serialized: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> xgb.XGBClassifier:
    """
    Deserializes numpy bytes back to an XGBoost model.
    
    Reconstructs the XGBoost model from serialized bytes created by
    xgboost_to_numpy. This enables federated aggregation results
    to be loaded back into a functional classifier.
    
    Args:
        serialized: Numpy array containing model bytes.
        metadata: Optional metadata dict (not used for deserialization,
                  but can be used for validation).
    
    Returns:
        Reconstructed XGBClassifier.
    
    Raises:
        ValueError: If deserialization fails.
    
    Example:
        >>> model = numpy_to_xgboost(serialized_weights, metadata)
        >>> predictions = model.predict(X_test)
    """
    # Convert numpy array back to bytes, then to JSON string
    model_bytes = serialized.tobytes()
    model_json = model_bytes.decode('utf-8')
    
    # Write to temp file and load
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write(model_json)
        temp_path = f.name
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(temp_path)
    except Exception as e:
        raise ValueError(f"Failed to deserialize XGBoost model: {e}") from e
    finally:
        Path(temp_path).unlink(missing_ok=True)
    
    logger.info("Deserialized XGBoost model successfully")
    
    return model


def get_combined_weights(
    autoencoder_model: torch.nn.Module,
    xgboost_model: xgb.XGBClassifier,
    label_encoder: Optional[Any] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Combines weights from both Autoencoder and XGBoost into a single list.
    
    This function prepares a unified weight representation for federated
    learning, allowing both models to be updated in a single round.
    
    Args:
        autoencoder_model: PyTorch autoencoder.
        xgboost_model: XGBoost classifier.
        label_encoder: Optional LabelEncoder for XGBoost classes.
    
    Returns:
        Tuple of:
            - List of numpy arrays (autoencoder weights + XGBoost serialized)
            - Metadata dict describing the weight structure
    
    Example:
        >>> weights, metadata = get_combined_weights(agent_one.model, agent_two.classifier)
    """
    # Get autoencoder weights
    ae_weights = autoencoder_weights_to_numpy(autoencoder_model)
    
    # Get XGBoost serialized
    xgb_serialized, xgb_metadata = xgboost_to_numpy(xgboost_model, label_encoder)
    
    # Combine: autoencoder weights list + XGBoost as single array
    combined = ae_weights + [xgb_serialized]
    
    metadata = {
        "autoencoder_weight_count": len(ae_weights),
        "xgboost_metadata": xgb_metadata,
        "total_weight_count": len(combined),
    }
    
    return combined, metadata


def split_combined_weights(
    combined_weights: List[np.ndarray],
    autoencoder_weight_count: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Splits combined weights back into Autoencoder and XGBoost components.
    
    Args:
        combined_weights: Combined weight list from federated aggregation.
        autoencoder_weight_count: Number of autoencoder weight arrays.
    
    Returns:
        Tuple of:
            - List of autoencoder weight arrays
            - XGBoost serialized bytes array
    """
    ae_weights = combined_weights[:autoencoder_weight_count]
    xgb_serialized = combined_weights[autoencoder_weight_count]
    
    return ae_weights, xgb_serialized


def aggregate_xgboost_models(
    serialized_models: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Aggregates multiple XGBoost models using weighted averaging of predictions.
    
    Since XGBoost models are tree-based and cannot be averaged directly,
    this function creates an ensemble by averaging the raw JSON models.
    For a true ensemble, use prediction averaging at inference time.
    
    Note:
        This is a simplified aggregation that returns the first model.
        For production, implement proper ensemble methods or use
        tree-based aggregation techniques.
    
    Args:
        serialized_models: List of serialized XGBoost model bytes.
        weights: Optional weights for each model (default: equal weights).
    
    Returns:
        Aggregated model as serialized bytes. Currently returns first model.
    
    Warning:
        XGBoost aggregation is non-trivial. This implementation serves
        as a placeholder. Consider using:
        - Prediction averaging (ensemble)
        - Feature space averaging via embedding layers
        - Gradient-based aggregation for XGBoost
    """
    if not serialized_models:
        raise ValueError("No models to aggregate")
    
    # TODO: Implement proper XGBoost aggregation
    # For now, return the first model (simulating selection-based aggregation)
    # In production, you might want to:
    # 1. Use a meta-learner to combine predictions
    # 2. Average leaf node predictions
    # 3. Use federated boosting techniques
    
    logger.warning(
        "XGBoost aggregation returns first model. "
        "Implement proper aggregation for production use."
    )
    
    return serialized_models[0]
