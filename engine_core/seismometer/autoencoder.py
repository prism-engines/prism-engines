"""
Reconstruction Error Detector - Linear Autoencoder based anomaly detection

Pure numpy implementation - no torch/tensorflow dependency.
Detects anomalies via reconstruction error from a learned encoding.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
import logging

from .base import BaseDetector

logger = logging.getLogger(__name__)


class ReconstructionErrorDetector(BaseDetector):
    """
    Linear autoencoder for reconstruction error based anomaly detection.

    Architecture: X -> tanh(X @ W_enc + b_enc) -> Z @ W_dec + b_dec -> X_recon

    High reconstruction error indicates the observation differs significantly
    from patterns learned during baseline fitting.
    """

    def __init__(
        self,
        encoding_dim: int = 8,
        lookback_days: int = 252,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        batch_size: int = 32,
        percentile_threshold: float = 95.0,
        random_state: int = 42
    ):
        """
        Initialize reconstruction error detector.

        Args:
            encoding_dim: Dimension of the latent encoding
            lookback_days: Lookback window for calculations
            learning_rate: Learning rate for gradient descent
            n_epochs: Number of training epochs
            batch_size: Mini-batch size for training
            percentile_threshold: Percentile for normalizing errors
            random_state: Random seed for reproducibility
        """
        super().__init__('reconstruction_error', lookback_days)
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.percentile_threshold = percentile_threshold
        self.random_state = random_state

        # Model parameters (initialized during fit)
        self._W_enc: Optional[np.ndarray] = None
        self._b_enc: Optional[np.ndarray] = None
        self._W_dec: Optional[np.ndarray] = None
        self._b_dec: Optional[np.ndarray] = None
        self._input_dim: Optional[int] = None
        self._feature_names: Optional[list] = None

    def fit(self, features: pd.DataFrame) -> 'ReconstructionErrorDetector':
        """
        Fit autoencoder on baseline period.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            self for method chaining
        """
        df = self._validate_features(features)
        self._feature_names = list(df.columns)
        self._input_dim = len(self._feature_names)

        # Get feature matrix and handle NaN
        X = df.values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_valid = X[valid_mask]

        if len(X_valid) < 50:
            logger.warning(f"Limited baseline data: {len(X_valid)} samples")

        # Standardize
        X_scaled = self._standardize(X_valid, fit=True)

        # Initialize weights
        np.random.seed(self.random_state)
        self._init_weights()

        # Train via mini-batch gradient descent
        n_samples = len(X_scaled)
        best_loss = float('inf')

        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_scaled[indices]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, self.batch_size):
                batch = X_shuffled[i:i + self.batch_size]
                if len(batch) == 0:
                    continue

                # Forward pass
                hidden, reconstruction = self._forward(batch)

                # Backward pass and update
                loss = self._backward(batch, hidden, reconstruction)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch + 1}/{self.n_epochs}, loss: {avg_loss:.6f}")

        # Compute baseline reconstruction errors
        _, reconstructions = self._forward(X_scaled)
        errors = np.mean((X_scaled - reconstructions) ** 2, axis=1)

        # Store baseline statistics
        self._baseline_stats = {
            'error_mean': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'error_p95': float(np.percentile(errors, self.percentile_threshold)),
            'error_max': float(np.max(errors)),
            'final_loss': float(best_loss),
            'n_samples': len(X_valid),
        }

        self.is_fitted = True
        logger.info(
            f"Fitted {self.name}: encoding_dim={self.encoding_dim}, "
            f"baseline p95 error = {self._baseline_stats['error_p95']:.6f}"
        )

        return self

    def score(self, features: pd.DataFrame) -> pd.Series:
        """
        Score observations by reconstruction error.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            Series with instability scores in [0, 1]
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted - call fit() first")

        df = self._validate_features(features)

        # Ensure same features
        missing_features = set(self._feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")

        # Align columns
        df = df[self._feature_names]

        # Get feature matrix
        X = df.values

        # Initialize scores with NaN
        scores = np.full(len(df), np.nan)

        # Find valid rows
        valid_mask = ~np.isnan(X).any(axis=1)

        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            X_scaled = self._standardize(X_valid, fit=False)

            # Compute reconstruction errors
            _, reconstructions = self._forward(X_scaled)
            errors = np.mean((X_scaled - reconstructions) ** 2, axis=1)

            # Normalize against baseline p95
            p95 = self._baseline_stats['error_p95']
            if p95 > 0:
                normalized = errors / p95
            else:
                normalized = errors

            # Clip to [0, 1]
            scores[valid_mask] = self._clip_scores(normalized)

        return pd.Series(scores, index=df.index, name=self.name)

    def _init_weights(self) -> None:
        """Initialize encoder and decoder weights using Xavier initialization."""
        # Xavier initialization
        enc_scale = np.sqrt(2.0 / (self._input_dim + self.encoding_dim))
        dec_scale = np.sqrt(2.0 / (self.encoding_dim + self._input_dim))

        self._W_enc = np.random.randn(self._input_dim, self.encoding_dim) * enc_scale
        self._b_enc = np.zeros(self.encoding_dim)
        self._W_dec = np.random.randn(self.encoding_dim, self._input_dim) * dec_scale
        self._b_dec = np.zeros(self._input_dim)

    def _forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through autoencoder.

        Args:
            X: Input array (batch_size, input_dim)

        Returns:
            Tuple of (hidden activations, reconstructions)
        """
        # Encoder: tanh activation
        hidden = np.tanh(X @ self._W_enc + self._b_enc)

        # Decoder: linear output
        reconstruction = hidden @ self._W_dec + self._b_dec

        return hidden, reconstruction

    def _backward(
        self,
        X: np.ndarray,
        hidden: np.ndarray,
        reconstruction: np.ndarray
    ) -> float:
        """
        Backward pass and weight update.

        Args:
            X: Input array
            hidden: Hidden layer activations
            reconstruction: Reconstructed outputs

        Returns:
            Mean squared error loss
        """
        batch_size = len(X)

        # Compute loss
        error = reconstruction - X
        loss = np.mean(error ** 2)

        # Gradient of decoder
        d_W_dec = hidden.T @ error / batch_size
        d_b_dec = np.mean(error, axis=0)

        # Gradient through decoder
        d_hidden = error @ self._W_dec.T

        # Gradient through tanh
        d_hidden = d_hidden * (1 - hidden ** 2)

        # Gradient of encoder
        d_W_enc = X.T @ d_hidden / batch_size
        d_b_enc = np.mean(d_hidden, axis=0)

        # Update weights (gradient descent)
        self._W_dec -= self.learning_rate * d_W_dec
        self._b_dec -= self.learning_rate * d_b_dec
        self._W_enc -= self.learning_rate * d_W_enc
        self._b_enc -= self.learning_rate * d_b_enc

        return loss

    def get_encoding(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Get latent encodings for observations.

        Args:
            features: DataFrame with date index and feature columns

        Returns:
            DataFrame with encoding dimensions as columns
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted - call fit() first")

        df = self._validate_features(features)
        df = df[self._feature_names]

        X = df.values
        encodings = np.full((len(df), self.encoding_dim), np.nan)

        valid_mask = ~np.isnan(X).any(axis=1)
        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            X_scaled = self._standardize(X_valid, fit=False)
            hidden, _ = self._forward(X_scaled)
            encodings[valid_mask] = hidden

        encoding_cols = [f'enc_{i}' for i in range(self.encoding_dim)]
        return pd.DataFrame(encodings, index=df.index, columns=encoding_cols)
