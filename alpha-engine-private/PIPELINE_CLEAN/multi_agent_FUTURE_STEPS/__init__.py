"""
Multi-Agent Alpha Synthesis

Each ticker gets its own specialized agent trained on ticker-specific features.
This avoids state pollution from cross-sectional averaging.
"""

from .config import TRAINING_TICKERS, TICKER_CATEGORIES, HYPERPARAMETERS