"""
Configuration for Multi-Agent Training

Centralized configuration for:
- Ticker lists by category
- Hyperparameters
- Path settings
"""

import os

# =============================================================================
# PATHS
# =============================================================================

_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(_CONFIG_DIR)
DATA_DIR = os.path.join(PIPELINE_DIR, "data")
MODELS_DIR = os.path.join(_CONFIG_DIR, "models")
ALPHAS_DIR = os.path.join(_CONFIG_DIR, "alphas")

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ALPHAS_DIR, exist_ok=True)

# =============================================================================
# TICKERS BY CATEGORY
# =============================================================================

TICKER_CATEGORIES = {
    'tech': ['AAPL', 'MSFT', 'GOOGL', 'INTC', 'CSCO'],
    'finance': ['JPM', 'BAC', 'GS', 'WFC'],
    'healthcare': ['JNJ', 'PFE', 'UNH', 'MRK'],
    'consumer': ['WMT', 'KO', 'PG', 'MCD'],
    'industrial': ['GE', 'CAT', 'BA'],
    'energy': ['XOM', 'CVX'],
}

# Flat list of all training tickers
TRAINING_TICKERS = [
    ticker 
    for tickers in TICKER_CATEGORIES.values() 
    for ticker in tickers
]

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

HYPERPARAMETERS = {
    # Environment
    'max_steps': 20,
    'min_expression_depth': 3,
    'eval_window': 20,
    'sharpe_lookback': 60,
    
    # Agent architecture
    'hidden_size': 256,
    'learning_rate': 0.0001,
    
    # PPO parameters
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'eps_clip': 0.2,
    'k_epochs': 4,
    'value_coef': 0.5,
    'entropy_coef': 0.10,
    
    # Training
    'n_episodes': 50000,      # Per ticker (less than multi-ticker since focused)
    'batch_episodes': 32,     # Smaller batches since single ticker
    'print_every': 500,
    
    # Feature windows
    'short_window': 5,
    'medium_window': 20,
    'long_window': 60,
    
    # Window options for TS operators
    'window_options': [2, 3, 5, 10, 20, 50, 120],
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_model_path(ticker: str) -> str:
    """Get the path for a ticker's model file."""
    return os.path.join(MODELS_DIR, f"agent_{ticker}.pt")

def get_alpha_library_path(ticker: str) -> str:
    """Get the path for a ticker's alpha library."""
    return os.path.join(ALPHAS_DIR, f"alpha_library_{ticker}.json")

def get_category(ticker: str) -> str:
    """Get the category for a ticker."""
    for category, tickers in TICKER_CATEGORIES.items():
        if ticker in tickers:
            return category
    return 'unknown'