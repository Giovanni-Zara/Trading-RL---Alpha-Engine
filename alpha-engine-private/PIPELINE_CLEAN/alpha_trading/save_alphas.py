"""
Save discovered alphas from training to the alpha library.

Run this after train_PPO.py to persist the best alphas.
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from alpha_library import AlphaLibrary


def save_alphas_from_history(
    history: dict,
    library_path: str = "alphas/alpha_library.json",
    top_n: int = 20,
    min_sharpe: float = 0.0,
    ticker: str = "",
    period: str = ""
):
    """
    Save the best alphas from training history to library.
    
    Parameters:
        history: Training history dict
        library_path: Where to save the library
        top_n: Number of top alphas to keep
        min_sharpe: Minimum Sharpe ratio threshold
        ticker: Training ticker (metadata)
        period: Training period (metadata)
    """
    # Load existing library or create new
    if os.path.exists(library_path):
        library = AlphaLibrary.load(library_path)
        print(f"Loaded existing library with {len(library.alphas)} alphas")
    else:
        library = AlphaLibrary(name="discovered_alphas")
        print("Created new alpha library")
    
    # Add alphas from training
    added = library.add_from_training_history(
        history=history,
        top_n=top_n,
        min_sharpe=min_sharpe,
        ticker=ticker,
        period=period
    )
    
    # Save
    library.save(library_path)
    
    # Show what was saved
    print("\n" + "=" * 60)
    print("SAVED ALPHAS")
    print("=" * 60)
    df = library.list_alphas()
    print(df[['name', 'expression', 'sharpe_ratio']].to_string())
    
    return library


# =============================================================================
# Integration with train_PPO.py
# =============================================================================

if __name__ == "__main__":
    # This would normally be called after training
    # For demo, we create a mock history
    
    # Example: Load history from a pickle file saved during training
    import pickle
    
    history_path = "training_history.pkl"
    
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        
        library = save_alphas_from_history(
            history=history,
            top_n=20,
            min_sharpe=0.1,
            ticker="AAPL",
            period="2016-02-09 to 2022-02-09"
        )
    else:
        print("No training history found.")
        print("Run train_PPO.py first, then use save_alphas_from_history()")