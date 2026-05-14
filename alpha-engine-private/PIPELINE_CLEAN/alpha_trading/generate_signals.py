"""
Generate Trading Signals from Discovered Alphas

This script:
1. Loads the trained alpha library
2. Evaluates alphas on new/current data
3. Generates trading signals (long/short/neutral)
4. Outputs signals for use in a trading system
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from alpha_library import AlphaLibrary, compute_alpha_metrics


def generate_signals(
    library: AlphaLibrary,
    data: pd.DataFrame,
    alpha_names: list = None,
    signal_threshold: float = 0.5,
    combine_method: str = 'equal_weight'
) -> pd.DataFrame:
    """
    Generate trading signals from alphas.
    
    Parameters:
        library: AlphaLibrary with discovered alphas
        data: Current OHLCV data
        alpha_names: Which alphas to use (None = all)
        signal_threshold: Threshold for long/short (z-score)
        combine_method: How to combine alphas ('equal_weight', 'sharpe_weight')
        
    Returns:
        DataFrame with columns:
        - Date
        - Each alpha's raw signal
        - combined_signal: Weighted combination
        - position: -1 (short), 0 (neutral), +1 (long)
    """
    if alpha_names is None:
        alpha_names = list(library.alphas.keys())
    
    # Evaluate all selected alphas
    signals = pd.DataFrame(index=data.index)
    
    for name in alpha_names:
        try:
            signals[name] = library.evaluate_alpha(name, data, normalize=True)
        except Exception as e:
            print(f"Skipping {name}: {e}")
    
    if signals.empty:
        raise ValueError("No alphas could be evaluated")
    
    # Combine alphas
    if combine_method == 'equal_weight':
        weights = {name: 1.0 / len(signals.columns) for name in signals.columns}
    
    elif combine_method == 'sharpe_weight':
        # Weight by Sharpe ratio (higher Sharpe = more weight)
        sharpes = {}
        for name in signals.columns:
            record = library.get_alpha(name)
            sharpes[name] = max(record.sharpe_ratio, 0.01)  # Avoid zero weights
        
        total = sum(sharpes.values())
        weights = {name: s / total for name, s in sharpes.items()}
    
    else:
        raise ValueError(f"Unknown combine method: {combine_method}")
    
    # Compute weighted combination
    signals['combined_signal'] = sum(
        signals[name] * weights[name] for name in weights.keys()
    )
    
    # Convert to discrete positions
    signals['position'] = 0
    signals.loc[signals['combined_signal'] > signal_threshold, 'position'] = 1
    signals.loc[signals['combined_signal'] < -signal_threshold, 'position'] = -1
    
    # Add date column if index is datetime
    signals = signals.reset_index()
    if 'Date' not in signals.columns:
        signals = signals.rename(columns={'index': 'Date'})
    
    return signals


def backtest_signals(
    signals: pd.DataFrame,
    data: pd.DataFrame,
    transaction_cost: float = 0.001  # 0.1% per trade
) -> dict:
    """
    Simple backtest of generated signals.
    
    Parameters:
        signals: DataFrame with 'position' column
        data: OHLCV data
        transaction_cost: Cost per trade (as fraction)
        
    Returns:
        Dictionary with backtest results
    """
    # Align data
    signals = signals.set_index('Date')
    data = data.set_index('Date') if 'Date' in data.columns else data
    
    # Forward returns (what we'd earn)
    returns = (data['Close'].shift(-1) - data['Open'].shift(-1)) / data['Open'].shift(-1)
    
    # Strategy returns
    position = signals['position']
    strategy_returns = position * returns
    
    # Transaction costs (when position changes)
    position_changes = position.diff().abs()
    costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - costs
    
    # Remove NaN
    strategy_returns = strategy_returns.dropna()
    
    # Compute metrics
    cum_returns = (1 + strategy_returns).cumprod()
    
    total_return = cum_returns.iloc[-1] - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    
    # Drawdown
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Win rate
    winning_days = (strategy_returns > 0).sum()
    total_days = len(strategy_returns)
    win_rate = winning_days / total_days
    
    # Number of trades
    n_trades = (position.diff() != 0).sum()
    
    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'n_trades': int(n_trades),
        'n_days': int(total_days),
        'cumulative_returns': cum_returns
    }


# =============================================================================
# MAIN: Example usage
# =============================================================================

if __name__ == "__main__":
    # ----- Load alpha library -----
    library_path = "alphas/alpha_library.json"
    
    if os.path.exists(library_path):
        library = AlphaLibrary.load(library_path)
    else:
        print("No alpha library found. Run save_alphas.py first.")
        sys.exit(1)
    
    # ----- Load data -----
    data_path = "data/ohlcv_data.csv"
    
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        # Use one ticker for now
        if 'Ticker' in data.columns:
            ticker = data['Ticker'].unique()[0]
            data = data[data['Ticker'] == ticker].copy()
            data = data.sort_values('Date').reset_index(drop=True)
    else:
        print("No data found.")
        sys.exit(1)
    
    # ----- Generate signals -----
    print("=" * 60)
    print("GENERATING TRADING SIGNALS")
    print("=" * 60)
    
    # List available alphas
    print("\nAvailable alphas:")
    print(library.list_alphas()[['name', 'expression', 'sharpe_ratio']])
    
    # Generate signals using top alphas
    top_alphas = library.list_alphas().head(5)['name'].tolist()
    
    signals = generate_signals(
        library=library,
        data=data,
        alpha_names=top_alphas,
        signal_threshold=0.5,
        combine_method='sharpe_weight'
    )
    
    print(f"\nGenerated signals for {len(signals)} days")
    print(signals.tail(10))
    
    # ----- Backtest -----
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    
    results = backtest_signals(signals, data)
    
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Number of Trades: {results['n_trades']}")
    
    # ----- Save signals -----
    output_path = "signals/trading_signals.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    signals.to_csv(output_path, index=False)
    print(f"\nSignals saved to {output_path}")