"""
Training Script for All Tickers

Orchestrates training of all 20 specialized agents.
Can run sequentially or in parallel.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTI_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)
_PIPELINE_DIR = os.path.dirname(_MULTI_AGENT_DIR)

if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from multi_agent.config import (
    TRAINING_TICKERS, 
    TICKER_CATEGORIES,
    HYPERPARAMETERS,
    MODELS_DIR,
    ALPHAS_DIR,
    DATA_DIR,
    get_model_path,
    get_alpha_library_path,
    get_category,
)
from multi_agent.train import train_single_ticker


def train_all_tickers(
    data: pd.DataFrame,
    tickers: List[str] = None,
    n_episodes: int = None,
    parallel: bool = False,
    n_workers: int = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Train agents for all tickers.
    
    Parameters:
        data: Full DataFrame with all tickers
        tickers: List of tickers to train (default: all from config)
        n_episodes: Episodes per ticker (default from config)
        parallel: Whether to train in parallel (default: False)
        n_workers: Number of parallel workers (default: CPU count - 1)
        verbose: Whether to print progress
    
    Returns:
        results: Dict mapping ticker -> training history
    """
    tickers = tickers or TRAINING_TICKERS
    n_episodes = n_episodes or HYPERPARAMETERS['n_episodes']
    
    # Filter to available tickers
    available = set(data['Ticker'].unique())
    tickers = [t for t in tickers if t in available]
    
    if len(tickers) == 0:
        print("No valid tickers found in data!")
        return {}
    
    print("\n" + "=" * 80)
    print("MULTI-AGENT TRAINING: ONE AGENT PER TICKER")
    print("=" * 80)
    print(f"Tickers to train: {len(tickers)}")
    print(f"Episodes per ticker: {n_episodes}")
    print(f"Parallel: {parallel}")
    print(f"Categories:")
    for cat, cat_tickers in TICKER_CATEGORIES.items():
        cat_available = [t for t in cat_tickers if t in tickers]
        if cat_available:
            print(f"  {cat}: {cat_available}")
    print("=" * 80)
    
    results = {}
    
    if parallel:
        results = _train_parallel(data, tickers, n_episodes, n_workers, verbose)
    else:
        results = _train_sequential(data, tickers, n_episodes, verbose)
    
    # ----- Summary -----
    _print_summary(results)
    
    # ----- Save combined results -----
    _save_combined_results(results)
    
    return results


def _train_sequential(
    data: pd.DataFrame,
    tickers: List[str],
    n_episodes: int,
    verbose: bool,
) -> Dict[str, Dict[str, Any]]:
    """Train tickers one by one."""
    results = {}
    
    for i, ticker in enumerate(tickers):
        print(f"\n{'='*80}")
        print(f"TRAINING TICKER {i+1}/{len(tickers)}: {ticker}")
        print(f"{'='*80}")
        
        agent, history = train_single_ticker(
            data=data,
            ticker=ticker,
            n_episodes=n_episodes,
            verbose=verbose,
        )
        
        if history is not None:
            results[ticker] = history
    
    return results


def _train_parallel(
    data: pd.DataFrame,
    tickers: List[str],
    n_episodes: int,
    n_workers: int,
    verbose: bool,
) -> Dict[str, Dict[str, Any]]:
    """Train tickers in parallel using multiprocessing."""
    n_workers = n_workers or max(1, mp.cpu_count() - 1)
    print(f"Using {n_workers} parallel workers")
    
    results = {}
    
    # Note: For true parallelism, we'd need to save data to disk
    # and load in each worker. For simplicity, we do sequential here
    # but with reduced verbosity.
    
    # TODO: Implement proper parallel training with ProcessPoolExecutor
    # For now, fall back to sequential
    print("Note: Parallel training not fully implemented, using sequential")
    return _train_sequential(data, tickers, n_episodes, verbose=False)


def _print_summary(results: Dict[str, Dict[str, Any]]):
    """Print training summary."""
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    # Sort by best Sharpe
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('best_sharpe', 0),
        reverse=True
    )
    
    print(f"\n{'Ticker':<8} {'Category':<12} {'Best Sharpe':>12} {'Best Reward':>12} {'Expression':<40}")
    print("-" * 90)
    
    for ticker, history in sorted_results:
        cat = get_category(ticker)
        sharpe = history.get('best_sharpe', 0)
        reward = history.get('best_reward', 0)
        expr = history.get('best_expression', '')[:40]
        print(f"{ticker:<8} {cat:<12} {sharpe:>12.4f} {reward:>12.4f} {expr:<40}")
    
    # Category summary
    print("\n" + "-" * 80)
    print("BY CATEGORY:")
    print("-" * 80)
    
    for cat, cat_tickers in TICKER_CATEGORIES.items():
        cat_results = [results[t] for t in cat_tickers if t in results]
        if cat_results:
            avg_sharpe = np.mean([r.get('best_sharpe', 0) for r in cat_results])
            print(f"  {cat:<12}: Avg Sharpe = {avg_sharpe:.4f} ({len(cat_results)} tickers)")
    
    print("=" * 80)


def _save_combined_results(results: Dict[str, Dict[str, Any]]):
    """Save combined results to JSON."""
    output_path = os.path.join(_MULTI_AGENT_DIR, "training_results.json")
    
    # Prepare serializable results
    serializable = {}
    for ticker, history in results.items():
        serializable[ticker] = {
            'best_reward': float(history.get('best_reward', 0)),
            'best_sharpe': float(history.get('best_sharpe', 0)),
            'best_expression': history.get('best_expression', ''),
            'category': get_category(ticker),
            'n_episodes': len(history.get('rewards', [])),
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train multi-agent system")
    parser.add_argument("--tickers", nargs="+", default=None, 
                        help="Specific tickers to train (default: all)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Episodes per ticker")
    parser.add_argument("--parallel", action="store_true",
                        help="Train in parallel")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    # Load data
    data_path = os.path.join(DATA_DIR, "ohlcv_data.csv")
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data):,} rows")
    print(f"Tickers: {sorted(data['Ticker'].unique().tolist())}")
    
    # Train
    results = train_all_tickers(
        data=data,
        tickers=args.tickers,
        n_episodes=args.episodes,
        parallel=args.parallel,
        n_workers=args.workers,
    )