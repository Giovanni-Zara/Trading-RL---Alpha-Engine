"""
Training Script for Single-Ticker Agent

Trains one agent specialized for one ticker.
"""

import os
import sys
import pandas as pd
import numpy as np
from collections import deque
from typing import Dict, Any, Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTI_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)
_PIPELINE_DIR = os.path.dirname(_MULTI_AGENT_DIR)

if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from agent import PPOAgentEnhanced
from multi_agent.environment import SingleTickerAlphaEnv
from multi_agent.config import HYPERPARAMETERS, get_model_path, get_alpha_library_path


def train_single_ticker(
    data: pd.DataFrame,
    ticker: str,
    n_episodes: int = None,
    max_steps: int = None,
    hidden_size: int = None,
    learning_rate: float = None,
    gamma: float = None,
    gae_lambda: float = None,
    entropy_coef: float = None,
    batch_episodes: int = None,
    print_every: int = None,
    save_path: str = None,
    verbose: bool = True,
) -> Tuple[PPOAgentEnhanced, Dict[str, Any]]:
    """
    Train a single agent for a specific ticker.
    
    Parameters:
        data: Full DataFrame with all tickers
        ticker: The ticker to train on
        n_episodes: Number of training episodes (default from config)
        ... other hyperparameters (default from config)
        save_path: Path to save model (default from config)
        verbose: Whether to print progress
    
    Returns:
        agent: Trained PPOAgentEnhanced
        history: Training history dict
    """
    # ----- Load defaults from config -----
    hp = HYPERPARAMETERS
    n_episodes = n_episodes or hp['n_episodes']
    max_steps = max_steps or hp['max_steps']
    hidden_size = hidden_size or hp['hidden_size']
    learning_rate = learning_rate or hp['learning_rate']
    gamma = gamma or hp['gamma']
    gae_lambda = gae_lambda or hp['gae_lambda']
    entropy_coef = entropy_coef or hp['entropy_coef']
    batch_episodes = batch_episodes or hp['batch_episodes']
    print_every = print_every or hp['print_every']
    save_path = save_path or get_model_path(ticker)
    
    # ----- Create environment -----
    try:
        env = SingleTickerAlphaEnv(
            data=data,
            ticker=ticker,
            max_steps=max_steps,
        )
    except ValueError as e:
        print(f"[{ticker}] Cannot create environment: {e}")
        return None, None
    
    # ----- Create agent -----
    agent = PPOAgentEnhanced(
        state_size=env.state_size,
        action_size=env.action_space_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        batch_episodes=batch_episodes,
    )
    
    # ----- Training history -----
    history = {
        'ticker': ticker,
        'rewards': [],
        'sharpes': [],
        'expressions': [],
        'losses': [],
    }
    
    recent_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    best_expression = None
    best_sharpe = float('-inf')
    
    # ----- Print header -----
    if verbose:
        print("\n" + "=" * 70)
        print(f"TRAINING AGENT FOR: {ticker}")
        print("=" * 70)
        print(f"Episodes:       {n_episodes}")
        print(f"Max steps:      {max_steps}")
        print(f"State size:     {env.state_size}")
        print(f"Action size:    {env.action_space_size}")
        print(f"Valid dates:    {len(env.valid_start_dates)}")
        print(f"Hidden size:    {hidden_size}")
        print(f"Batch episodes: {batch_episodes}")
        print(f"Save path:      {save_path}")
        print("=" * 70)
    
    # ----- Training loop -----
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get action mask and select action
            mask = env.get_action_mask()
            action = agent.select_action(state, mask)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience - done must be stored EVERY step
            agent.store_reward(reward)
            agent.store_done(done)
            
            episode_reward += reward
            state = next_state

        # Record history
        history['rewards'].append(episode_reward)
        sharpe_val = info.get('sharpe', None)
        sharpe_val = sharpe_val if sharpe_val is not None else 0.0
        history['sharpes'].append(sharpe_val)
        history['expressions'].append(info.get('expression', '') or '')
        recent_rewards.append(episode_reward)
        
        # Update best
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_expression = info.get('expression', '')
            best_sharpe = sharpe_val
        
        # Update agent
        if agent.should_update():
            loss = agent.update()
            history['losses'].append(loss)
        
        # Print progress
        if verbose and episode % print_every == 0:
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            recent_sharpes = [s for s in history['sharpes'][-100:] if s is not None]
            avg_sharpe = np.mean(recent_sharpes) if recent_sharpes else 0.0
            print(
                f"[{ticker}] Ep {episode:>6}/{n_episodes} | "
                f"Avg Reward: {avg_reward:>7.3f} | "
                f"Avg Sharpe: {avg_sharpe:>6.3f} | "
                f"Best: {best_reward:.3f}"
            )
    
    # ----- Save model -----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        if verbose:
            print(f"[{ticker}] Model saved to {save_path}")
    
    # ----- Final summary -----
    if verbose:
        print("\n" + "-" * 70)
        print(f"[{ticker}] TRAINING COMPLETE")
        print(f"  Best Reward:     {best_reward:.4f}")
        print(f"  Best Sharpe:     {best_sharpe:.4f}")
        print(f"  Best Expression: {best_expression}")
        print("-" * 70)
    
    # Store best in history
    history['best_reward'] = best_reward
    history['best_sharpe'] = best_sharpe
    history['best_expression'] = best_expression
    
    return agent, history


# =============================================================================
# MAIN (for testing single ticker)
# =============================================================================

if __name__ == "__main__":
    from multi_agent.config import PIPELINE_DIR, DATA_DIR
    
    # Load data
    data_path = os.path.join(DATA_DIR, "ohlcv_data.csv")
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data):,} rows")
    
    # Train single ticker as test
    ticker = "AAPL"
    agent, history = train_single_ticker(
        data=data,
        ticker=ticker,
        n_episodes=1000,  # Short run for testing
        print_every=100,
    )