"""
Training Script for Alpha Expression Synthesis

Trains the RL agent to discover alpha expressions.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import deque

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from environment.env import AlphaEnv
from agent.agentPPO import PPOAgent



def train(
    data: pd.DataFrame,
    n_episodes: int = 1000,
    max_steps: int = 10,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    print_every: int = 100,
    save_path: str = None
):
    """
    Train the agent to discover alpha expressions.
    
    Parameters:
        data: OHLCV DataFrame
        n_episodes: Number of training episodes
        max_steps: Max steps per episode
        hidden_size: Size of hidden layer
        learning_rate: Learning rate
        gamma: Discount factor
        print_every: Print progress every N episodes
        save_path: Where to save the trained agent
    
    Returns:
        agent: Trained agent
        history: Training history (rewards, expressions)
    """
    # Create environment
    env = AlphaEnv(data, max_steps=max_steps)
    
    # Create agent
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=env.action_space_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma
    )
    
    # Training history
    history = {
        'rewards': [],
        'expressions': [],
        'losses': []
    }
    
    # For tracking recent performance
    recent_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    best_expression = None
    
    print("=" * 60)
    print("TRAINING ALPHA SYNTHESIS AGENT")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Actions: {env.actions}")
    print("=" * 60)
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get valid actions
            action_mask = env.get_action_mask()
            
            # Select action
            action = agent.select_action(state, action_mask)
            
            # Step
            next_state, reward, done, info = env.step(action)
            
            # Store reward
            agent.store_reward(reward)
            
            state = next_state
            episode_reward += reward
        
        # Update agent
        loss = agent.update()
        
        # Store history
        expr = info.get('expression', 'None')
        history['rewards'].append(episode_reward)
        history['expressions'].append(expr)
        history['losses'].append(loss)
        recent_rewards.append(episode_reward)
        
        # Track best
        if episode_reward > best_reward and expr is not None:
            best_reward = episode_reward
            best_expression = expr
        
        # Print progress
        if episode % print_every == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode:5d} | "
                  f"Avg Reward: {avg_reward:7.4f} | "
                  f"Best: {best_reward:7.4f} | "
                  f"Last Expr: {expr}")
    
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best Reward (Sharpe): {best_reward:.4f}")
    print(f"Best Expression: {best_expression}")
    print("=" * 60)
    
    # Save agent
    if save_path:
        agent.save(save_path)
    
    return agent, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # ----- Load or create data -----
    # Try to load real data, otherwise use sample data
    data_path = "./alpha-engine-private/PIPELINE_CLEAN/data/ohlcv_data.csv"
    
    if os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        data = pd.read_csv(data_path)
        # Use just one ticker for now
        if 'Ticker' in data.columns:
            ticker = data['Ticker'].unique()[0]
            data = data[data['Ticker'] == ticker].copy()
            data = data.sort_values('Date').reset_index(drop=True)
            print(f"Using ticker: {ticker}, {len(data)} rows")
    else:
        print("Using sample data...")
        np.random.seed(42)
        n_days = 500
        price = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
        
        data = pd.DataFrame({
            'Open': price + np.random.randn(n_days) * 0.2,
            'High': price + np.abs(np.random.randn(n_days)) * 0.5,
            'Low': price - np.abs(np.random.randn(n_days)) * 0.5,
            'Close': price,
            'Volume': 1000 + np.random.randint(0, 500, n_days)
        })
    
    # ----- Train -----
    agent, history = train(
        data=data,
        n_episodes=1500,
        max_steps=8,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        print_every=50,
        save_path="agent_checkpointPPO.pt"
    )
    
    # ----- Show some discovered expressions -----
    print("\n" + "=" * 60)
    print("TOP DISCOVERED EXPRESSIONS")
    print("=" * 60)
    
    # Get unique expressions with their rewards
    expr_rewards = {}
    for expr, reward in zip(history['expressions'], history['rewards']):
        if expr and expr != 'None':
            if expr not in expr_rewards or reward > expr_rewards[expr]:
                expr_rewards[expr] = reward
    
    # Sort by reward
    sorted_exprs = sorted(expr_rewards.items(), key=lambda x: x[1], reverse=True)
    
    for i, (expr, reward) in enumerate(sorted_exprs[:10]):
        print(f"{i+1:2d}. Sharpe={reward:7.4f} | {expr}")