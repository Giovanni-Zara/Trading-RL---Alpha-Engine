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
from agent.agentPPO_enhanced import PPOAgentEnhanced
from alpha_trading.alpha_library import AlphaLibrary
import pickle



def train(
    data: pd.DataFrame,
    n_episodes: int = 1000,
    max_steps: int = 10,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    batch_episodes: int = 4,
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
        gae_lambda: GAE lambda for advantage estimation (0=TD, 1=MC)
        entropy_coef: Entropy bonus weight (higher → more exploration)
        batch_episodes: Number of episodes to collect before each PPO update
        print_every: Print progress every N episodes
        save_path: Where to save the trained agent
    
    Returns:
        agent: Trained agent
        history: Training history (rewards, expressions)
    """
    # Create environment
    env = AlphaEnv(data, max_steps=max_steps)
    
    # Create agent
    agent = PPOAgentEnhanced(
        state_size=env.state_size,
        action_size=env.action_space_size,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        entropy_coef=entropy_coef,
        batch_episodes=batch_episodes
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
    print(f"Batch episodes: {batch_episodes}")
    print(f"GAE lambda: {gae_lambda}")
    print(f"Entropy coef: {entropy_coef}")
    print(f"Actions: {env.actions}")
    print("=" * 60)
    
    # Accumulate per-episode info between updates
    batch_rewards = []
    batch_expressions = []
    last_loss = 0.0
    
    seen_expressions = {} #to penalizy already seen stuff

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
            
            #penalty: diminish reward for already seen stuff
            if done and info.get('expression') is not None:
                expr_str = info['expression']
                seen_expressions[expr_str] = seen_expressions.get(expr_str,0) + 1
                count = seen_expressions[expr_str]
                if count > 3:
                    #after 3 repeats, decay reward
                    reward *= max(0.1, 1.0 / (1 + 0.2 * (count - 3))) #decay from 1.0 to 0.1 as count increases

            # Store reward and done flag
            agent.store_reward(reward)
            agent.store_done(done)
            
            state = next_state
            episode_reward += reward
        
        # Track this episode's results
        expr = info.get('expression', 'None')
        batch_rewards.append(episode_reward)
        batch_expressions.append(expr)
        recent_rewards.append(episode_reward)

        #decay entropy coeff: I want high exploration at the beginning, but later on I want to focus on exploitation of good expressions
        agent.entropy_coef = max(0.01, 0.1 * (1 - episode / n_episodes)) #decay from 0.1 to 0.01 over time
        
        # Track best
        if episode_reward > best_reward and expr is not None:
            best_reward = episode_reward
            best_expression = expr
        
        # Update agent only when enough episodes have been collected
        if agent.should_update():
            last_loss = agent.update()
            
            # Record history for each episode in the batch
            history['rewards'].extend(batch_rewards)
            history['expressions'].extend(batch_expressions)
            history['losses'].extend([last_loss] * len(batch_rewards))
            
            batch_rewards = []
            batch_expressions = []
        
        # Print progress
        if episode % print_every == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode:5d} | "
                  f"Avg Reward: {avg_reward:7.4f} | "
                  f"Best: {best_reward:7.4f} | "
                  f"Last Expr: {expr}")
    
    # Flush any remaining episodes in the buffer
    if len(batch_rewards) > 0:
        last_loss = agent.update()
        history['rewards'].extend(batch_rewards)
        history['expressions'].extend(batch_expressions)
        history['losses'].extend([last_loss] * len(batch_rewards))
    
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
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ohlcv_data.csv")
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
        print("DATA NOT FOUND. CHECK PATH")
        Exception("Data file not found. Please provide a valid path to the OHLCV data.")    

    # ----- Train -----
    agent, history = train(
        data=data,
        n_episodes=2000,
        max_steps=8,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.1,
        batch_episodes=4,
        print_every=50,
        save_path="agent_checkpointPPO_enh.pt"
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

    # SAVE ALPHAS TO LIB
    # creat/load library
    library_path = "alphas/alpha_library.json"
    if os.path.exists(library_path):
        library = AlphaLibrary.load(library_path)
    else:
        library = AlphaLibrary(name="ppo_discovered")
    
    # Add best alphas from this training run
    ticker_name = ticker if 'ticker' in dir() else "sample"
    library.add_from_training_history(
        history=history,
        top_n=20,
        min_sharpe=0.1,
        ticker=ticker_name,
        period="training"
    )
    
    # Save library
    library.save(library_path)
    
    # Also save raw history for later analysis

    with open("training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved to training_history.pkl")



    """
    note: the saved library should look something like this 
    
    
    {
        "name": "ppo_discovered",
        "created_at": "2026-02-09T14:30:00",
        "alphas": {
            "alpha_0001": {
            "name": "alpha_0001",
            "expression": "ts_mean(sub(close, open), 5)",
            "sharpe_ratio": 1.23,
            "mean_return": 0.0012,
            "std_return": 0.015,
            "created_at": "2026-02-09T14:30:00",
            "training_ticker": "AAPL",
            "complexity": 4
            },
            "alpha_0002": {
            "expression": "div(delta(close, 1), ts_std(close, 10))",
            "sharpe_ratio": 0.98,
            ...
            }
        }
    }
    
    """