
# =============================================================================
# TESTING
# =============================================================================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Optional
import os
import sys

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
    
from agent.agentPPO import PPOAgent



if __name__ == "__main__":
    from environment.env import AlphaEnv
    import pandas as pd
    
    # Create sample data
    np.random.seed(42)
    n_days = 100
    price = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    
    data = pd.DataFrame({
        'Open': price + np.random.randn(n_days) * 0.2,
        'High': price + np.abs(np.random.randn(n_days)) * 0.5,
        'Low': price - np.abs(np.random.randn(n_days)) * 0.5,
        'Close': price,
        'Volume': 1000 + np.random.randint(0, 500, n_days)
    })
    
    # Create environment and PPO agent
    env = AlphaEnv(data, max_steps=10)
    agent = PPOAgent(
        state_size=env.state_size,
        action_size=env.action_space_size,
        hidden_size=64,
        learning_rate=0.0003,
        eps_clip=0.2,
        k_epochs=4
    )
    
    print("=" * 60)
    print("PPO AGENT - Test")
    print("=" * 60)
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_space_size}")
    
    # ----- Run a few episodes -----
    n_episodes = 5
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask)
            next_state, reward, done, info = env.step(action)
            agent.store_reward(reward)
            state = next_state
            total_reward += reward
            steps += 1
        
        loss = agent.update()
        
        expr = info.get('expression', 'None')
        print(f"Episode {episode+1}: Steps={steps}, Reward={total_reward:.4f}, "
              f"Loss={loss:.4f}, Expr={expr}")
    
    print("\n" + "=" * 60)
    print("PPO Agent is ready!")
    print("=" * 60)