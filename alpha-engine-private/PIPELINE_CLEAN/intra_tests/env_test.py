
"""Environment test"""

import os
import sys
import numpy as np
import pandas as pd

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from environment.env import AlphaEnv


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
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
    
    print("=" * 60)
    print("ALPHA ENVIRONMENT - Test")
    print("=" * 60)
    
    env = AlphaEnv(data, max_steps=10)
    
    print(f"\nAction space size: {env.action_space_size}")
    print(f"State size: {env.state_size}")
    print(f"Actions: {env.actions}")
    
    # ----- Test Episode 1: Manual build of sub(close, open) + STOP -----
    print("\n" + "-" * 40)
    print("Episode 1: Build sub(close, open)")
    print("-" * 40)
    
    state = env.reset()
    env.render()
    
    # close
    state, reward, done, info = env.step(env.action_name_to_id['close'])
    print("Action: close")
    env.render()
    
    # open
    state, reward, done, info = env.step(env.action_name_to_id['open'])
    print("Action: open")
    env.render()
    
    # sub
    state, reward, done, info = env.step(env.action_name_to_id['sub'])
    print("Action: sub")
    env.render()
    
    # STOP (should now be valid since size=3 >= min_expression_size=3)
    #state, reward, done, info = env.step(env.action_name_to_id['STOP'])
    print(f"Action: STOP")
    print(f"Done: {done}, Reward: {reward:.4f}")
    print(f"Expression: {info.get('expression')}")
    print(f"Size: {info.get('size')}")
    
    # ----- Test Episode 2: Random valid actions -----
    print("\n" + "-" * 40)
    print("Episode 2: Random valid actions")
    print("-" * 40)
    
    state = env.reset()
    done = False
    step = 0
    
    while not done:
        valid_actions = env._get_valid_actions()
        action = np.random.choice(valid_actions)
        state, reward, done, info = env.step(action)
        print(f"Step {step}: {env.actions[action]}")
        step += 1
    
    print(f"\nDone={done}, Reward={reward:.4f}")
    print(f"Expression: {info.get('expression')}")
    
    print("\n" + "=" * 60)
    print("Environment is ready!")
    print("=" * 60)