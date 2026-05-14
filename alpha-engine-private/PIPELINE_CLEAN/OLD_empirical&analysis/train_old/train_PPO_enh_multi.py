
import os
import sys
import numpy as np
import pandas as pd
from collections import deque
from typing import List
import pickle

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from environment import MultiTickerAlphaEnv, DEFAULT_TRAINING_TICKERS
from agent import PPOAgentEnhanced
from alpha_trading import AlphaLibrary

def train_batch(
    data: pd.DataFrame,
    tickers: List[str] = None,
    n_episodes: int = 2000,
    max_steps: int = 8,
    hidden_size: int = 64,
    learning_rate: float = 0.001,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.1,
    batch_episodes: int = 4,
    print_every: int = 50,
    save_path: str = None
):
    """
    Train the agent on multiple tickers simultaneously.
    
    Parameters:
        data: Full OHLCV DataFrame with all tickers
        tickers: List of tickers to train on (uses default if None)
        n_episodes: Number of training episodes
        ... (other params same as train_PPO_enh.py)
    
    Returns:
        agent: Trained agent
        history: Training history
    """
    if tickers is None:
        tickers = DEFAULT_TRAINING_TICKERS
    
    # Filter to tickers that exist in data
    available_tickers = data['Ticker'].unique().tolist()
    tickers = [t for t in tickers if t in available_tickers]
    
    if len(tickers) == 0:
        raise ValueError("None of the specified tickers found in data!")
    
    print(f"Training on {len(tickers)} tickers: {tickers}")
    
    # Create multi-ticker environment
    env = MultiTickerAlphaEnv(
        data=data,
        tickers=tickers,
        max_steps=max_steps,
        min_sharpe_tickers=len(tickers) // 3  # At least 1/3 must have positive Sharpe
    )
    
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
        'losses': [],
        'ticker_sharpes': []  # Store per-ticker breakdown
    }
    
    recent_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    best_expression = None
    best_ticker_sharpes = {}
    
    print("=" * 70)
    print("BATCH TRAINING: ALPHA SYNTHESIS AGENT")
    print("=" * 70)
    print(f"Tickers: {len(env.active_tickers)}")
    print(f"Episodes: {n_episodes}")
    print(f"Max steps: {max_steps}")
    print(f"Reward = Average Sharpe across all tickers")
    print("=" * 70)
    
    batch_rewards = []
    batch_expressions = []
    batch_ticker_sharpes = []
    seen_expressions = {}
    
    for episode in range(1, n_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask)
            next_state, reward, done, info = env.step(action)
            
            # Penalize repeated expressions
            if done and info.get('expression') is not None:
                expr_str = info['expression']
                seen_expressions[expr_str] = seen_expressions.get(expr_str, 0) + 1
                count = seen_expressions[expr_str]
                if count > 3:
                    reward *= max(0.1, 1.0 / (1 + 0.2 * (count - 3)))
            
            agent.store_reward(reward)
            agent.store_done(done)
            
            state = next_state
            episode_reward += reward
        
        expr = info.get('expression', 'None')
        ticker_sharpes = info.get('ticker_sharpes', {})
        
        batch_rewards.append(episode_reward)
        batch_expressions.append(expr)
        batch_ticker_sharpes.append(ticker_sharpes)
        recent_rewards.append(episode_reward)
        
        # Entropy decay
        agent.entropy_coef = max(0.01, 0.1 * (1 - episode / n_episodes))
        
        # Track best
        if episode_reward > best_reward and expr is not None:
            best_reward = episode_reward
            best_expression = expr
            best_ticker_sharpes = ticker_sharpes
        
        # Update
        if agent.should_update():
            last_loss = agent.update()
            
            history['rewards'].extend(batch_rewards)
            history['expressions'].extend(batch_expressions)
            history['losses'].extend([last_loss] * len(batch_rewards))
            history['ticker_sharpes'].extend(batch_ticker_sharpes)
            
            batch_rewards = []
            batch_expressions = []
            batch_ticker_sharpes = []
        
        # Print progress
        if episode % print_every == 0:
            avg_reward = np.mean(recent_rewards)
            
            # Show per-ticker breakdown for recent best
            if ticker_sharpes:
                positive_tickers = sum(1 for s in ticker_sharpes.values() if s > 0)
                print(f"Episode {episode:5d} | "
                      f"Avg Sharpe: {avg_reward:7.4f} | "
                      f"Best: {best_reward:7.4f} | "
                      f"+ve tickers: {positive_tickers}/{len(ticker_sharpes)}")
            else:
                print(f"Episode {episode:5d} | "
                      f"Avg Sharpe: {avg_reward:7.4f} | "
                      f"Best: {best_reward:7.4f}")
    
    # Flush remaining
    if len(batch_rewards) > 0:
        last_loss = agent.update()
        history['rewards'].extend(batch_rewards)
        history['expressions'].extend(batch_expressions)
        history['losses'].extend([last_loss] * len(batch_rewards))
        history['ticker_sharpes'].extend(batch_ticker_sharpes)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Average Sharpe: {best_reward:.4f}")
    print(f"Best Expression: {best_expression}")
    print("\nPer-ticker breakdown:")
    for ticker, sharpe in sorted(best_ticker_sharpes.items(), key=lambda x: x[1], reverse=True):
        status = "✓" if sharpe > 0 else "✗"
        print(f"  {status} {ticker}: {sharpe:.4f}")
    print("=" * 70)
    
    if save_path:
        agent.save(save_path)
    
    return agent, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Load data
    data_path = os.path.join(_SCRIPT_DIR, "..", "data", "ohlcv_data.csv")
    
    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    print(f"Total rows: {len(data)}")
    print(f"Available tickers: {data['Ticker'].nunique()}")
    
    # ----- Option 1: Use default diversified tickers -----
    # agent, history = train_batch(
    #     data=data,
    #     tickers=None,  # Uses DEFAULT_TRAINING_TICKERS
    #     n_episodes=3000,
    #     ...
    # )
    
    # ----- Option 2: Specify custom tickers -----
    # Diversified selection (tech, finance, healthcare, consumer, industrial)
    training_tickers = [
        # Large-cap Tech
        'AAPL', 'MSFT', 'GOOGL', 'INTC',
        # Finance
        'JPM', 'BAC', 'GS',
        # Healthcare
        'JNJ', 'PFE', 'UNH',
        # Consumer
        'WMT', 'KO', 'PG',
        # Industrial
        'GE', 'CAT',
        # Energy
        'XOM', 'CVX',
    ]
    
    # Check which tickers are available
    available = data['Ticker'].unique()
    training_tickers = [t for t in training_tickers if t in available]
    print(f"\nUsing {len(training_tickers)} tickers: {training_tickers}")
    
    # Train
    agent, history = train_batch(
        data=data,
        tickers=training_tickers,
        n_episodes=3000,  # More episodes since we need to learn patterns across many tickers
        max_steps=8,
        hidden_size=64,
        learning_rate=0.001,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.1,
        batch_episodes=4,
        print_every=50,
        save_path="agent_batch_trained.pt"
    )
    
    # ----- Show top expressions -----
    print("\n" + "=" * 70)
    print("TOP DISCOVERED EXPRESSIONS (Universal Alphas)")
    print("=" * 70)
    
    expr_rewards = {}
    for expr, reward in zip(history['expressions'], history['rewards']):
        if expr and expr != 'None':
            if expr not in expr_rewards or reward > expr_rewards[expr]:
                expr_rewards[expr] = reward
    
    sorted_exprs = sorted(expr_rewards.items(), key=lambda x: x[1], reverse=True)
    
    for i, (expr, reward) in enumerate(sorted_exprs[:15]):
        print(f"{i+1:2d}. Avg Sharpe={reward:7.4f} | {expr}")
    
    # ----- Save to library -----
    library_path = "alphas/batch_alpha_library.json"
    os.makedirs("alphas", exist_ok=True)
    
    if os.path.exists(library_path):
        library = AlphaLibrary.load(library_path)
    else:
        library = AlphaLibrary(name="batch_discovered")
    
    library.add_from_training_history(
        history=history,
        top_n=20,
        min_sharpe=0.0,  # Lower threshold since average across tickers is harder
        ticker=",".join(training_tickers),
        period="batch_training"
    )
    
    library.save(library_path)
    
    # Save history
    with open("batch_training_history.pkl", 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved to batch_training_history.pkl")
    
    # ----- Detailed analysis of best alpha -----
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS OF BEST ALPHA")
    print("=" * 70)
    
    if sorted_exprs:
        best_expr = sorted_exprs[0][0]
        print(f"Expression: {best_expr}")
        
        # Find the ticker breakdown for this expression
        for expr, ticker_sharpes in zip(history['expressions'], history['ticker_sharpes']):
            if expr == best_expr and ticker_sharpes:
                print("\nPer-ticker Sharpe ratios:")
                positive = 0
                for ticker in sorted(ticker_sharpes.keys()):
                    sharpe = ticker_sharpes[ticker]
                    status = "✓" if sharpe > 0 else "✗"
                    if sharpe > 0:
                        positive += 1
                    print(f"  {status} {ticker:6s}: {sharpe:7.4f}")
                print(f"\nPositive Sharpe: {positive}/{len(ticker_sharpes)} tickers ({positive/len(ticker_sharpes)*100:.1f}%)")
                break