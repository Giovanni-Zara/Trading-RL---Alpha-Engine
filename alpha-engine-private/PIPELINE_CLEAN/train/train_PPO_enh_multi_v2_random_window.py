"""
Training Script for Multi-Ticker Market-Aware Alpha Synthesis (V2)

Uses MultiTickerAlphaEnvV2 which includes market regime features in the state.
The agent learns to adapt expression building to market conditions.

Key differences from v1:
- State includes 8 market regime features (trend, vol, mean-rev, volume)
- Each episode samples a random date → different market conditions
- Reward is evaluated on a forward-looking window from that date
- Agent learns: "in trending markets, build momentum alphas"
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import deque
from typing import List
import pickle

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from environment import MultiTickerAlphaEnvV2, DEFAULT_TRAINING_TICKERS
from agent import PPOAgentEnhanced
from alpha_trading import AlphaLibrary


def train_batch_v2(
    data: pd.DataFrame,
    tickers: List[str] = None,
    n_episodes: int = 50000,          # MUCH more episodes
    max_steps: int = 20,              # Allow deeper expressions
    hidden_size: int = 256,           # Bigger network
    learning_rate: float = 0.0001,    # Lower LR for stability
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.10,       # Slightly lower entropy
    batch_episodes: int = 64,         # Bigger batches
    print_every: int = 500,           # Less frequent printing
    save_path: str = None
):
    """
    Train with production-aligned evaluation.
    
    V2 Changes:
        - Uses alpha primitives as building blocks
        - Longer evaluation window (20 days)
        - Deeper networks
        - More training episodes
    """
    if tickers is None:
        tickers = DEFAULT_TRAINING_TICKERS

    available_tickers = data['Ticker'].unique().tolist()
    tickers = [t for t in tickers if t in available_tickers]

    if len(tickers) == 0:
        raise ValueError("None of the specified tickers found in data!")

    print(f"Training on {len(tickers)} tickers: {tickers}")

    env = MultiTickerAlphaEnvV2(
        data=data,
        tickers=tickers,
        max_steps=max_steps,
        min_sharpe_tickers=len(tickers) // 3
    )

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
        'ticker_sharpes': [],
        'episode_dates': [],
        'market_features': [],
    }

    recent_rewards = deque(maxlen=100)
    best_reward = float('-inf')
    best_expression = None
    best_ticker_sharpes = {}

    print("=" * 80)
    print("BATCH TRAINING V2: PRODUCTION-ALIGNED ALPHA SYNTHESIS")
    print("=" * 80)
    print(f"Tickers:          {len(env.active_tickers)}")
    print(f"Episodes:         {n_episodes}")
    print(f"Max steps:        {max_steps}")
    print(f"Min expr depth:   {env.MIN_EXPRESSION_DEPTH}")
    print(f"Eval window:      {env.EVAL_WINDOW} days (single-date, production-aligned)")
    print(f"Sharpe lookback:  {env.SHARPE_LOOKBACK} days (for rolling normalization)")
    print(f"Window options:   {env.WINDOW_OPTIONS} (min={env.MIN_TS_WINDOW})")
    print(f"State size:       {env.state_size} "
          f"(short={env.short_feature_size} + medium={env.medium_feature_size} "
          f"+ long={env.long_feature_size} + stack={env.stack_feature_size})")
    print(f"Action size:      {env.action_space_size} (incl {len(env.WINDOW_OPTIONS)} window actions)")
    print(f"Hidden size:      {hidden_size}")
    print(f"Batch episodes:   {batch_episodes}")
    print(f"Entropy coef:     {entropy_coef}")
    print(f"Timeframes:       {env.SHORT_WINDOW}/{env.MEDIUM_WINDOW}/{env.LONG_WINDOW} days")
    print(f"Valid start dates:{len(env.valid_start_dates)}")
    print(f"Noise reduction:  {len(env.tickers)} tickers × {batch_episodes} ep/batch")
    print("=" * 80)

    batch_rewards = []
    batch_expressions = []
    batch_ticker_sharpes = []
    batch_dates = []
    batch_market_features = []
    seen_expressions = {}

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        done = False
        episode_reward = 0

        # Store episode metadata
        episode_date = env.episode_date
        episode_market_features = state[:env.market_feature_size].tolist()

        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask)
            next_state, reward, done, info = env.step(action)

            # Light penalty for repeated expressions (don't distort too much)
            if done and info.get('expression') is not None:
                expr_str = info['expression']
                seen_expressions[expr_str] = seen_expressions.get(expr_str, 0) + 1
                count = seen_expressions[expr_str]
                if count > 10:  # Only penalize after 10 repeats
                    reward *= max(0.5, 1.0 / (1 + 0.05 * (count - 10)))

            agent.store_reward(reward)
            agent.store_done(done)

            state = next_state
            episode_reward += reward

        expr = info.get('expression', 'None')
        ticker_sharpes = info.get('ticker_sharpes', {})
        raw_sharpe = info.get('raw_sharpe', 0.0)

        batch_rewards.append(episode_reward)
        batch_expressions.append(expr)
        batch_ticker_sharpes.append(ticker_sharpes)
        batch_dates.append(episode_date)
        batch_market_features.append(episode_market_features)
        recent_rewards.append(episode_reward)

        # Entropy decay (slower)
        agent.entropy_coef = max(0.05, entropy_coef * (1 - episode / (n_episodes * 3.0)))

        # Track best (now reward ≈ Sharpe!)
        if episode_reward > best_reward and expr is not None and expr != 'None':
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
            history['episode_dates'].extend(batch_dates)
            history['market_features'].extend(batch_market_features)

            batch_rewards = []
            batch_expressions = []
            batch_ticker_sharpes = []
            batch_dates = []
            batch_market_features = []

        if episode % print_every == 0:
            rec_rew = np.mean(recent_rewards)  # Rolling avg of last 100 episode rewards

            if ticker_sharpes:
                # Current episode's actual Sharpe (averaged across tickers)
                valid_sharpes = [s for s in ticker_sharpes.values() if s > -5]
                avg_sharpe = np.mean(valid_sharpes) if valid_sharpes else 0.0
                positive_tickers = sum(1 for s in ticker_sharpes.values() if s > 0)
                
                print(f"Ep {episode:5d} | "
                      f"AvgSharpe: {avg_sharpe:+.2f} | "
                      f"RecRew: {rec_rew:+.3f} | "
                      f"Best: {best_reward:+.2f} | "
                      f"+ve: {positive_tickers}/{len(ticker_sharpes)} | "
                      f"Date: {episode_date}")
            else:
                print(f"Ep {episode:5d} | "
                      f"RecRew: {rec_rew:+.3f} | "
                      f"Best: {best_reward:+.2f}")


    # Flush remaining
    if len(batch_rewards) > 0:
        last_loss = agent.update()
        history['rewards'].extend(batch_rewards)
        history['expressions'].extend(batch_expressions)
        history['losses'].extend([last_loss] * len(batch_rewards))
        history['ticker_sharpes'].extend(batch_ticker_sharpes)
        history['episode_dates'].extend(batch_dates)
        history['market_features'].extend(batch_market_features)

    # === TRAINING COMPLETE ===
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Average Sharpe: {best_reward:.4f}")
    print(f"Best Expression: {best_expression}")
    print("\nPer-ticker breakdown:")
    for ticker, sharpe in sorted(best_ticker_sharpes.items(), key=lambda x: x[1], reverse=True):
        status = "✓" if sharpe > 0 else "✗"
        print(f"  {status} {ticker:6s}: {sharpe:7.4f}")
    print("=" * 80)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        agent.save(save_path)
        print(f"Agent saved to {save_path}")


    return agent, history


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Load actual dataset
    data_path = os.path.join(_PIPELINE_DIR, "data", "ohlcv_data.csv")

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}")
        print("Expected format: Date,Ticker,Open,High,Low,Close,Volume")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)

    print(f"Total rows: {len(data):,}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Available tickers: {data['Ticker'].nunique()}")
    print(f"Tickers: {sorted(data['Ticker'].unique().tolist())[:20]}...")

    # ----- Select diversified training tickers -----
    training_tickers = [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'INTC', 'CSCO',
        # Finance
        'JPM', 'BAC', 'GS',
        # Healthcare
        'JNJ', 'PFE', 'UNH',
        # Consumer
        'WMT', 'KO', 'PG',
        # Industrial / Aerospace
        'GE', 'CAT', 'BA',
        # Energy
        'XOM', 'CVX',
    ]

    # Filter to available tickers
    available = set(data['Ticker'].unique())
    training_tickers = [t for t in training_tickers if t in available]

    # If not enough from preferred list, add more from dataset
    if len(training_tickers) < 10:
        remaining = [t for t in sorted(available) if t not in training_tickers]
        needed = 10 - len(training_tickers)
        # Sample evenly spaced tickers for diversity
        step = max(1, len(remaining) // needed)
        extra = remaining[::step][:needed]
        training_tickers.extend(extra)

    print(f"\nUsing {len(training_tickers)} tickers: {training_tickers}")

    # ----- Train -----
    agent, history = train_batch_v2(
        data=data,
        tickers=training_tickers,
        n_episodes=50000,            # REDUCED from 150k — less overfitting
        max_steps=20,
        hidden_size=256,
        learning_rate=0.0001,
        gamma=0.99,
        gae_lambda=0.95,
        entropy_coef=0.15,           # INCREASED from 0.10 — more exploration
        batch_episodes=64,
        print_every=500,
        save_path=os.path.join(_PIPELINE_DIR, "models", "agent_batch_v2_primitives_fix.pt")
    )

    # ----- Show top expressions -----
    print("\n" + "=" * 80)
    print("TOP DISCOVERED EXPRESSIONS (Universal Alphas)")
    print("=" * 80)

    expr_rewards = {}
    for expr, reward in zip(history['expressions'], history['rewards']):
        if expr and expr != 'None':
            if expr not in expr_rewards or reward > expr_rewards[expr]:
                expr_rewards[expr] = reward

    sorted_exprs = sorted(expr_rewards.items(), key=lambda x: x[1], reverse=True)

    for i, (expr, reward) in enumerate(sorted_exprs[:30]):
        print(f"{i+1:2d}. AvgSharpe={reward:7.4f} | {expr}")

    # ----- Save to library -----
    library_dir = os.path.join(_PIPELINE_DIR, "alphas")
    os.makedirs(library_dir, exist_ok=True)
    library_path = os.path.join(library_dir, "batch_v2_alpha_library.json")

    if os.path.exists(library_path):
        library = AlphaLibrary.load(library_path)
    else:
        library = AlphaLibrary(name="batch_v2_discovered")

    library.add_from_training_history(
        history=history,
        top_n=30,
        min_sharpe=0.0,
        ticker=",".join(training_tickers),
        period="batch_v2_training"
    )

    library.save(library_path)
    print(f"Library saved to {library_path}")

    # Save full history
    history_path = os.path.join(_PIPELINE_DIR, "models", "batch_v2_training_history.pkl")
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")

    # ----- Detailed analysis of best alpha -----
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS OF BEST ALPHA")
    print("=" * 80)

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
                print(f"\nPositive: {positive}/{len(ticker_sharpes)} "
                      f"({positive/len(ticker_sharpes)*100:.1f}%)")
                break

    # ----- Show market regime analysis -----
    print("\n" + "=" * 80)
    print("MARKET REGIME ANALYSIS")
    print("=" * 80)
    print("Do different market conditions produce different alphas?")
    print()

    # Group episodes by market regime
    if history['market_features'] and history['expressions']:
        trending_exprs = []
        volatile_exprs = []
        calm_exprs = []

        for mkt, expr, reward in zip(
            history['market_features'],
            history['expressions'],
            history['rewards']
        ):
            if expr is None or expr == 'None' or reward <= 0:
                continue

            avg_trend = mkt[0]   # index 0 = avg_momentum
            avg_vol = mkt[2]     # index 2 = avg_volatility

            if avg_trend > 0.3:
                trending_exprs.append((expr, reward))
            elif avg_vol > 0.25:
                volatile_exprs.append((expr, reward))
            else:
                calm_exprs.append((expr, reward))

        for regime_name, regime_exprs in [
            ("TRENDING (avg_momentum > 0.3)", trending_exprs),
            ("VOLATILE (avg_vol > 0.25)", volatile_exprs),
            ("CALM (neither)", calm_exprs)
        ]:
            print(f"\n{regime_name}: {len(regime_exprs)} positive episodes")
            if regime_exprs:
                # Show top 5 expressions for this regime
                regime_best = sorted(regime_exprs, key=lambda x: x[1], reverse=True)
                for expr, reward in regime_best[:5]:
                    print(f"  Sharpe={reward:7.4f} | {expr}")