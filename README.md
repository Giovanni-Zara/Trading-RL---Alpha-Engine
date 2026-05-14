# Alpha Engine — Reinforcement Learning for Alpha Expression Discovery

## Overview

**Alpha Engine** is a reinforcement learning system that automatically discovers interpretable financial alpha expressions. Rather than training a black-box predictive model, the agent learns to compose **symbolic formulas** — human-readable mathematical expressions — that serve as trading signals (alpha factors).

The core loop works as follows:

1. The RL agent observes the current state (stack contents, market regime features).
2. It selects a **token** from a Domain-Specific Language (DSL): an operand (`Close`, `Volume`), a constant (`2`, `0.5`), an operator (`sub`, `ts_mean`), or a pre-built alpha primitive (`alpha_mom`, `alpha_rsi`).
3. Tokens are applied to a **stack** in Reverse Polish Notation (RPN) style, progressively building an expression tree.
4. When the stack contains exactly one complete expression, the episode ends and the expression is evaluated on historical OHLCV data.
5. The **reward** is the annualized Sharpe ratio of the strategy implied by the expression, averaged across multiple tickers.

The result is a library of interpretable formulas like:

```
sub(ts_mean(Close, 5), ts_mean(Close, 20))   →  short-term vs long-term momentum
alpha_mean_rev(10)                            →  10-day mean reversion signal
mul(alpha_mom(20), neg(alpha_vol_surge(10)))  →  momentum filtered by volume
```

These expressions can be evaluated directly on new data, inspected by a human, and composed into trading strategies.

---

## Architecture

```
PIPELINE_CLEAN/
│
├── DSL/                            # Domain-Specific Language
│   ├── utils.py                    # Token types, operators, math functions, alpha primitives
│   ├── node.py                     # ExpressionNode — tree node with recursive evaluation
│   ├── expression.py               # Expression — wrapper around root node
│   └── dsl.py                      # DSL — registry of all available tokens
│
├── agent/
│   └── agentPPO_enhanced.py        # PPO agent with GAE, entropy regularization, multi-episode batching
│
├── environment/
│   └── multi_env_v2.py             # Multi-ticker market-aware environment (production-aligned)
│
├── train/
│   └── train_PPO_enh_multi_v2.py   # Training script for V2 environment
│
├── alpha_trading/                  # Post-training alpha management
│   ├── alpha_record.py             # AlphaRecord dataclass (expression + performance metadata)
│   ├── alpha_library.py            # AlphaLibrary — store, load, rank, evaluate alphas
│   ├── generate_signals.py         # Evaluate alphas on new data → trading signals
│   └── save_alphas.py              # Persist best alphas from training history
│
├── simulations/
│   └── animate_episode.py          # Animate agent inference and stack building
│
├── data_pipeline/                  # Data acquisition utilities
│   ├── download_sp500_history.py   # Download S&P 500 historical components from GitHub
│   ├── sp500_tickers.py            # Extract S&P 500 constituents at a historical date
│   └── download_ohlcv.py           # Download OHLCV data via yfinance
│
├── intra_tests/                    # Unit/integration tests
│   ├── dsl_test.py                 # Tests for DSL tokens and operators
│   ├── expr_test.py                # Tests for expression building and evaluation
│   ├── env_test.py                 # Tests for environments
│   ├── agentPPO_test.py            # Tests for PPO agent
│   └── agentREINF_test.py          # Tests for REINFORCE agent
│
├── data/
│   └── ohlcv_data.csv              # Historical OHLCV data (extracted with download script)
|   └── data_utils_info&components/ # info and utils used for data developing
│   └── test/
|       └── TEST_ohlcv_data.csv     #test ohlcv data for inference
|   └── train/
|       └── ohlcv_data.csv          #train ohlcv data
├── alphas/
│   └── batch_v2_alpha_library.json # Discovered alpha expressions from training
│
├── models/                         # Saved agent checkpoints and training histories
│   ├── agent_batch_v2_primitives.pt
│   └── batch_v2_training_history.pkl
│
├── figures/                        # Output directory for visualization plots
└── results/                        # Output directory for analysis results
```

---

## Component Details

### 1. Domain-Specific Language (`DSL/`)

The DSL defines all the building blocks the agent can use to construct expressions.

#### `utils.py` — Tokens, Operators & Primitives

Defines the fundamental types:

| Token Type | Description | Examples |
|---|---|---|
| `OPERAND` | Raw OHLCV columns | `Close`, `Open`, `High`, `Low`, `Volume` |
| `CONSTANT` | Numeric literals | `-1`, `0`, `0.5`, `1`, `2` |
| `UNARY_OP` | Single-input transformations | `neg(x)`, `abs(x)`, `log(x)`, `sign(x)` |
| `BINARY_OP` | Two-input combinations | `add(x,y)`, `sub(x,y)`, `mul(x,y)`, `div(x,y)`, `max(x,y)`, `min(x,y)` |
| `TS_OP` | Time-series operators with a window | `ts_mean(x,d)`, `ts_std(x,d)`, `delay(x,d)`, `delta(x,d)`, `ts_rank(x,d)`, `ts_zscore(x,d)`, `returns(x,d)` |

Additionally, **known-good alpha primitives** are defined as standalone functions:

- **Momentum family**: `momentum`, `volatility_adjusted_momentum`, `price_acceleration`
- **Mean reversion family**: `mean_reversion`, `relative_strength`, `bollinger_position`
- **Volume family**: `volume_price_trend`, `volume_surge`, `price_volume_divergence`
- **Price action family**: `overnight_gap`, `intraday_range`, `close_location`

All math functions include safety guards (`safe_div` returns 0 instead of dividing by zero, `safe_log` handles non-positive inputs).

#### `node.py` — Expression Nodes

`ExpressionNode` is a recursive tree node. Each node holds:
- A `Token` (what operation or data it represents)
- A list of `children` (sub-expressions, empty for leaf nodes)
- An optional `window` parameter (for time-series operators)

Key methods:
- `evaluate(data)` — Recursively evaluates the subtree on a DataFrame, dispatching by token type.
- `to_string()` — Produces a human-readable formula like `ts_mean(sub(Close, Open), 5)`.
- `depth()` / `size()` — Measure expression complexity.

Alpha primitives are treated as special leaf nodes: they have no children in the tree but internally pull multiple columns from the DataFrame and apply their own logic.

#### `expression.py` — Expression Wrapper

`Expression` wraps a root `ExpressionNode` and provides a clean API:
- `evaluate(data) → pd.Series` — Evaluates the full tree, replaces infinities and NaNs with 0.
- `to_string()` — Formula string.
- `complexity()` — Node count (can be used as a regularization penalty).

A helper function `build_expression(dsl, token_name, children, window)` simplifies manual construction.

#### `dsl.py` — DSL Registry

The `DSL` class is the central registry that holds all available tokens. On initialization, it creates:
- 5 operands, 5 constants, 4 unary ops, 6 binary ops, 9 time-series ops
- 12 **alpha primitives** via `CompoundAlphaOperator` — these are pre-built factors encapsulating known-good alpha logic (e.g., RSI, Bollinger position, volume-price trend). They accept a window parameter and internally extract the required OHLCV columns.

`CompoundAlphaOperator` extends `Token` with `token_type = TS_OP` so it fits into the existing action space, but it pulls data directly from the DataFrame rather than operating on child nodes.

Methods:
- `get_all_tokens()` — Returns all tokens grouped by category (including `alpha_primitives`).
- `get_token(name)` — Lookup a token by name.
- `get_action_space()` — Flat list of all token names for the RL agent.
- `get_token_info()` — Summary table of all tokens as a DataFrame.

---

### 2. RL Agent (`agent/agentPPO_enhanced.py`)

The agent uses **Proximal Policy Optimization (PPO)** with several enhancements over vanilla implementations.

#### Network Architecture: `ActorCriticNetwork`

A shared-trunk architecture with two heads:

```
State (dim=26) → [Linear → LayerNorm → ReLU] × 2 → Shared Features (dim=H)
                                                       ├─→ Actor Head  → Action Probabilities (dim=A)
                                                       └─→ Critic Head → State Value (dim=1)
```

Both heads have an additional hidden layer (`H/2` neurons) for increased representational capacity. An **action mask** is applied to the actor logits before softmax, setting invalid actions to `-inf` so they receive zero probability.

#### PPO with Enhancements: `PPOAgentEnhanced`

| Feature | Description |
|---|---|
| **GAE (Generalized Advantage Estimation)** | Computes advantages using the recursive formula $A_t = \delta_t + \gamma\lambda(1-\text{done})A_{t+1}$, smoothly interpolating between TD(0) and Monte Carlo estimates. Parameter `gae_lambda` (default 0.95) controls the bias-variance trade-off. |
| **Entropy Regularization** | An entropy bonus is added to the loss, encouraging the policy to maintain exploration and avoid premature convergence on a narrow set of formulas. |
| **Multi-Episode Batching** | The agent collects `batch_episodes` complete episodes before performing a gradient update, providing richer and more stable gradient signals. |
| **Clipped Surrogate Objective** | The standard PPO clipping with `eps_clip` (default 0.2), preventing the policy from changing too aggressively in a single update. |
| **Gradient Clipping** | `clip_grad_norm_` with `max_norm=0.5` prevents exploding gradients. |

The combined loss is:

$$L = L_{\text{policy}} + c_v \cdot L_{\text{value}} - c_e \cdot H(\pi)$$

where $c_v$ is the value coefficient, $c_e$ is the entropy coefficient, and $H(\pi)$ is the policy entropy.

**Interface** (compatible with a REINFORCE agent for easy swapping):
- `select_action(state, action_mask) → int`
- `store_reward(reward)` / `store_done(done)`
- `should_update() → bool` — Returns `True` when `batch_episodes` episodes have been collected.
- `update() → float` — Runs `k_epochs` update passes on the buffered data, returns average loss.
- `save(path)` / `load(path)`

---

### 3. Environment (`environment/multi_env_v2.py`)

The production-aligned multi-ticker environment used by the main training script.

#### `MultiTickerAlphaEnvV2` — Market-Aware Multi-Ticker Environment

The agent builds an expression that is evaluated on **multiple tickers simultaneously**. A random date is sampled per episode, defining the market regime observed by the agent. The key insight: an alpha that works across ~19 diverse stocks is more likely to be real signal than one tuned to a single stock.

**Key design features**:

| Aspect | Description |
|---|---|
| **Episode sampling** | Random date → defines market state; reward uses rolling Sharpe over the full evaluation range |
| **State** | 16 market regime features + 10 stack features = **26 dimensions** |
| **Market awareness** | Short (5d) / Medium (20d) / Long (60d) cross-sectional regime features |
| **Alpha primitives** | Supported as leaf-node actions |
| **Window selection** | **Explicit window-selection actions**: 2, 3, 5, 10, 20, 50, 120 days |
| **Expression depth** | Minimum 3 steps (`MIN_EXPRESSION_DEPTH`), maximum 20 steps |

**State vector** (26 dimensions):

```
SHORT-TERM (5-day, 6 features):
  avg_momentum, momentum_dispersion, avg_volatility,
  avg_mean_reversion, avg_volume_ratio, pct_positive_return

MEDIUM-TERM (20-day, 6 features):
  avg_momentum, momentum_dispersion, avg_volatility,
  avg_mean_reversion, avg_volume_ratio, pct_positive_return

LONG-TERM (60-day, 4 features):
  avg_momentum, momentum_dispersion, avg_volatility,
  avg_mean_reversion

STACK STATE (10 features):
  stack_size, current_step, normalized_window, n_operands, n_constants,
  n_unary, n_binary, n_ts, n_alpha_primitives, steps_remaining
```

All market features are computed **cross-sectionally** (averaged across all tickers at the sampled date), giving the agent a read on the overall market regime.

**Episode lifecycle**:
1. Sample a random date from the valid range.
2. Compute market features at that date (state observation).
3. Agent builds an expression by selecting tokens + window actions step by step.
4. When the stack has exactly one complete expression (and ≥3 steps taken), the episode terminates.
5. The expression is evaluated via rolling Sharpe ratio across **all tickers over the full evaluation range**.

**Reward computation — Rolling Sharpe**:

For each ticker:
1. Evaluate the expression on the ticker's **full historical data** → raw alpha signal.
2. Apply **60-day rolling rank normalization**: convert the signal to a percentile rank within a 60-day window, then scale to $[-1, 1]$.
3. Compute **strategy returns** = normalized alpha × forward returns, where forward return on day $T$ = $(Close_{T+1} - Open_{T+1}) / Open_{T+1}$.
4. Compute **annualized Sharpe** = $\frac{\mu}{\sigma} \times \sqrt{252}$, clipped to $[-10, 10]$.
5. Average Sharpe across all tickers, with a **consistency adjustment**: if >70% of tickers have positive Sharpe → reward ×1.10; if <30% → reward ×0.85. Final reward clipped to $[-5, 5]$.

**Structural guard**: Expressions that don't reference any OHLCV column or alpha primitive receive a $-1.0$ reward without Sharpe evaluation.

**Performance optimizations**:
- All per-ticker features (at 5/20/60-day timeframes) and forward returns are **precomputed** once at initialization.
- Common dates across tickers are pre-identified.
- Structural checks gate expensive Sharpe computation.

---

### 4. Training Script (`train/train_PPO_enh_multi_v2.py`)

The main training loop for the V2 environment.

**Default hyperparameters** (as used in the `__main__` block):

| Parameter | Value | Rationale |
|---|---|---|
| `n_episodes` | 150,000 | Large budget for exploring the combinatorial expression space |
| `max_steps` | 20 | Allows expressions of moderate depth |
| `hidden_size` | 256 | Larger network to handle the 26-dim state |
| `learning_rate` | 0.0001 | Low LR for stability with PPO |
| `batch_episodes` | 64 | Collects 64 episodes per PPO update for stable gradients |
| `entropy_coef` | 0.10 (decaying) | Starts high for exploration, decays to 0.02 |
| `gamma` | 0.99 | Standard discount factor |
| `gae_lambda` | 0.95 | Standard GAE parameter |

**Training tickers** (19 diversified stocks):

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, INTC, CSCO |
| Finance | JPM, BAC, GS |
| Healthcare | JNJ, PFE, UNH |
| Consumer | WMT, KO, PG |
| Industrial / Aerospace | GE, CAT, BA |
| Energy | XOM, CVX |

**Training features**:
- **Entropy decay**: `entropy_coef` linearly decays over training from its initial value toward 0.02, shifting from exploration to exploitation.
- **Expression deduplication**: After 10 repeats of the same expression, a mild penalty (scaling down to ×0.5) is applied to encourage diversity.
- **Comprehensive logging**: Tracks rewards, expressions, per-ticker Sharpes, episode dates, and market features.
- **Alpha library**: Top 30 expressions are saved to a JSON-based `AlphaLibrary`.
- **Market regime analysis**: Post-training, episodes are grouped by market regime (trending / volatile / calm) to see if different conditions produce different alpha families.

**Output**:
- Saved agent checkpoint: `models/agent_batch_v2_primitives.pt`
- Full training history: `models/batch_v2_training_history.pkl`
- Alpha library: `alphas/batch_v2_alpha_library.json`

---

### 5. Alpha Management (`alpha_trading/`)

After training, discovered alphas are stored, evaluated, and converted to trading signals.

- **`alpha_record.py`** — `AlphaRecord` dataclass holding expression string, Sharpe ratio, mean return, standard deviation, max drawdown, turnover, market correlation, and metadata (creation date, training ticker, period, complexity).

- **`alpha_library.py`** — `AlphaLibrary` class that manages a collection of `AlphaRecord` objects. Supports adding alphas from training history, saving/loading to JSON, filtering by performance metrics, evaluating alphas on new data, and combining multiple alphas into an ensemble.

- **`generate_signals.py`** — Evaluates discovered alphas on new OHLCV data and generates trading signals (long / short / neutral). Supports equal-weight and Sharpe-weight ensemble combination methods.

- **`save_alphas.py`** — Utility to persist the best alphas from a training history dictionary into the alpha library.

---

### 6. Data Pipeline (`data_pipeline/`)

Utilities for acquiring and preparing the OHLCV dataset:

- **`download_sp500_history.py`** — Downloads the S&P 500 historical components CSV from GitHub (used to avoid survivorship bias).
- **`sp500_tickers.py`** — Extracts S&P 500 constituents at a specific historical date (default: `2016-02-09`), ensuring the training set reflects the index composition at the time of the data.
- **`download_ohlcv.py`** — Downloads daily OHLCV data for all tickers via `yfinance`. Default date range: `2016-02-09` to `2022-02-09`.

---

### 7. Simulations & Visualization (`simulations/`)

- **`visualize_inference.py`** — Generates presentation-ready plots showing the agent's step-by-step inference process: action probability distributions, expression tree growth, stack state evolution, and final evaluation. Outputs to the `figures/` directory.

---

### 8. Tests (`intra_tests/`)

Unit and integration tests for each component:

- **`dsl_test.py`** — Tests DSL token creation and operator behavior.
- **`expr_test.py`** — Tests expression tree building and evaluation.
- **`env_test.py`** — Tests environment reset, step, action masking, and reward computation.
- **`agentPPO_test.py`** — Tests PPO agent action selection, update, and save/load.
- **`agentREINF_test.py`** — Tests REINFORCE agent baseline.

---

## Reward Design Philosophy

The reward function is carefully designed to avoid common pitfalls in quantitative finance:

1. **No future leakage**: Rolling rank normalization uses only past data within a 60-day window. Forward returns use next-period open-to-close (not close-to-close), modeling the realistic scenario of observing at close of day $T$, entering at open of day $T+1$, and exiting at close of day $T+1$.

2. **Rank-based normalization**: Instead of z-score normalization (which gives trending signals like raw `Close` a trivially profitable trend-following signal), the alpha is converted to a 60-day rolling percentile rank. A monotonic series gets rank ≈ 0.5 → near-zero signal after centering.

3. **Cross-sectional evaluation**: Evaluating on 19 diversified tickers across multiple sectors prevents overfitting to a single stock's idiosyncrasies.

4. **Rolling evaluation**: The Sharpe ratio is computed over the entire valid evaluation range (not just a small forward window), providing a robust performance estimate.

5. **Consistency adjustment**: Cross-ticker agreement (>70% positive → bonus, <30% → penalty) favors expressions that generalize rather than exploit noise in a few tickers.

6. **Structural guards**: Expressions that don't reference any OHLCV data or alpha primitives receive a heavy penalty ($-1.0$), preventing the agent from gaming the reward with constant expressions.

---

## Data Format

The system expects a CSV file at `data/ohlcv_data.csv` with the following columns:

| Column | Type | Description |
|---|---|---|
| `Date` | string/datetime | Trading date |
| `Ticker` | string | Stock symbol (e.g., `AAPL`) |
| `Open` | float | Opening price |
| `High` | float | Highest price |
| `Low` | float | Lowest price |
| `Close` | float | Closing price |
| `Volume` | float/int | Trading volume |

Multiple tickers should be stacked vertically (long format). Each ticker should have at least ~320 rows of data (120 warmup + 200 minimum evaluation days).

---

## Requirements

```txt
numpy>=1.21
pandas>=1.3
torch>=1.10
matplotlib>=3.5
yfinance>=0.2       # only for data_pipeline/ download scripts
```

No GPU is required — training runs on CPU by default. For large-scale experiments, set `device='cuda'` in the agent constructor.

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd alpha_engine

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy pandas torch matplotlib

# Optional: only needed for downloading data
pip install yfinance
```

---

## How to Run

### 1. Prepare Data

**Option A — Download from S&P 500 (recommended)**:

```bash
cd PIPELINE_CLEAN

# Step 1: Download S&P 500 historical components
python data_pipeline/download_sp500_history.py

# Step 2: Extract tickers at a historical date (avoids survivorship bias)
python data_pipeline/sp500_tickers.py

# Step 3: Download OHLCV data for those tickers
python data_pipeline/download_ohlcv.py
```

**Option B — Provide your own data**:

Place your OHLCV CSV at `PIPELINE_CLEAN/data/ohlcv_data.csv`. Ensure it contains columns: `Date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Volume`, with multiple tickers stacked vertically.

### 2. Train the Agent

```bash
cd PIPELINE_CLEAN
python train/train_PPO_enh_multi_v2.py
```

This will:
- Load the data and select 19 diversified training tickers.
- Train for 150,000 episodes (~2,344 PPO updates with batch size 64).
- Print progress every 500 episodes with rolling reward, average Sharpe, and per-ticker breakdown.
- Save the trained agent to `models/agent_batch_v2_primitives.pt`.
- Save discovered alphas to `alphas/batch_v2_alpha_library.json`.
- Print the top 30 discovered expressions, a per-ticker breakdown of the best alpha, and a market regime analysis.

### 3. Customize Training

Edit the bottom of `train_PPO_enh_multi_v2.py` to change:

```python
agent, history = train_batch_v2(
    data=data,
    tickers=training_tickers,
    n_episodes=150000,        # Number of training episodes
    max_steps=20,             # Max expression depth
    hidden_size=256,          # Network size
    learning_rate=0.0001,     # Learning rate
    entropy_coef=0.10,        # Initial exploration coefficient (decays to 0.02)
    batch_episodes=64,        # Episodes per PPO update
    print_every=500,          # Logging frequency
    save_path="models/my_agent.pt"
)
```

### 4. Inspect Results

After training, examine:

- **Console output**: Top 30 expressions ranked by Sharpe, per-ticker breakdown of the best alpha, and market regime analysis (trending / volatile / calm).
- **`alphas/batch_v2_alpha_library.json`**: Machine-readable library of discovered alphas with metadata.
- **`models/batch_v2_training_history.pkl`**: Full training history for custom analysis.

```python
import pickle

with open("models/batch_v2_training_history.pkl", "rb") as f:
    history = pickle.load(f)

# history['rewards']          — list of per-episode rewards
# history['expressions']      — list of expression strings
# history['losses']           — list of PPO loss values
# history['ticker_sharpes']   — list of {ticker: sharpe} dicts
# history['episode_dates']    — list of sampled dates
# history['market_features']  — list of market feature vectors
```

### 5. Generate Trading Signals

After training, use the discovered alphas to generate signals on new data:

```python
from alpha_trading import AlphaLibrary

library = AlphaLibrary.load("alphas/batch_v2_alpha_library.json")
# Use generate_signals.py to evaluate alphas on new OHLCV data
```

### 6. Visualize Agent Inference

```bash
python simulations/visualize_inference.py
```

Generates plots in the `figures/` directory showing the agent's step-by-step expression building process.

---

## Example Discovered Expressions

Typical expressions found by the agent (results vary by dataset):

```
alpha_mean_rev(5)                              # 5-day mean reversion
sub(alpha_mom(20), alpha_mom(5))               # Momentum cross-over
mul(alpha_rsi(14), neg(alpha_vol_surge(10)))   # RSI filtered by unusual volume
ts_zscore(sub(Close, ts_mean(Close, 20)), 60)  # Z-score of deviation from 20-day MA
div(delta(Close, 1), ts_std(Close, 10))        # Normalized daily change
```

Each expression is fully interpretable and can be evaluated on new data with a single call to `expr.evaluate(new_data)`.

---

## Created by [Giovanni Zara](https://github.com/Giovanni-Zara) and [Paolo Laurenti](https://github.com/PaoloLaur)

## License

See [LICENSE.txt](LICENSE.txt).
