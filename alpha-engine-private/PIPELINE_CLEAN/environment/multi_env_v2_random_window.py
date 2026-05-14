"""
Multi-Ticker Environment with Market-Aware State (V2) — FIXED

Root cause of training/OOS discrepancy:
    OLD: Reward = Sharpe over ENTIRE 6-year training period (fixed window every episode)
         → Agent overfits to 2016-2022 over 150K episodes
         → episode_date used for state but NOT for reward = meaningless
    
    NEW: Reward = Sharpe over RANDOM sub-window per episode
         → Agent must generalize across many different market periods
         → episode_date defines BOTH state AND evaluation window
         → Train/validation split detects overfitting during training

Changes from old version:
    1. Train/validation split (80/20) in _find_common_dates
    2. Reward uses random sub-window (EVAL_WINDOW days) anchored to episode_date  
    3. Removed consistency bonus that inflated training metrics
    4. Added validate_expression() for overfitting detection
    5. MIN_EVAL_DAYS reduced to 60 (compatible with shorter eval windows)
    6. EVAL_WINDOW = 120 days (~6 months) per episode

State structure (24 dimensions):
[
    # Short-term regime (5-day lookback) - 6 features
    avg_momentum_5, momentum_dispersion_5, avg_volatility_5,
    avg_mean_reversion_5, avg_volume_ratio_5, pct_positive_return_5,

    # Medium-term regime (20-day lookback) - 6 features  
    avg_momentum_20, momentum_dispersion_20, avg_volatility_20,
    avg_mean_reversion_20, avg_volume_ratio_20, pct_positive_return_20,

    # Long-term regime (60-day lookback) - 4 features
    avg_momentum_60, momentum_dispersion_60, avg_volatility_60,
    avg_mean_reversion_60,

    # Stack state - 10 features
    stack_size, current_step, window_norm, n_operands, n_constants, 
    n_unary, n_binary, n_ts, n_alpha_primitives, steps_remaining
]
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from DSL import DSL
from DSL import Expression
from DSL import ExpressionNode
from DSL import TokenType
from DSL import CompoundAlphaOperator


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_TRAINING_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'INTC', 'CSCO',
    'JPM', 'BAC', 'GS', 'WFC',
    'JNJ', 'PFE', 'UNH', 'MRK',
    'WMT', 'KO', 'PG', 'MCD',
    'GE', 'CAT', 'BA',
    'XOM', 'CVX',
]


class MultiTickerAlphaEnvV2:
    """
    Environment with production-aligned evaluation.
    
    Key design principle: ONE date per episode.
    The episode date defines BOTH the market state AND the evaluation window.
    This ensures the agent learns: "given THIS market regime → build THIS alpha"
    
    Anti-overfitting measures:
    - Random eval window per episode (not fixed full-period)
    - Train/validation split (80/20)
    - No consistency bonus inflating rewards
    - Validation method for periodic overfitting checks
    
    Noise reduction comes from:
    - 19 tickers per evaluation (cross-sectional averaging)
    - Many episodes sampling different dates (temporal averaging via PPO batches)
    """

    SHORT_WINDOW = 5
    MEDIUM_WINDOW = 20
    LONG_WINDOW = 60

    # Rolling evaluation parameters
    WARMUP_PERIOD = 120          # Days needed before we can compute alpha
    MIN_EVAL_DAYS = 60           # Minimum days for Sharpe calculation (reduced for sub-windows)
    
    # Expression constraints
    MIN_EXPRESSION_DEPTH = 3     # Minimum steps before expression can complete
    
    # Evaluation parameters
    EVAL_WINDOW = 120            # Days to evaluate expression per episode (~6 months)
    SHARPE_LOOKBACK = 60         # Days for rolling normalization warmup
    
    # Window constraints
    MIN_TS_WINDOW = 2
    WINDOW_OPTIONS = [2, 3, 5, 10, 20, 50, 120]

    # Train/validation split
    VALIDATION_RATIO = 0.20      # Hold out last 20% of training data

    def __init__(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        max_steps: int = 20,
        min_sharpe_tickers: int = None,
    ):
        self.max_steps = max_steps
        self.window_options = self.WINDOW_OPTIONS
        self.min_sharpe_tickers = min_sharpe_tickers or len(tickers) // 3

        # ----- Organize data by ticker -----
        self.ticker_data: Dict[str, pd.DataFrame] = {}
        self.tickers = []

        for ticker in tickers:
            ticker_df = data[data['Ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
            if len(ticker_df) > self.WARMUP_PERIOD + self.MIN_EVAL_DAYS:
                self.ticker_data[ticker] = ticker_df
                self.tickers.append(ticker)

        if len(self.tickers) == 0:
            raise ValueError("No tickers have sufficient data!")

        print(f"Loaded {len(self.tickers)} tickers with sufficient data")
        self.active_tickers = self.tickers

        # ----- Precompute -----
        self._precompute_ticker_features()
        self._find_common_dates()
        self._precompute_returns()

        # ----- Initialize DSL -----
        self.dsl = DSL()
        self._build_action_space()

        # ----- State dimensions -----
        self.short_feature_size = 6
        self.medium_feature_size = 6
        self.long_feature_size = 4
        self.market_feature_size = self.short_feature_size + self.medium_feature_size + self.long_feature_size
        self.stack_feature_size = 10

        # ----- Episode state -----
        self.stack: List[ExpressionNode] = []
        self.current_step = 0
        self.current_window = self.window_options[2]
        self.episode_date = None

    def _find_common_dates(self):
        """
        Find common dates AND split into train/validation.
        
        Timeline:
        |--- warmup ---|--- train eval (80%) ---|--- validation (20%) ---|
        
        The agent's reward is computed ONLY on the train portion.
        The validation portion is used for periodic overfitting checks.
        
        Within the train portion, each episode samples a random start date
        and evaluates over EVAL_WINDOW days forward from that date.
        """
        date_sets = []
        for ticker, df in self.ticker_data.items():
            dates = set(df['Date'].tolist())
            date_sets.append(dates)

        common_dates = sorted(list(set.intersection(*date_sets)))

        if len(common_dates) < self.WARMUP_PERIOD + self.MIN_EVAL_DAYS:
            raise ValueError(f"Not enough common dates: {len(common_dates)}")

        self.common_dates = common_dates

        self.date_to_idx: Dict[str, Dict[str, int]] = {}
        for ticker, df in self.ticker_data.items():
            self.date_to_idx[ticker] = {
                date: idx for idx, date in enumerate(df['Date'].tolist())
            }

        # =====================================================================
        # TRAIN / VALIDATION SPLIT
        # =====================================================================
        n_total = len(common_dates)
        n_warmup = self.WARMUP_PERIOD
        n_usable = n_total - n_warmup - 1  # -1 because forward return needs T+1

        # Split usable dates: first 80% for training eval, last 20% for validation
        n_val = int(n_usable * self.VALIDATION_RATIO)
        n_train = n_usable - n_val

        # Train eval range (indices into common_dates)
        self.train_start_idx = n_warmup
        self.train_end_idx = n_warmup + n_train

        # Validation eval range
        self.val_start_idx = self.train_end_idx
        self.val_end_idx = n_total - 1

        # Named dates for convenience
        self.eval_start_date = common_dates[self.train_start_idx]
        self.eval_end_date = common_dates[self.train_end_idx]

        self.val_start_date = common_dates[self.val_start_idx]
        self.val_end_date = common_dates[self.val_end_idx]

        # Valid start dates for episode sampling:
        # Each episode evaluates EVAL_WINDOW days forward from episode_date
        # So episode_date must be at least EVAL_WINDOW days before train_end
        # and at least WARMUP_PERIOD days into the data
        max_start_common_idx = self.train_end_idx - self.EVAL_WINDOW
        if max_start_common_idx <= n_warmup:
            # Fallback: if training data is too short for EVAL_WINDOW,
            # use whatever is available
            max_start_common_idx = self.train_end_idx - self.MIN_EVAL_DAYS

        self.valid_start_dates = common_dates[n_warmup:max(n_warmup + 1, max_start_common_idx)]

        print(f"Common date range:  {common_dates[0]} to {common_dates[-1]} ({n_total} days)")
        print(f"Train eval range:   {self.eval_start_date} to {self.eval_end_date} ({n_train} days)")
        print(f"Val eval range:     {self.val_start_date} to {self.val_end_date} ({n_val} days)")
        print(f"Valid start dates:  {len(self.valid_start_dates)} "
              f"(episode window: {self.EVAL_WINDOW} days)")

    # =========================================================================
    # PRECOMPUTATION
    # =========================================================================

    def _precompute_features_for_window(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Compute features for a single ticker at a single timeframe."""
        features = pd.DataFrame(index=df.index)
        returns = df['Close'].pct_change()

        features['momentum'] = returns.rolling(window, min_periods=window).sum()
        features['volatility'] = returns.rolling(window, min_periods=window).std() * np.sqrt(252)

        ma = df['Close'].rolling(window, min_periods=window).mean()
        std = df['Close'].rolling(window, min_periods=window).std()
        features['mean_reversion'] = ((df['Close'] - ma) / (std + 1e-8)).clip(-13, 13)

        vol_ma = df['Volume'].rolling(window, min_periods=window).mean()
        features['volume_ratio'] = np.log1p(df['Volume'] / (vol_ma + 1))

        features['return_positive'] = (returns.rolling(window, min_periods=window).sum() > 0).astype(float)

        return features

    def _precompute_ticker_features(self):
        """Precompute features at all three timeframes for all tickers."""
        self.ticker_features_short: Dict[str, pd.DataFrame] = {}
        self.ticker_features_medium: Dict[str, pd.DataFrame] = {}
        self.ticker_features_long: Dict[str, pd.DataFrame] = {}

        for ticker, df in self.ticker_data.items():
            self.ticker_features_short[ticker] = self._precompute_features_for_window(df, self.SHORT_WINDOW)
            self.ticker_features_medium[ticker] = self._precompute_features_for_window(df, self.MEDIUM_WINDOW)
            self.ticker_features_long[ticker] = self._precompute_features_for_window(df, self.LONG_WINDOW)

        print(f"Precomputed features at {self.SHORT_WINDOW}/{self.MEDIUM_WINDOW}/{self.LONG_WINDOW}-day timeframes")

    def _precompute_returns(self):
        """Precompute forward returns for each ticker once."""
        self.ticker_forward_returns: Dict[str, pd.Series] = {}
        for ticker, df in self.ticker_data.items():
            self.ticker_forward_returns[ticker] = (
                (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
            )
        print("Precomputed forward returns for all tickers")

    # =========================================================================
    # MARKET FEATURES
    # =========================================================================

    def _collect_cross_sectional(self, features_dict: Dict[str, pd.DataFrame], date: str) -> Dict[str, List[float]]:
        collected = defaultdict(list)
        for ticker in self.tickers:
            if date not in self.date_to_idx[ticker]:
                continue
            idx = self.date_to_idx[ticker][date]
            features = features_dict[ticker]
            if idx >= len(features):
                continue
            row = features.iloc[idx]
            for col in features.columns:
                val = row[col]
                if not pd.isna(val):
                    collected[col].append(val)
        return collected

    def _aggregate_features(self, collected: Dict[str, List[float]], include_pct_positive: bool = True) -> List[float]:
        def safe_mean(arr):
            return float(np.mean(arr)) if len(arr) > 0 else 0.0
        def safe_std(arr):
            return float(np.std(arr)) if len(arr) > 1 else 0.0

        result = [
            safe_mean(collected.get('momentum', [])) * 10,
            safe_std(collected.get('momentum', [])) * 10,
            safe_mean(collected.get('volatility', [])),
            safe_mean(collected.get('mean_reversion', [])),
            safe_mean(collected.get('volume_ratio', [])),
        ]
        if include_pct_positive:
            pos_returns = collected.get('return_positive', [])
            pct_positive = safe_mean(pos_returns) if pos_returns else 0.5
            result.append(pct_positive)
        return result

    def _compute_market_features(self, date: str) -> np.ndarray:
        short_collected = self._collect_cross_sectional(self.ticker_features_short, date)
        short_features = self._aggregate_features(short_collected, include_pct_positive=True)

        medium_collected = self._collect_cross_sectional(self.ticker_features_medium, date)
        medium_features = self._aggregate_features(medium_collected, include_pct_positive=True)

        long_collected = self._collect_cross_sectional(self.ticker_features_long, date)
        long_features = self._aggregate_features(long_collected, include_pct_positive=False)
        long_features = long_features[:4]

        all_features = short_features + medium_features + long_features
        result = np.array(all_features, dtype=np.float32)
        result = np.clip(result, -20, 20)
        return result

    # =========================================================================
    # ACTION SPACE (with window selection as action)
    # =========================================================================

    def _build_action_space(self):
        """
        Build action space including alpha primitives.
        Alpha primitives act as leaf nodes that can be combined.
        """
        self.actions = []
        self.action_to_token = {}
        self.action_is_window = {}
        self.action_is_alpha_primitive = {}

        all_tokens = self.dsl.get_all_tokens()

        # Standard operands and constants
        for category in ['operands', 'constants']:
            for name, token in all_tokens[category].items():
                action_id = len(self.actions)
                self.actions.append(name)
                self.action_to_token[action_id] = token
                self.action_is_window[action_id] = False
                self.action_is_alpha_primitive[action_id] = False

        # Operators
        for category in ['unary_ops', 'binary_ops', 'ts_ops']:
            for name, token in all_tokens[category].items():
                action_id = len(self.actions)
                self.actions.append(name)
                self.action_to_token[action_id] = token
                self.action_is_window[action_id] = False
                self.action_is_alpha_primitive[action_id] = False

        # Alpha primitives (these are leaf nodes!)
        for name, token in all_tokens['alpha_primitives'].items():
            action_id = len(self.actions)
            self.actions.append(name)
            self.action_to_token[action_id] = token
            self.action_is_window[action_id] = False
            self.action_is_alpha_primitive[action_id] = True

        # Window selection actions
        for w in self.window_options:
            action_id = len(self.actions)
            self.actions.append(f'window_{w}')
            self.action_to_token[action_id] = None
            self.action_is_window[action_id] = True
            self.action_is_alpha_primitive[action_id] = False

        self.n_actions = len(self.actions)
        self.action_name_to_id = {name: i for i, name in enumerate(self.actions)}
        
        # Count primitives for logging
        n_primitives = sum(1 for v in self.action_is_alpha_primitive.values() if v)
        print(f"Action space: {self.n_actions} total ({n_primitives} alpha primitives)")

    @property
    def action_space_size(self) -> int:
        return self.n_actions

    @property
    def state_size(self) -> int:
        return self.market_feature_size + self.stack_feature_size

    # =========================================================================
    # EPISODE LIFECYCLE
    # =========================================================================

    def reset(self) -> np.ndarray:
        """
        Reset for new episode. 
        ONE date = ONE market state = ONE evaluation window.
        
        The episode_date now defines BOTH:
        - The market features (state)
        - The evaluation window (reward)
        
        This is the key fix: the agent can't overfit to one fixed period
        because each episode evaluates on a different random sub-window.
        """
        self.episode_date = np.random.choice(self.valid_start_dates)
        self.stack = []
        self.current_step = 0
        self.current_window = self.window_options[2]
        return self._get_state()

    def _get_stack_state(self) -> np.ndarray:
        """
        Enhanced stack state including alpha primitive count.
        
        Returns exactly self.stack_feature_size (10) features:
        [stack_size, current_step, window_norm, n_operands, n_constants, 
         n_unary, n_binary, n_ts, n_alpha_primitives, steps_remaining]
        """
        type_counts = {tt: 0 for tt in TokenType}
        n_alpha_primitives = 0
        
        for node in self.stack:
            if hasattr(node, 'is_alpha_primitive') and node.is_alpha_primitive():
                n_alpha_primitives += 1
            else:
                type_counts[node.token.token_type] += 1

        return np.array([
            len(self.stack),                          # 0: stack size
            self.current_step,                        # 1: current step
            self.current_window / 50.0,               # 2: normalized window
            type_counts[TokenType.OPERAND],           # 3: operand count
            type_counts[TokenType.CONSTANT],          # 4: constant count
            type_counts[TokenType.UNARY_OP],          # 5: unary op count
            type_counts[TokenType.BINARY_OP],         # 6: binary op count
            type_counts[TokenType.TS_OP],             # 7: ts op count
            n_alpha_primitives,                       # 8: alpha primitive count
            self.max_steps - self.current_step,       # 9: steps remaining
        ], dtype=np.float32)

    def _get_state(self) -> np.ndarray:
        market_features = self._compute_market_features(self.episode_date)
        stack_state = self._get_stack_state()
        return np.concatenate([market_features, stack_state])

    def _get_valid_actions(self) -> List[int]:
        """Get valid actions including alpha primitives."""
        valid = []
        stack_size = len(self.stack)
        
        for action_id in range(self.n_actions):
            # Window selection always valid
            if self.action_is_window[action_id]:
                valid.append(action_id)
                continue
            
            # Alpha primitives act as leaf nodes (push to stack)
            if self.action_is_alpha_primitive[action_id]:
                valid.append(action_id)
                continue

            token = self.action_to_token[action_id]
            token_type = token.token_type

            if token_type in (TokenType.OPERAND, TokenType.CONSTANT):
                valid.append(action_id)
            elif token_type in (TokenType.UNARY_OP, TokenType.TS_OP):
                if stack_size >= 1:
                    valid.append(action_id)
            elif token_type == TokenType.BINARY_OP:
                if stack_size >= 2:
                    valid.append(action_id)
        
        return valid

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for action_id in self._get_valid_actions():
            mask[action_id] = 1.0
        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step - reward computed via RANDOM WINDOW Sharpe."""

        # ---- Handle window selection ----
        if self.action_is_window[action]:
            window_str = self.actions[action]
            self.current_window = int(window_str.split('_')[1])
            return self._get_state(), 0.0, False, {'window_set': self.current_window}

        # ---- Normal token action ----
        self.current_step += 1

        token = self.action_to_token[action]
        
        # ---- Alpha primitive: create leaf node with current window ----
        if self.action_is_alpha_primitive.get(action, False):
            node = ExpressionNode(token, children=[], window=self.current_window)
            self.stack.append(node)
        else:
            token_type = token.token_type

            if token_type in (TokenType.OPERAND, TokenType.CONSTANT):
                node = ExpressionNode(token)
                self.stack.append(node)
            elif token_type == TokenType.UNARY_OP:
                child = self.stack.pop()
                node = ExpressionNode(token, children=[child])
                self.stack.append(node)
            elif token_type == TokenType.BINARY_OP:
                right = self.stack.pop()
                left = self.stack.pop()
                node = ExpressionNode(token, children=[left, right])
                self.stack.append(node)
            elif token_type == TokenType.TS_OP:
                child = self.stack.pop()
                node = ExpressionNode(token, children=[child], window=self.current_window)
                self.stack.append(node)

        # ---- Check termination ----
        done = False
        reward = 0.0
        info = {'expression': None, 'ticker_sharpes': {}, 'raw_sharpe': 0.0}

        can_complete = (
            len(self.stack) == 1
            and self.current_step >= self.MIN_EXPRESSION_DEPTH
        )

        if can_complete:
            done = True
            expr = Expression(self.stack[0])
            expr_str = expr.to_string()

            # Quick structural check
            structural_check = self._compute_structural_reward(expr)

            if structural_check < -1.0:
                reward = -1.0
                info['expression'] = expr_str
                info['raw_sharpe'] = 0.0
                info['skipped_eval'] = True
            else:
                # RANDOM WINDOW SHARPE EVALUATION (key fix)
                reward, ticker_sharpes = self._compute_rolling_sharpe(expr)
                info['expression'] = expr_str
                info['ticker_sharpes'] = ticker_sharpes
                info['raw_sharpe'] = reward

        elif self.current_step >= self.max_steps:
            done = True
            reward = -1.0
            info['timeout'] = True

        return self._get_state(), reward, done, info

    # =========================================================================
    # STRUCTURAL REWARD
    # =========================================================================

    def _compute_structural_reward(self, expr: Expression) -> float:
        """
        Minimal structural check - ONLY reject truly broken expressions.
        No bonuses, just pass/fail gate.
        
        Returns:
            0.0 if expression is valid
            -2.0 if expression is broken (skip Sharpe eval)
        """
        expr_str = expr.to_string()
        
        # Must use price/volume data OR alpha primitives
        uses_ohlcv = any(op in expr_str for op in ['Open', 'High', 'Low', 'Close', 'Volume'])
        uses_alpha = 'alpha_' in expr_str
        
        if not (uses_ohlcv or uses_alpha):
            return -2.0  # Broken: doesn't use any market data
        
        # Pass - let Sharpe determine the reward
        return 0.0

    # =========================================================================
    # REWARD COMPUTATION — RANDOM SUB-WINDOW (FIXED)
    # =========================================================================

    def _compute_rolling_sharpe(self, expr: Expression) -> Tuple[float, Dict[str, float]]:
        """
        Compute Sharpe ratio on a RANDOM SUB-WINDOW of training data.
        
        CRITICAL FIX: Instead of evaluating on the entire training period
        (which allows overfitting over 150K episodes), each episode evaluates
        on a random EVAL_WINDOW-day slice anchored to self.episode_date.
        
        This means:
        - Episode 1 (date=2017-03-15): eval on 2017-03-15 to 2017-09-15
        - Episode 2 (date=2019-11-01): eval on 2019-11-01 to 2020-05-01
        - Episode 3 (date=2018-06-20): eval on 2018-06-20 to 2018-12-20
        
        The agent must find expressions that work across MANY different
        sub-periods, not just one that fits the entire dataset.
        
        Returns:
            avg_sharpe: Average Sharpe across tickers (no consistency bonus)
            ticker_sharpes: Dict of per-ticker Sharpe ratios
        """
        ticker_sharpes: Dict[str, float] = {}

        for ticker in self.tickers:
            sharpe = self._compute_single_ticker_window_sharpe(
                expr, ticker, self.episode_date
            )
            ticker_sharpes[ticker] = sharpe

        valid_sharpes = [s for s in ticker_sharpes.values() if s > -10]

        if len(valid_sharpes) == 0:
            return -1.0, ticker_sharpes

        avg_sharpe = float(np.mean(valid_sharpes))

        # NO consistency bonus — removed to prevent reward inflation.
        # The old code added +10% when >70% of tickers were positive,
        # which inflated training metrics and masked overfitting.

        return float(np.clip(avg_sharpe, -5, 5)), ticker_sharpes

    def _compute_single_ticker_window_sharpe(
        self, expr: Expression, ticker: str, start_date: str
    ) -> float:
        """
        Compute Sharpe for one ticker on a SPECIFIC date window.
        
        This is the key anti-overfitting measure:
        - Evaluate alpha on full history (backward-looking ops need it)
        - But only measure Sharpe on [start_date, start_date + EVAL_WINDOW]
        - Each episode gets a different window via random episode_date
        
        Timeline:
        |======= full history (for alpha computation) =======|
                              |--- EVAL_WINDOW ---|
                              ^                   ^
                              start_date          start + 120 days
        
        Args:
            expr: Expression to evaluate
            ticker: Ticker symbol
            start_date: Episode date — start of evaluation window
            
        Returns:
            Sharpe ratio for this ticker on this window, clipped to [-10, 10]
        """
        df = self.ticker_data[ticker]

        try:
            # Step 1: Evaluate expression on FULL data
            # (backward-looking ops like ts_mean need history)
            alpha_raw = expr.evaluate(df)

            # Step 2: Check alpha quality
            alpha_clean = alpha_raw.replace(0, np.nan).dropna()
            if len(alpha_clean) < self.WARMUP_PERIOD or alpha_clean.std() < 1e-8:
                return 0.0

            # Step 3: Rolling rank normalization (so alpha is comparable across time)
            alpha_rank = alpha_raw.rolling(
                window=60,       # 60-day rolling rank
                min_periods=30
            ).rank(pct=True)
            alpha_normalized = (alpha_rank - 0.5) * 2.0  # Scale to [-1, 1]

            # Step 4: Get forward returns (precomputed)
            forward_returns = self.ticker_forward_returns[ticker]

            # Step 5: Find eval window indices FOR THIS EPISODE
            start_idx = self.date_to_idx[ticker].get(start_date)
            if start_idx is None:
                return 0.0

            end_idx = min(start_idx + self.EVAL_WINDOW, len(df) - 1)

            if end_idx - start_idx < self.MIN_EVAL_DAYS:
                return 0.0

            # Step 6: Strategy returns over THIS window only
            strategy_returns = (
                alpha_normalized.iloc[start_idx:end_idx] *
                forward_returns.iloc[start_idx:end_idx]
            ).dropna()

            if len(strategy_returns) < self.MIN_EVAL_DAYS:
                return 0.0

            # Step 7: Compute Sharpe
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()

            if std_ret < 1e-10:
                return 0.0

            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            return float(np.clip(sharpe, -10, 10))

        except Exception:
            return -1.0

    # =========================================================================
    # FULL-PERIOD SHARPE (for logging / library — NOT for reward)
    # =========================================================================

    def compute_full_train_sharpe(self, expr: Expression) -> Tuple[float, Dict[str, float]]:
        """
        Compute Sharpe over the FULL training period.
        
        USE ONLY FOR:
        - Logging the "library Sharpe" when saving expressions
        - Comparing with validation Sharpe to detect overfitting
        
        DO NOT USE FOR REWARD — that would reintroduce the overfitting bug.
        """
        ticker_sharpes: Dict[str, float] = {}

        for ticker in self.tickers:
            sharpe = self._compute_single_ticker_full_sharpe(expr, ticker)
            ticker_sharpes[ticker] = sharpe

        valid_sharpes = [s for s in ticker_sharpes.values() if s > -10]

        if len(valid_sharpes) == 0:
            return -1.0, ticker_sharpes

        avg_sharpe = float(np.mean(valid_sharpes))
        return float(np.clip(avg_sharpe, -5, 5)), ticker_sharpes

    def _compute_single_ticker_full_sharpe(self, expr: Expression, ticker: str) -> float:
        """
        Compute Sharpe for one ticker over the full TRAIN period.
        (eval_start_date to eval_end_date, NOT including validation)
        """
        df = self.ticker_data[ticker]

        try:
            alpha_raw = expr.evaluate(df)

            alpha_clean = alpha_raw.replace(0, np.nan).dropna()
            if len(alpha_clean) < self.WARMUP_PERIOD or alpha_clean.std() < 1e-8:
                return 0.0

            alpha_rank = alpha_raw.rolling(
                window=60, min_periods=30
            ).rank(pct=True)
            alpha_normalized = (alpha_rank - 0.5) * 2.0

            forward_returns = self.ticker_forward_returns[ticker]

            eval_start_idx = self.date_to_idx[ticker].get(self.eval_start_date)
            eval_end_idx = self.date_to_idx[ticker].get(self.eval_end_date)

            if eval_start_idx is None or eval_end_idx is None:
                return 0.0

            strategy_returns = (
                alpha_normalized.iloc[eval_start_idx:eval_end_idx] *
                forward_returns.iloc[eval_start_idx:eval_end_idx]
            ).dropna()

            if len(strategy_returns) < self.MIN_EVAL_DAYS:
                return 0.0

            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()

            if std_ret < 1e-10:
                return 0.0

            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            return float(np.clip(sharpe, -10, 10))

        except Exception:
            return -1.0

    # =========================================================================
    # VALIDATION (call periodically during training)
    # =========================================================================

    def validate_expression(self, expr: Expression) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate expression on the HELD-OUT validation period.
        
        Call this every N episodes to check for overfitting:
        - If train Sharpe >> val Sharpe → overfitting
        - If train Sharpe ≈ val Sharpe → generalizing well
        
        Args:
            expr: Expression to validate
            
        Returns:
            avg_sharpe: Average Sharpe on validation period
            ticker_sharpes: Per-ticker Sharpe on validation period
        """
        ticker_sharpes: Dict[str, float] = {}

        for ticker in self.tickers:
            df = self.ticker_data[ticker]

            try:
                alpha_raw = expr.evaluate(df)

                alpha_clean = alpha_raw.replace(0, np.nan).dropna()
                if len(alpha_clean) < self.WARMUP_PERIOD or alpha_clean.std() < 1e-8:
                    ticker_sharpes[ticker] = 0.0
                    continue

                alpha_rank = alpha_raw.rolling(window=60, min_periods=30).rank(pct=True)
                alpha_normalized = (alpha_rank - 0.5) * 2.0

                forward_returns = self.ticker_forward_returns[ticker]

                val_start = self.date_to_idx[ticker].get(self.val_start_date)
                val_end = self.date_to_idx[ticker].get(self.val_end_date)

                if val_start is None or val_end is None:
                    ticker_sharpes[ticker] = 0.0
                    continue

                strategy_returns = (
                    alpha_normalized.iloc[val_start:val_end] *
                    forward_returns.iloc[val_start:val_end]
                ).dropna()

                if len(strategy_returns) < 30:
                    ticker_sharpes[ticker] = 0.0
                    continue

                mean_ret = strategy_returns.mean()
                std_ret = strategy_returns.std()

                if std_ret < 1e-10:
                    ticker_sharpes[ticker] = 0.0
                    continue

                sharpe = (mean_ret / std_ret) * np.sqrt(252)
                ticker_sharpes[ticker] = float(np.clip(sharpe, -10, 10))

            except Exception:
                ticker_sharpes[ticker] = -1.0

        valid = [s for s in ticker_sharpes.values() if s > -10]
        avg = float(np.mean(valid)) if valid else 0.0

        return avg, ticker_sharpes

    # =========================================================================
    # LEGACY COMPATIBILITY (kept for old code that references these)
    # =========================================================================

    def _get_eval_data_extended(self, ticker: str, date: str) -> Optional[pd.DataFrame]:
        """
        Get extended data: SHARPE_LOOKBACK before + EVAL_WINDOW after.
        
        Kept for backward compatibility. Not used by the new reward computation.
        """
        df = self.ticker_data[ticker]
        if date not in self.date_to_idx[ticker]:
            return None

        eval_start_idx = self.date_to_idx[ticker][date]
        data_start_idx = max(0, eval_start_idx - self.SHARPE_LOOKBACK)
        eval_end_idx = min(eval_start_idx + self.EVAL_WINDOW, len(df))

        if eval_end_idx - eval_start_idx < self.MIN_EVAL_DAYS:
            return None
        if eval_start_idx - data_start_idx < 20:
            return None

        return df.iloc[data_start_idx:eval_end_idx].copy()

    def _get_eval_returns_extended(self, ticker: str, date: str) -> Optional[Tuple[pd.Series, int]]:
        """Get precomputed returns for the extended window. Kept for backward compatibility."""
        df = self.ticker_data[ticker]
        if date not in self.date_to_idx[ticker]:
            return None

        eval_start_idx = self.date_to_idx[ticker][date]
        data_start_idx = max(0, eval_start_idx - self.SHARPE_LOOKBACK)
        eval_end_idx = min(eval_start_idx + self.EVAL_WINDOW, len(df))

        if eval_end_idx - eval_start_idx < self.MIN_EVAL_DAYS:
            return None
        if eval_start_idx - data_start_idx < 20:
            return None

        returns = self.ticker_forward_returns[ticker].iloc[data_start_idx:eval_end_idx]
        eval_offset = eval_start_idx - data_start_idx

        return returns, eval_offset