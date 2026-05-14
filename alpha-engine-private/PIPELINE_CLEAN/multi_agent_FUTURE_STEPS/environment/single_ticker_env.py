"""
Single-Ticker Alpha Environment

Key differences from MultiTickerAlphaEnvV2:
- State features come from ONE ticker only (no cross-sectional averaging)
- Reward is rolling Sharpe ratio for that single ticker over ALL valid dates
- Agent learns ticker-specific patterns

Rolling evaluation (same as V2):
    1. Evaluate expression on FULL ticker history
    2. Rolling rank normalization (60-day window)
    3. Compute strategy returns = alpha[T] * forward_return[T] for ALL valid dates
    4. Sharpe = mean(strategy_returns) / std(strategy_returns) * sqrt(252)

State structure (24 dimensions):
[
    # Short-term regime (5-day lookback) - 5 features
    momentum_5, volatility_5, mean_reversion_5, volume_ratio_5, return_sign_5,

    # Medium-term regime (20-day lookback) - 5 features  
    momentum_20, volatility_20, mean_reversion_20, volume_ratio_20, return_sign_20,

    # Long-term regime (60-day lookback) - 4 features
    momentum_60, volatility_60, mean_reversion_60, volume_ratio_60,

    # Stack state - 10 features
    stack_size, current_step, current_window,
    n_operands, n_constants, n_unary, n_binary, n_ts,
    expression_depth, can_complete
]
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import List, Dict, Tuple, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MULTI_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)
_PIPELINE_DIR = os.path.dirname(_MULTI_AGENT_DIR)

if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from DSL import DSL, Expression, ExpressionNode, TokenType


class SingleTickerAlphaEnv:
    """
    Environment for training a single-ticker specialized agent.
    
    The agent sees only features from its assigned ticker,
    avoiding state pollution from other tickers.
    
    Reward is computed via ROLLING Sharpe over the entire valid
    date range, exactly like MultiTickerAlphaEnvV2 but for one ticker.
    """

    # Time windows for feature computation
    SHORT_WINDOW = 5
    MEDIUM_WINDOW = 20
    LONG_WINDOW = 60

    # Data requirements
    WARMUP_PERIOD = 120          # Days needed before we can compute features + rolling norm
    MIN_EVAL_DAYS = 200          # Minimum days in the eval period
    
    # Expression constraints
    MIN_EXPRESSION_DEPTH = 3     # Minimum steps before expression can complete
    
    # Rolling normalization
    ROLLING_NORM_WINDOW = 60     # Window for rolling rank normalization
    ROLLING_NORM_MIN = 30        # Minimum periods for rolling rank
    
    # Window constraints for TS operators
    MIN_TS_WINDOW = 2
    WINDOW_OPTIONS = [2, 3, 5, 10, 20, 50, 120]

    def __init__(
        self,
        data: pd.DataFrame,
        ticker: str,
        max_steps: int = 20,
    ):
        """
        Initialize environment for a single ticker.
        
        Parameters:
            data: Full DataFrame with all tickers
            ticker: The specific ticker this env is for
            max_steps: Maximum steps per episode
        """
        self.ticker = ticker
        self.max_steps = max_steps
        self.window_options = self.WINDOW_OPTIONS

        # ----- Extract ticker data -----
        ticker_df = data[data['Ticker'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
        
        if len(ticker_df) < self.WARMUP_PERIOD + self.MIN_EVAL_DAYS:
            raise ValueError(
                f"Ticker {ticker} has only {len(ticker_df)} days, "
                f"need at least {self.WARMUP_PERIOD + self.MIN_EVAL_DAYS}"
            )
        
        self.data = ticker_df
        self.dates = ticker_df['Date'].tolist()
        self.date_to_idx = {date: idx for idx, date in enumerate(self.dates)}
        print(f"[{ticker}] Loaded {len(self.data)} days of data")

        # ----- Precompute features -----
        self._precompute_features()
        self._precompute_returns()
        self._setup_eval_range()

        # ----- Initialize DSL -----
        self.dsl = DSL()
        self._build_action_space()

        # ----- State dimensions -----
        self.short_feature_size = 5   # momentum, vol, mean_rev, vol_ratio, ret_sign
        self.medium_feature_size = 5
        self.long_feature_size = 4    # No ret_sign for long-term
        self.stack_feature_size = 10
        
        self.market_feature_size = (
            self.short_feature_size + 
            self.medium_feature_size + 
            self.long_feature_size
        )
        self.state_size = self.market_feature_size + self.stack_feature_size

        # ----- Episode state -----
        self.stack: List[ExpressionNode] = []
        self.current_step = 0
        self.current_window = 5
        self.episode_date = None
        self.episode_date_idx = None
        self.action_counts = {'operand': 0, 'constant': 0, 'unary': 0, 'binary': 0, 'ts': 0}

    # =========================================================================
    # PRECOMPUTATION
    # =========================================================================

    def _precompute_features(self):
        """Precompute features for all three time windows."""
        self.features = {
            'short': self._compute_features_for_window(self.SHORT_WINDOW),
            'medium': self._compute_features_for_window(self.MEDIUM_WINDOW),
            'long': self._compute_features_for_window(self.LONG_WINDOW),
        }

    def _compute_features_for_window(self, window: int) -> pd.DataFrame:
        """Compute ticker-specific features for a given window."""
        df = self.data
        features = pd.DataFrame(index=df.index)
        
        close = df['Close'].astype(float)
        volume = df['Volume'].astype(float)
        returns = close.pct_change()
        
        # Momentum: cumulative returns over window
        features['momentum'] = returns.rolling(window, min_periods=window).sum()
        
        # Volatility: annualized rolling std of returns
        features['volatility'] = returns.rolling(window, min_periods=window).std() * np.sqrt(252)
        
        # Mean reversion: z-score distance from rolling mean
        rolling_mean = close.rolling(window, min_periods=window).mean()
        rolling_std = close.rolling(window, min_periods=window).std()
        features['mean_reversion'] = ((close - rolling_mean) / (rolling_std + 1e-8)).clip(-13, 13)
        
        # Volume ratio: log of current vs average
        avg_volume = volume.rolling(window, min_periods=window).mean()
        features['volume_ratio'] = np.log1p(volume / (avg_volume + 1))
        
        # Return sign: fraction of positive returns in window
        features['return_sign'] = (returns > 0).rolling(window, min_periods=window).mean()
        
        return features

    def _precompute_returns(self):
        """Precompute forward returns (open-to-close next day, matching V2)."""
        df = self.data
        self.forward_returns = (
            (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
        )
        print(f"[{self.ticker}] Precomputed forward returns")

    def _setup_eval_range(self):
        """Setup evaluation range for rolling Sharpe (matching V2 logic)."""
        n = len(self.data)
        
        # Eval starts after warmup, ends near the end (need T+1 return)
        self.eval_start_idx = self.WARMUP_PERIOD
        self.eval_end_idx = n - 2  # -2 because we need T+1 return
        
        n_eval_days = self.eval_end_idx - self.eval_start_idx
        if n_eval_days < self.MIN_EVAL_DAYS:
            raise ValueError(
                f"Not enough eval days for ticker {self.ticker}: "
                f"{n_eval_days} < {self.MIN_EVAL_DAYS}"
            )
        
        self.eval_start_date = self.dates[self.eval_start_idx]
        self.eval_end_date = self.dates[self.eval_end_idx]
        
        # Valid dates for episode sampling (state computation)
        # Sample from the eval range so the state reflects conditions
        # that actually matter for the rolling evaluation
        self.valid_start_indices = list(range(self.eval_start_idx, self.eval_end_idx))
        self.valid_start_dates = [self.dates[i] for i in self.valid_start_indices]
        
        print(f"[{self.ticker}] Eval range: {self.eval_start_date} to {self.eval_end_date} ({n_eval_days} days)")
        print(f"[{self.ticker}] {len(self.valid_start_dates)} valid start dates")

    # =========================================================================
    # ACTION SPACE
    # =========================================================================

    def _build_action_space(self):
        """Build action space with DSL tokens and window selections."""
        self.actions = []
        self.action_to_token = {}
        self.action_to_window = {}
        self.action_is_alpha_primitive = {}
        
        action_idx = 0
        all_tokens = self.dsl.get_all_tokens()
        
        # Flatten nested dicts from get_all_tokens()
        # Handle both flat and nested structures
        for category_key, category_value in all_tokens.items():
            if isinstance(category_value, dict):
                is_alpha = (category_key == 'alpha_primitives')
                for name, token in category_value.items():
                    self.actions.append(name)
                    self.action_to_token[action_idx] = token
                    self.action_is_alpha_primitive[action_idx] = is_alpha
                    action_idx += 1
            else:
                self.actions.append(category_key)
                self.action_to_token[action_idx] = category_value
                self.action_is_alpha_primitive[action_idx] = False
                action_idx += 1
        
        # Add window selection actions
        for w in self.window_options:
            action_name = f"window_{w}"
            self.actions.append(action_name)
            self.action_to_window[action_idx] = w
            self.action_is_alpha_primitive[action_idx] = False
            action_idx += 1
        
        self.n_tokens = action_idx - len(self.window_options)
        self.n_windows = len(self.window_options)
        
        n_primitives = sum(1 for v in self.action_is_alpha_primitive.values() if v)
        print(f"[{self.ticker}] Action space: {len(self.actions)} total ({n_primitives} alpha primitives)")

    @property
    def action_space_size(self) -> int:
        return len(self.actions)

    @property
    def observation_space_size(self) -> int:
        return self.state_size

    # =========================================================================
    # EPISODE LIFECYCLE
    # =========================================================================

    def reset(self) -> np.ndarray:
        """
        Reset environment for a new episode.
        
        Samples a random date for the STATE (market regime context).
        Reward is always computed over the FULL rolling window regardless.
        """
        # Sample random date for state context
        idx = np.random.randint(len(self.valid_start_indices))
        self.episode_date_idx = self.valid_start_indices[idx]
        self.episode_date = self.dates[self.episode_date_idx]
        
        # Reset episode state
        self.stack = []
        self.current_step = 0
        self.current_window = 5
        self.action_counts = {'operand': 0, 'constant': 0, 'unary': 0, 'binary': 0, 'ts': 0}
        
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        market_features = self._compute_market_features()
        stack_features = self._get_stack_state()
        return np.concatenate([market_features, stack_features])

    def _compute_market_features(self) -> np.ndarray:
        """Compute market features for the current episode date."""
        features = []
        date_idx = self.episode_date_idx
        
        # Short-term features (5)
        short_df = self.features['short']
        if date_idx < len(short_df):
            row = short_df.iloc[date_idx]
            features.extend([
                self._safe_value(row['momentum']) * 10,    # Scale to match V2
                self._safe_value(row['volatility']),
                self._safe_value(row['mean_reversion']),
                self._safe_value(row['volume_ratio']),
                self._safe_value(row['return_sign'], default=0.5),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.5])
        
        # Medium-term features (5)
        medium_df = self.features['medium']
        if date_idx < len(medium_df):
            row = medium_df.iloc[date_idx]
            features.extend([
                self._safe_value(row['momentum']) * 10,
                self._safe_value(row['volatility']),
                self._safe_value(row['mean_reversion']),
                self._safe_value(row['volume_ratio']),
                self._safe_value(row['return_sign'], default=0.5),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.5])
        
        # Long-term features (4) - no return_sign
        long_df = self.features['long']
        if date_idx < len(long_df):
            row = long_df.iloc[date_idx]
            features.extend([
                self._safe_value(row['momentum']) * 10,
                self._safe_value(row['volatility']),
                self._safe_value(row['mean_reversion']),
                self._safe_value(row['volume_ratio']),
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)

    def _safe_value(self, val, default=0.0) -> float:
        """Safely convert value, handling NaN/inf."""
        if pd.isna(val) or np.isinf(val):
            return default
        return float(np.clip(val, -20, 20))

    def _get_stack_state(self) -> np.ndarray:
        """
        Get stack-related features (10 features).
        Includes alpha primitive tracking like V2.
        """
        stack_size = len(self.stack)
        
        # Expression depth (max depth of any node on stack)
        max_depth = 0
        n_alpha_primitives = 0
        if stack_size > 0:
            max_depth = max(node.depth() for node in self.stack)
            for node in self.stack:
                if hasattr(node, 'is_alpha_primitive') and node.is_alpha_primitive():
                    n_alpha_primitives += 1

        return np.array([
            len(self.stack),                                # 0: stack size
            self.current_step,                              # 1: current step
            self.current_window / 50.0,                     # 2: normalized window
            self.action_counts['operand'],                  # 3: operand count
            self.action_counts['constant'],                 # 4: constant count
            self.action_counts['unary'],                    # 5: unary op count
            self.action_counts['binary'],                   # 6: binary op count
            self.action_counts['ts'],                       # 7: ts op count
            n_alpha_primitives,                             # 8: alpha primitive count
            self.max_steps - self.current_step,             # 9: steps remaining
        ], dtype=np.float32)

    def _can_complete(self) -> bool:
        """Check if expression can be completed."""
        return (
            len(self.stack) == 1 and 
            self.current_step >= self.MIN_EXPRESSION_DEPTH
        )

    # =========================================================================
    # ACTION MASKING
    # =========================================================================

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions."""
        valid = self._get_valid_actions()
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        mask[valid] = 1.0
        return mask

    def _get_valid_actions(self) -> List[int]:
        """Get list of valid action indices (matching V2 logic)."""
        valid = []
        stack_size = len(self.stack)
        
        for action_idx in range(self.action_space_size):
            # Window selection always valid
            if action_idx in self.action_to_window:
                valid.append(action_idx)
                continue
            
            # Alpha primitives act as leaf nodes (push to stack)
            if self.action_is_alpha_primitive.get(action_idx, False):
                valid.append(action_idx)
                continue
            
            token = self.action_to_token.get(action_idx)
            if token is None:
                continue
            
            token_type = token.token_type
            
            if token_type in (TokenType.OPERAND, TokenType.CONSTANT):
                valid.append(action_idx)
            elif token_type in (TokenType.UNARY_OP, TokenType.TS_OP):
                if stack_size >= 1:
                    valid.append(action_idx)
            elif token_type == TokenType.BINARY_OP:
                if stack_size >= 2:
                    valid.append(action_idx)
        
        return valid

    # =========================================================================
    # STEP
    # =========================================================================

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (state, reward, done, info)."""
        info = {'expression': None, 'sharpe': None}
        
        # ----- Window selection (does NOT count as a step) -----
        if action in self.action_to_window:
            self.current_window = self.action_to_window[action]
            return self._get_state(), 0.0, False, {'window_set': self.current_window}
        
        # ----- Token action (counts as a step) -----
        self.current_step += 1
        token = self.action_to_token[action]
        
        # Alpha primitive: create leaf node with current window
        if self.action_is_alpha_primitive.get(action, False):
            node = ExpressionNode(token, children=[], window=self.current_window)
            self.stack.append(node)
        else:
            token_type = token.token_type
            
            if token_type in (TokenType.OPERAND, TokenType.CONSTANT):
                if token_type == TokenType.OPERAND:
                    self.action_counts['operand'] += 1
                else:
                    self.action_counts['constant'] += 1
                node = ExpressionNode(token)
                self.stack.append(node)
            
            elif token_type == TokenType.UNARY_OP:
                self.action_counts['unary'] += 1
                child = self.stack.pop()
                node = ExpressionNode(token, children=[child])
                self.stack.append(node)
            
            elif token_type == TokenType.BINARY_OP:
                self.action_counts['binary'] += 1
                right = self.stack.pop()
                left = self.stack.pop()
                node = ExpressionNode(token, children=[left, right])
                self.stack.append(node)
            
            elif token_type == TokenType.TS_OP:
                self.action_counts['ts'] += 1
                child = self.stack.pop()
                node = ExpressionNode(token, children=[child], window=self.current_window)
                self.stack.append(node)
        
        # ----- Check termination -----
        done = False
        reward = 0.0
        
        can_complete = (
            len(self.stack) == 1 
            and self.current_step >= self.MIN_EXPRESSION_DEPTH
        )
        
        if can_complete:
            done = True
            expr = Expression(self.stack[0])
            expr_str = expr.to_string()
            info['expression'] = expr_str
            
            # Structural check first (cheap)
            structural_check = self._compute_structural_check(expr)
            
            if structural_check < -1.0:
                reward = -1.0
                info['sharpe'] = 0.0
                info['skipped_eval'] = True
            else:
                # ROLLING SHARPE (expensive but accurate)
                sharpe = self._compute_rolling_sharpe(expr)
                info['sharpe'] = sharpe
                reward = sharpe
        
        elif self.current_step >= self.max_steps:
            done = True
            reward = -1.0
            info['timeout'] = True
        
        return self._get_state(), reward, done, info

    # =========================================================================
    # REWARD: ROLLING SHARPE (matching V2 logic)
    # =========================================================================

    def _compute_rolling_sharpe(self, expr: Expression) -> float:
        """
        Compute Sharpe ratio using rolling window over ALL valid dates.
        
        Same approach as MultiTickerAlphaEnvV2._compute_single_ticker_rolling_sharpe
        but for this single ticker.
        
        Steps:
            1. Evaluate expression on FULL ticker history
            2. Rolling rank normalization (60-day window) → alpha in [-1, 1]
            3. Strategy returns = alpha[T] * forward_return[T]
            4. Sharpe = mean / std * sqrt(252) over eval period
        """
        try:
            # 1. Evaluate expression on FULL data
            alpha_raw = expr.evaluate(self.data)
            
            # 2. Quality check
            alpha_clean = alpha_raw.replace(0, np.nan).dropna()
            if len(alpha_clean) < self.WARMUP_PERIOD or alpha_clean.std() < 1e-8:
                return 0.0
            
            # 3. Rolling rank normalization (same as V2)
            alpha_rank = alpha_raw.rolling(
                window=self.ROLLING_NORM_WINDOW,
                min_periods=self.ROLLING_NORM_MIN,
            ).rank(pct=True)
            alpha_normalized = (alpha_rank - 0.5) * 2.0  # Scale to [-1, 1]
            
            # 4. Strategy returns over eval range
            strategy_returns = (
                alpha_normalized.iloc[self.eval_start_idx:self.eval_end_idx]
                * self.forward_returns.iloc[self.eval_start_idx:self.eval_end_idx]
            )
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < self.MIN_EVAL_DAYS:
                return 0.0
            
            # 5. Compute Sharpe
            mean_ret = strategy_returns.mean()
            std_ret = strategy_returns.std()
            
            if std_ret < 1e-10:
                return 0.0
            
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            return float(np.clip(sharpe, -10, 10))
        
        except Exception as e:
            return -1.0

    def _compute_structural_check(self, expr: Expression) -> float:
        """
        Minimal structural check — ONLY reject truly broken expressions.
        No bonuses, just pass/fail gate (matching V2 approach).
        
        Returns:
            0.0 if expression is valid (proceed to Sharpe eval)
            -2.0 if expression is broken (skip Sharpe eval)
        """
        expr_str = expr.to_string()
        
        # Must use price/volume data OR alpha primitives
        uses_ohlcv = any(op in expr_str for op in ['Open', 'High', 'Low', 'Close', 'Volume'])
        uses_alpha = 'alpha_' in expr_str
        
        if not (uses_ohlcv or uses_alpha):
            return -2.0  # Broken: doesn't use any market data
        
        return 0.0