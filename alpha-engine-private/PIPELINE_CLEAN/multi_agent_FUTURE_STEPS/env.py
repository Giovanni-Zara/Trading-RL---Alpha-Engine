"""
RL Environment for Alpha Expression Synthesis

The agent builds alpha expressions step-by-step by selecting tokens.
Uses a stack-based approach (like Reverse Polish Notation).

Example episode:
    1. Agent selects 'close'  → stack: [close]
    2. Agent selects 'open'   → stack: [close, open]
    3. Agent selects 'sub'    → stack: [sub(close, open)]
    4. Episode ends, reward = Sharpe ratio of the alpha
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from enum import IntEnum

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from DSL.dsl import DSL
from DSL.node import ExpressionNode
from DSL.expression import Expression
from DSL.utils import TokenType


class AlphaEnv:
    """
    RL Environment for building alpha expressions.
    
    State: Current stack of expression nodes (encoded as vector)
    Action: Choose a token from the DSL
    Reward: Sharpe ratio when expression is complete, 0 otherwise
    Done: When stack has exactly 1 complete expression
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        max_steps: int = 10,
        window_options: List[int] = [1, 3, 5, 10]
    ):
        """
        Initialize the environment.
        
        Parameters:
            data: OHLCV DataFrame for evaluating alphas
            max_steps: Maximum steps before forcing termination
            window_options: Available window sizes for time-series ops
        """
        self.data = data
        self.max_steps = max_steps
        self.window_options = window_options
        
        # Initialize DSL
        self.dsl = DSL()
        
        # Build action list
        # Actions = all tokens + window options for TS operators
        self._build_action_space()
        
        # Episode state
        self.stack: List[ExpressionNode] = []
        self.current_step = 0
        self.current_window = window_options[0]  # Default window
    
    def _build_action_space(self):
        """
        Create the list of available actions.
        
        Actions are organized as:
        - Operands: close, open, high, low, volume
        - Constants: -1, 0, 0.5, 1, 2
        - Unary ops: neg, abs, log, sign
        - Binary ops: add, sub, mul, div, max, min
        - TS ops: delay, delta, ts_mean, ts_std, ts_max, ts_min, ts_rank
        """
        self.actions = []
        self.action_to_token = {}
        
        # Add all tokens from DSL
        all_tokens = self.dsl.get_all_tokens()
        
        for category in ['operands', 'constants', 'unary_ops', 'binary_ops', 'ts_ops']:
            for name, token in all_tokens[category].items():
                action_id = len(self.actions)
                self.actions.append(name)
                self.action_to_token[action_id] = token
        
        self.n_actions = len(self.actions)
        
        # For easy lookup
        self.action_name_to_id = {name: i for i, name in enumerate(self.actions)}
    
    @property
    def action_space_size(self) -> int:
        """Number of possible actions."""
        return self.n_actions
    
    @property
    def state_size(self) -> int:
        """Size of state vector."""
        return 8  # Matches _get_state output
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial state vector
        """
        self.stack = []
        self.current_step = 0
        self.current_window = self.window_options[0]
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Encode current state as a vector.
        
        Simple encoding:
        - Stack size
        - Current step
        - Current window
        - Which token types are on stack (counts)
        
        This is a simple state representation. Can be improved later.
        """
        # Count token types on stack
        type_counts = {tt: 0 for tt in TokenType}
        for node in self.stack:
            type_counts[node.token.token_type] += 1
        
        state = np.array([
            len(self.stack),                    # Stack size
            self.current_step,                  # Current step
            self.current_window,                # Current window
            type_counts[TokenType.OPERAND],     # Operand count
            type_counts[TokenType.CONSTANT],    # Constant count
            type_counts[TokenType.UNARY_OP],    # Unary op count
            type_counts[TokenType.BINARY_OP],   # Binary op count
            type_counts[TokenType.TS_OP],       # TS op count
        ], dtype=np.float32)
        
        return state
    

    
    def _get_valid_actions(self) -> List[int]:
        """
        Get list of valid action IDs given current stack.
        
        Rules:
        - Operands/Constants: Always valid
        - Unary ops: Need at least 1 item on stack
        - Binary ops: Need at least 2 items on stack
        - TS ops: Need at least 1 item on stack
        """
        valid = []
        stack_size = len(self.stack)
        
        for action_id, token in self.action_to_token.items():
            token_type = token.token_type
            
            # Operands and constants: always valid
            if token_type in (TokenType.OPERAND, TokenType.CONSTANT):
                valid.append(action_id)
            
            # Unary and TS ops: need 1 item
            elif token_type in (TokenType.UNARY_OP, TokenType.TS_OP):
                if stack_size >= 1:
                    valid.append(action_id)
            
            # Binary ops: need 2 items
            elif token_type == TokenType.BINARY_OP:
                if stack_size >= 2:
                    valid.append(action_id)
        
        return valid
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get binary mask of valid actions (for masked action selection).
        
        Returns:
            Array of 0s and 1s, where 1 = valid action
        """
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for action_id in self._get_valid_actions():
            mask[action_id] = 1.0
        return mask
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment.
        
        Parameters:
            action: Action ID (index into self.actions)
            
        Returns:
            state: New state vector
            reward: Reward (Sharpe ratio if done, else 0)
            done: Whether episode is finished
            info: Additional information
        """
        self.current_step += 1
        
        # Get the token for this action
        token = self.action_to_token[action]
        token_type = token.token_type
        
        # ----- Apply action based on token type -----
        
        if token_type in (TokenType.OPERAND, TokenType.CONSTANT):
            # Push operand/constant onto stack
            node = ExpressionNode(token)
            self.stack.append(node)
        
        elif token_type == TokenType.UNARY_OP:
            # Pop 1, apply unary op, push result
            child = self.stack.pop()
            node = ExpressionNode(token, children=[child])
            self.stack.append(node)
        
        elif token_type == TokenType.BINARY_OP:
            # Pop 2, apply binary op, push result
            right = self.stack.pop()
            left = self.stack.pop()
            node = ExpressionNode(token, children=[left, right])
            self.stack.append(node)
        
        elif token_type == TokenType.TS_OP:
            # Pop 1, apply TS op with window, push result
            child = self.stack.pop()
            node = ExpressionNode(token, children=[child], window=self.current_window)
            self.stack.append(node)
        
        # ----- Check if done -----
        done = False
        reward = 0.0
        info = {}
        
        # Done if: stack has exactly 1 item OR max steps reached
        if len(self.stack) == 1 and self.current_step >= 2:
            # We have a complete expression
            done = True
            expr = Expression(self.stack[0])
            reward = self._compute_reward(expr)
            info['expression'] = expr.to_string()
            info['sharpe'] = reward
        
        elif self.current_step >= self.max_steps:
            # Max steps reached without valid expression
            done = True
            reward = -1.0  # Penalty
            info['expression'] = None
            info['timeout'] = True
        
        return self._get_state(), reward, done, info
    

# ============================================================
# env.py — Replace _compute_reward entirely
# ============================================================

    def _compute_reward(self, expr: Expression) -> float:
        """
        Compute reward based on the alpha expression.

        Key design choices:
        - Rolling normalization (no future leakage)
        - Rank-based alpha signal (removes scale/trend artifacts)
        - Novelty penalty via expression cache (managed externally)
        - Minimum complexity requirement
        """
        try:
            # Evaluate alpha on data
            alpha = expr.evaluate(self.data)

            # =============================================================
            # REJECT trivial signals
            # =============================================================
            alpha_clean = alpha.replace(0, np.nan).dropna()
            if len(alpha_clean) < 30 or alpha_clean.std() < 1e-6:
                return 0.0

            # =============================================================
            # RANK-BASED normalization (instead of z-score)
            # =============================================================
            # Rolling z-score on a monotonic signal (like Volume or Close)
            # creates a trend-following signal that trivially profits on
            # trending stocks. Rank-based normalization removes this bias:
            # the signal is the percentile rank within a rolling window,
            # centered to [-1, 1]. A monotonic series gets rank ~0.5
            # (middle of window) most of the time → near-zero signal.
            lookback = 60

            # Rolling percentile rank: where does today's value sit
            # within the last `lookback` days? Returns 0..1
            alpha_rank = alpha.rolling(window=lookback, min_periods=20).apply(
                lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
            )
            # Center to [-1, 1]
            alpha_normalized = (alpha_rank - 0.5) * 2.0

            # =============================================================
            # Reject near-constant normalized signals
            # =============================================================
            norm_clean = alpha_normalized.dropna()
            if len(norm_clean) < 30:
                return 0.0
            if norm_clean.std() < 0.05:
                return 0.0  # Signal barely varies after rank normalization

            # =============================================================
            # EXPLICIT TIMING MODEL
            # =============================================================
            returns = (self.data['Close'].shift(-1) - self.data['Open'].shift(-1)) / self.data['Open'].shift(-1)

            # =============================================================
            # COMPUTE STRATEGY RETURNS
            # =============================================================
            strategy_returns = alpha_normalized * returns
            strategy_returns = strategy_returns.dropna()

            if len(strategy_returns) < 30:
                return 0.0

            # Sharpe ratio (annualized)
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()

            if std_return < 1e-10:
                return 0.0

            sharpe = mean_return / std_return * np.sqrt(252)
            sharpe = np.clip(sharpe, -5, 5)

            # =============================================================
            # Penalize expressions without market data
            # =============================================================
            expr_str = expr.to_string()
            uses_market_data = any(
                op in expr_str for op in ['Open', 'High', 'Low', 'Close', 'Volume']
            )
            if not uses_market_data:
                sharpe *= 0.1

            return float(sharpe)

        except Exception as e:
            return -1.0

    # def _compute_reward(self, expr: Expression) -> float:
    #     """
    #     Compute reward based on the alpha expression.
        
    #     POINT-IN-TIME CORRECT:
    #     - Alpha normalization uses only PAST data (rolling window)
    #     - Returns are computed from NEXT period (as intended)
    #     - No future information leaks into the alpha signal
    #     """
    #     try:
    #         # Evaluate alpha on data
    #         alpha = expr.evaluate(self.data)
    #         if alpha.std() < 1e-6:
    #             return 0.0 #if signal is constant -> no info
    #         # =====================================================
    #         # FIX 1: ROLLING NORMALIZATION (no future leakage)
    #         # =====================================================
    #         # Instead of global mean/std, we use expanding or rolling window
    #         # expanding() uses all data UP TO curr. point (NO future)
            
    #         lookback = 60  # Normalization lookback period
            
    #         # Option A: Expanding window (uses ALL past data)
    #         # alpha_mean = alpha.expanding(min_periods=lookback).mean()
    #         # alpha_std = alpha.expanding(min_periods=lookback).std()
            
    #         # Option B: Rolling window (uses last N days only)
    #         alpha_mean = alpha.rolling(window=lookback, min_periods=20).mean()
    #         alpha_std = alpha.rolling(window=lookback, min_periods=20).std()
            
    #         # Z-score normalization using only past data
    #         alpha_normalized = (alpha - alpha_mean) / (alpha_std + 1e-10)
            
    #         # Clip extreme values
    #         alpha_normalized = alpha_normalized.clip(-3, 3)
            
    #         # =====================================================
    #         # FIX 2: EXPLICIT TIMING MODEL
    #         # =====================================================
    #         # We observe data at end of day t, trade at OPEN of day t+1,
    #         # and measure return from open t+1 to close t+1
    #         # 
    #         # More conservative: trade at close t+1, measure return t+1 to t+2
            
    #         # Previous model (slightly optimistic): 
    #         # Signal on day t → Return from Close t to Close t+1
    #         # returns = self.data['Close'].pct_change().shift(-1)
            
    #         # Alternative (more realistic):
    #         # Signal on day t → Return from Open t+1 to Close t+1
    #         # a.k.a. we assume we get a signal at time t after closing t
    #         # we buy (sell) at open t+1 and do the reverse option at the close t+1
    #         returns = (self.data['Close'].shift(-1) - self.data['Open'].shift(-1)) / self.data['Open'].shift(-1)
            
    #         # =====================================================
    #         # COMPUTE STRATEGY RETURNS
    #         # =====================================================
    #         strategy_returns = alpha_normalized * returns
            
    #         # Remove NaN (first `lookback` days won't have valid normalization)
    #         strategy_returns = strategy_returns.dropna()
            
    #         if len(strategy_returns) < 30:  # Need enough data for meaningful Sharpe
    #             return 0.0
            
    #         # Sharpe ratio (annualized)
    #         mean_return = strategy_returns.mean()
    #         std_return = strategy_returns.std()
            
    #         if std_return < 1e-10:
    #             return 0.0
            
    #         sharpe = mean_return / std_return * np.sqrt(252)
    #         sharpe = np.clip(sharpe, -5, 5)
            
    #         complexity = expr.size()
    #         if complexity <= 2: #too simple, agent was stuck on spamming delay on 1 token
    #             sharpe *= 0.5 #penalize trivial stuff

    #         return float(sharpe)
        
    #     except Exception as e:
    #         return -1.0

    
    def set_window(self, window: int):
        """Set the window size for time-series operators."""
        if window in self.window_options:
            self.current_window = window
    
    def render(self):
        """Print current state (for debugging)."""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Stack size: {len(self.stack)}")
        print(f"Current window: {self.current_window}")
        if self.stack:
            print("Stack contents:")
            for i, node in enumerate(self.stack):
                print(f"  [{i}] {node.to_string()}")
        print(f"Valid actions: {[self.actions[a] for a in self._get_valid_actions()]}")
        print()

