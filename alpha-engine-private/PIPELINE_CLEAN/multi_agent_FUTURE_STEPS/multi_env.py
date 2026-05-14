"""
Batch Training for Alpha Expression Synthesis

Trains on MULTIPLE tickers simultaneously to find robust alpha expressions
that generalize across different stocks.

Key insight: An alpha that works on just AAPL might be noise.
             An alpha that works on 20 diverse stocks is likely real signal.

Training approach:
    Each episode, the agent builds ONE expression.
    That expression is evaluated on ALL tickers in the batch.
    Reward = average Sharpe ratio across all tickers.
    
This forces the agent to find UNIVERSAL patterns, not stock-specific quirks.
"""

import os
import sys
import numpy as np
import pandas as pd
from collections import deque
from typing import List, Dict, Tuple
import pickle

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from DSL.dsl import DSL
from DSL.expression import Expression
from DSL.node import ExpressionNode
from DSL.utils import TokenType
from agent.agentPPO_enhanced import PPOAgentEnhanced
from alpha_trading.alpha_library import AlphaLibrary


# =============================================================================
# CONFIGURATION: Which tickers to train on
# =============================================================================

# Diversified selection across sectors
# You can modify this list based on your preferences
DEFAULT_TRAINING_TICKERS = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'INTC', 'CSCO',
    # Finance
    'JPM', 'BAC', 'GS', 'WFC',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'MRK',
    # Consumer
    'WMT', 'KO', 'PG', 'MCD',
    # Industrial
    'GE', 'CAT', 'BA',
    # Energy
    'XOM', 'CVX',
]


# =============================================================================
# MULTI-TICKER ENVIRONMENT
# =============================================================================

# NOTE: very similar to the AlphaEnv class, clearly the main changes are
# wrt the constructor, the reward function (which is now related to many sharpes)
# and to the step function

class MultiTickerAlphaEnv:
    """
    Environment that evaluates alpha expressions on MULTIPLE tickers.
    
    The agent builds an expression, and it's evaluated on all tickers.
    Reward = average Sharpe ratio across tickers.
    
    This finds alphas that work UNIVERSALLY, not just on one stock.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        max_steps: int = 10,
        window_options: List[int] = [1, 3, 5, 10],
        min_sharpe_tickers: int = None,  # Minimum tickers that must have positive Sharpe
    ):
        """
        Parameters:
            data: Full OHLCV DataFrame with 'Ticker' column
            tickers: List of tickers to train on
            max_steps: Maximum steps per episode
            window_options: Available window sizes for TS ops
            min_sharpe_tickers: If set, penalize if fewer than this many tickers
                                have positive Sharpe (encourages broad applicability)
        """
        self.full_data = data
        self.tickers = tickers
        self.max_steps = max_steps
        self.window_options = window_options
        self.min_sharpe_tickers = min_sharpe_tickers or len(tickers) // 2
        
        # Split data by ticker
        self.ticker_data: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            ticker_df = data[data['Ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('Date').reset_index(drop=True)
            if len(ticker_df) > 100:  # Need enough data
                self.ticker_data[ticker] = ticker_df
        
        self.active_tickers = list(self.ticker_data.keys())
        print(f"Loaded {len(self.active_tickers)} tickers with sufficient data")
        
        if len(self.active_tickers) == 0:
            raise ValueError("No tickers have sufficient data!")
        
        # Initialize DSL
        self.dsl = DSL()
        self._build_action_space()
        
        # Episode state
        self.stack: List[ExpressionNode] = []
        self.current_step = 0
        self.current_window = window_options[0]
    
    def _build_action_space(self):
        """Create the list of available actions."""
        self.actions = []
        self.action_to_token = {}
        
        all_tokens = self.dsl.get_all_tokens()
        
        for category in ['operands', 'constants', 'unary_ops', 'binary_ops', 'ts_ops']:
            for name, token in all_tokens[category].items():
                action_id = len(self.actions)
                self.actions.append(name)
                self.action_to_token[action_id] = token
        
        self.n_actions = len(self.actions)
        self.action_name_to_id = {name: i for i, name in enumerate(self.actions)}
    
    @property
    def action_space_size(self) -> int:
        return self.n_actions
    
    @property
    def state_size(self) -> int:
        return 8
    
    def reset(self) -> np.ndarray:
        """Reset for new episode."""
        self.stack = []
        self.current_step = 0
        self.current_window = self.window_options[0]
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Encode current state as vector."""
        type_counts = {tt: 0 for tt in TokenType}
        for node in self.stack:
            type_counts[node.token.token_type] += 1
        
        state = np.array([
            len(self.stack),
            self.current_step,
            self.current_window,
            type_counts[TokenType.OPERAND],
            type_counts[TokenType.CONSTANT],
            type_counts[TokenType.UNARY_OP],
            type_counts[TokenType.BINARY_OP],
            type_counts[TokenType.TS_OP],
        ], dtype=np.float32)

        return state
    
    def _get_valid_actions(self) -> List[int]:
        """Get list of valid action IDs."""
        valid = []
        stack_size = len(self.stack)
        
        for action_id, token in self.action_to_token.items():
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
        """Get binary mask of valid actions."""
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for action_id in self._get_valid_actions():
            mask[action_id] = 1.0
        return mask
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Take a step."""
        self.current_step += 1
        
        token = self.action_to_token[action]
        token_type = token.token_type
        
        # Apply action
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
        
        # Check if done
        done = False
        reward = 0.0
        info = {'expression': None, 'ticker_sharpes': {}}
        
        if len(self.stack) == 1 and self.current_step >= 2:
            done = True
            expr = Expression(self.stack[0])
            reward, ticker_sharpes = self._compute_batch_reward(expr)
            info['expression'] = expr.to_string()
            info['ticker_sharpes'] = ticker_sharpes
            info['sharpe'] = reward
        elif self.current_step >= self.max_steps:
            done = True
            reward = -1.0
            info['timeout'] = True
        
        return self._get_state(), reward, done, info
    
    def _compute_single_ticker_sharpe(self, expr: Expression, data: pd.DataFrame) -> float:
        """Compute Sharpe ratio for one ticker."""
        try:
            alpha = expr.evaluate(data)
            
            # Reject trivial signals
            alpha_clean = alpha.replace(0, np.nan).dropna()
            if len(alpha_clean) < 30 or alpha_clean.std() < 1e-6:
                return 0.0
            
            # Rank-based normalization
            lookback = 60
            alpha_rank = alpha.rolling(window=lookback, min_periods=20).apply(
                lambda w: pd.Series(w).rank(pct=True).iloc[-1], raw=False
            )
            alpha_normalized = (alpha_rank - 0.5) * 2.0
            
            norm_clean = alpha_normalized.dropna()
            if len(norm_clean) < 30 or norm_clean.std() < 0.05:
                return 0.0
            
            # Forward returns
            returns = (data['Close'].shift(-1) - data['Open'].shift(-1)) / data['Open'].shift(-1)
            
            # Strategy returns
            strategy_returns = alpha_normalized * returns
            strategy_returns = strategy_returns.dropna()
            
            if len(strategy_returns) < 30:
                return 0.0
            
            mean_return = strategy_returns.mean()
            std_return = strategy_returns.std()
            
            if std_return < 1e-10:
                return 0.0
            
            sharpe = mean_return / std_return * np.sqrt(252)
            return float(np.clip(sharpe, -5, 5))
        
        except Exception:
            return -1.0
    
    def _compute_batch_reward(self, expr: Expression) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward across ALL tickers.
        
        Returns:
            (average_sharpe, {ticker: sharpe})
        """
        ticker_sharpes = {}
        
        for ticker, data in self.ticker_data.items():
            sharpe = self._compute_single_ticker_sharpe(expr, data)
            ticker_sharpes[ticker] = sharpe
        
        sharpes = list(ticker_sharpes.values())
        valid_sharpes = [s for s in sharpes if s > -5]  # Exclude complete failures
        
        if len(valid_sharpes) == 0:
            return -1.0, ticker_sharpes
        
        # Average Sharpe
        avg_sharpe = np.mean(valid_sharpes)
        
        # Bonus/penalty for consistency across tickers
        positive_count = sum(1 for s in valid_sharpes if s > 0)
        
        # If fewer than half the tickers have positive Sharpe, penalize
        if positive_count < self.min_sharpe_tickers:
            avg_sharpe *= 0.5  # Penalize inconsistent alphas
        
        # Bonus for very consistent alphas (most tickers positive)
        if positive_count >= len(valid_sharpes) * 0.8:
            avg_sharpe *= 1.2  # Reward consistent alphas
        
        # Penalize if expression doesn't use market data
        expr_str = expr.to_string()
        uses_market_data = any(
            op in expr_str for op in ['Open', 'High', 'Low', 'Close', 'Volume']
        )
        if not uses_market_data:
            avg_sharpe *= 0.1
        
        return float(np.clip(avg_sharpe, -3, 3)), ticker_sharpes


