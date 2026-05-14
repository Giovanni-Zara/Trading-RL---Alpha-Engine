"""
Alpha Library: Save, Load, and Manage Discovered Alpha Expressions

This module provides:
1. AlphaLibrary: Store and manage multiple alphas
2. Save/Load alphas to JSON (human-readable) or pickle (full objects)
3. Evaluate alphas on new data
4. Rank and filter alphas by performance
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Import DSL components
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from DSL import DSL
from DSL import ExpressionNode
from DSL import Expression
from DSL import TokenType

from alpha_record import AlphaRecord


# =============================================================================
# ALPHA LIBRARY: Collection of alphas
# =============================================================================

class AlphaLibrary:
    """
    Manage a collection of alpha expressions.
    
    Features:
    - Add alphas from training results
    - Save/load to JSON file
    - Filter by performance metrics
    - Evaluate alphas on new data
    - Combine multiple alphas into ensemble
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize an alpha library.
        
        Parameters:
            name: Library name (used for file naming)
        """
        self.name = name
        self.alphas: Dict[str, AlphaRecord] = {}
        self.dsl = DSL()
        self._next_id = 1
    
    # -------------------------------------------------------------------------
    # Adding alphas
    # -------------------------------------------------------------------------
    
    def add_alpha(
        self,
        expression: str,
        sharpe_ratio: float,
        name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Add a new alpha to the library.
        
        Parameters:
            expression: The alpha formula string
            sharpe_ratio: Sharpe ratio from training
            name: Optional custom name (auto-generated if not provided)
            **kwargs: Additional metrics (mean_return, std_return, etc.)
            
        Returns:
            The name/ID assigned to this alpha
        """
        # Generate name if not provided
        if name is None:
            name = f"alpha_{self._next_id:04d}"
            self._next_id += 1
        
        # Create record
        record = AlphaRecord(
            name=name,
            expression=expression,
            sharpe_ratio=sharpe_ratio,
            created_at=datetime.now().isoformat(),
            **kwargs
        )
        
        self.alphas[name] = record
        return name
    
    def add_from_training_history(
        self,
        history: dict[str, list],
        top_n: int = 10,
        min_sharpe: float = 0.0,
        ticker: str = "",
        period: str = ""
    ) -> List[str]:
        """
        Add the best alphas from a training run.
        
        Parameters:
            history: Training history dict with 'expressions' and 'rewards'
            top_n: How many top alphas to keep
            min_sharpe: Minimum Sharpe ratio threshold
            ticker: Training ticker (for metadata)
            period: Training period (for metadata)
            
        Returns:
            List of names of added alphas
        """
        # Get unique expressions with their best reward
        expr_rewards = {}
        for expr, reward in zip(history['expressions'], history['rewards']):
            if expr and expr != 'None' and reward > min_sharpe:
                if expr not in expr_rewards or reward > expr_rewards[expr]:
                    expr_rewards[expr] = reward
        
        # Sort by reward and take top N
        sorted_exprs = sorted(expr_rewards.items(), key=lambda x: x[1], reverse=True)
        top_exprs = sorted_exprs[:top_n]
        
        # Add each to library
        added = []
        for expr, sharpe in top_exprs:
            name = self.add_alpha(
                expression=expr,
                sharpe_ratio=sharpe,
                training_ticker=ticker,
                training_period=period
            )
            added.append(name)
        
        print(f"Added {len(added)} alphas to library")
        return added
    
    # -------------------------------------------------------------------------
    # Querying alphas
    # -------------------------------------------------------------------------
    
    def get_alpha(self, name: str) -> Optional[AlphaRecord]:
        """Get a single alpha by name."""
        return self.alphas.get(name)
    
    def list_alphas(self, sort_by: str = 'sharpe_ratio', ascending: bool = False) -> pd.DataFrame:
        """
        List all alphas as a DataFrame.
        
        Parameters:
            sort_by: Column to sort by
            ascending: Sort order
            
        Returns:
            DataFrame with all alpha records
        """
        if not self.alphas:
            return pd.DataFrame()
        
        df = pd.DataFrame([a.to_dict() for a in self.alphas.values()])
        df = df.sort_values(sort_by, ascending=ascending)
        return df
    
    def filter_alphas(
        self,
        min_sharpe: float = None,
        max_complexity: int = None,
        max_correlation: float = None
    ) -> List[AlphaRecord]:
        """
        Filter alphas by criteria.
        
        Parameters:
            min_sharpe: Minimum Sharpe ratio
            max_complexity: Maximum expression complexity
            max_correlation: Maximum correlation to existing alphas
            
        Returns:
            List of matching AlphaRecord objects
        """
        result = list(self.alphas.values())
        
        if min_sharpe is not None:
            result = [a for a in result if a.sharpe_ratio >= min_sharpe]
        
        if max_complexity is not None:
            result = [a for a in result if a.complexity <= max_complexity]
        
        return result
    
    # -------------------------------------------------------------------------
    # Save/Load
    # -------------------------------------------------------------------------
    
    def save(self, path: str):
        """
        Save library to JSON file.
        
        Parameters:
            path: File path (should end in .json)
        """
        data = {
            'name': self.name,
            'created_at': datetime.now().isoformat(),
            'alphas': {name: record.to_dict() for name, record in self.alphas.items()},
            '_next_id': self._next_id
        }
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.alphas)} alphas to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AlphaLibrary':
        """
        Load library from JSON file.
        
        Parameters:
            path: File path
            
        Returns:
            AlphaLibrary instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        lib = cls(name=data.get('name', 'loaded'))
        lib._next_id = data.get('_next_id', 1)
        
        for name, record_dict in data.get('alphas', {}).items():
            lib.alphas[name] = AlphaRecord.from_dict(record_dict)
        
        print(f"Loaded {len(lib.alphas)} alphas from {path}")
        return lib
    
    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    
    def parse_expression(self, expr_string: str) -> Optional[Expression]:
        """
        Parse an expression string back into an Expression object.
        
        This reconstructs the expression tree from the string representation.
        
        Parameters:
            expr_string: Formula string like "sub(close, open)"
            
        Returns:
            Expression object that can be evaluated
        """
        try:
            return self._parse_expr_recursive(expr_string)
        except Exception as e:
            print(f"Failed to parse '{expr_string}': {e}")
            return None
    
    def _parse_expr_recursive(self, s: str) -> Expression:
        """Recursively parse expression string into Expression object."""
        s = s.strip()
        
        # Check if it's a function call: name(args)
        if '(' in s and s.endswith(')'):
            # Find the function name
            paren_idx = s.index('(')
            func_name = s[:paren_idx]
            args_str = s[paren_idx + 1:-1]
            
            # Parse arguments (handle nested parentheses)
            args = self._split_args(args_str)
            
            # Get token
            token = self.dsl.get_token(func_name)
            if token is None:
                raise ValueError(f"Unknown token: {func_name}")
            
            # Handle different token types
            if token.token_type == TokenType.TS_OP:
                # Time-series op: last arg is window
                if len(args) >= 2 and args[-1].isdigit():
                    window = int(args[-1])
                    child_expr = self._parse_expr_recursive(args[0])
                    node = ExpressionNode(token, children=[child_expr.root], window=window)
                else:
                    child_expr = self._parse_expr_recursive(args[0])
                    node = ExpressionNode(token, children=[child_expr.root])
                return Expression(node)
            
            elif token.token_type == TokenType.UNARY_OP:
                child_expr = self._parse_expr_recursive(args[0])
                node = ExpressionNode(token, children=[child_expr.root])
                return Expression(node)
            
            elif token.token_type == TokenType.BINARY_OP:
                left_expr = self._parse_expr_recursive(args[0])
                right_expr = self._parse_expr_recursive(args[1])
                node = ExpressionNode(token, children=[left_expr.root, right_expr.root])
                return Expression(node)
        
        else:
            # It's a leaf node (operand or constant)
            token = self.dsl.get_token(s)
            if token is None:
                raise ValueError(f"Unknown token: {s}")
            node = ExpressionNode(token)
            return Expression(node)
    
    def _split_args(self, args_str: str) -> List[str]:
        """Split arguments while respecting nested parentheses."""
        args = []
        current = ""
        depth = 0
        
        for char in args_str:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            args.append(current.strip())
        
        return args
    
    def evaluate_alpha(
        self,
        name: str,
        data: pd.DataFrame,
        normalize: bool = True
    ) -> pd.Series:
        """
        Evaluate a single alpha on data.
        
        Parameters:
            name: Alpha name in library
            data: OHLCV DataFrame
            normalize: Whether to z-score normalize the output
            
        Returns:
            Alpha signal as pd.Series
        """
        record = self.get_alpha(name)
        if record is None:
            raise ValueError(f"Alpha '{name}' not found in library")
        
        expr = self.parse_expression(record.expression)
        if expr is None:
            raise ValueError(f"Could not parse expression: {record.expression}")
        
        # Evaluate
        alpha = expr.evaluate(data)
        
        # Normalize using rolling window (no future leakage)
        if normalize:
            lookback = 60
            alpha_mean = alpha.rolling(window=lookback, min_periods=20).mean()
            alpha_std = alpha.rolling(window=lookback, min_periods=20).std()
            alpha = (alpha - alpha_mean) / (alpha_std + 1e-10)
            alpha = alpha.clip(-3, 3)
        
        return alpha
    
    def evaluate_all(
        self,
        data: pd.DataFrame,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate all alphas on data.
        
        Parameters:
            data: OHLCV DataFrame
            normalize: Whether to normalize
            
        Returns:
            DataFrame with columns = alpha names, rows = dates
        """
        results = {}
        
        for name in self.alphas.keys():
            try:
                results[name] = self.evaluate_alpha(name, data, normalize)
            except Exception as e:
                print(f"Failed to evaluate {name}: {e}")
        
        return pd.DataFrame(results, index=data.index)


# =============================================================================
# HELPER: Compute additional metrics
# =============================================================================

def compute_alpha_metrics(
    alpha: pd.Series,
    data: pd.DataFrame,
    normalize: bool = True
) -> dict:
    """
    Compute comprehensive metrics for an alpha signal.
    
    Parameters:
        alpha: Alpha signal (raw or normalized)
        data: OHLCV data for computing returns
        normalize: Whether to normalize alpha first
        
    Returns:
        Dictionary of metrics
    """
    if normalize:
        lookback = 60
        alpha_mean = alpha.rolling(window=lookback, min_periods=20).mean()
        alpha_std = alpha.rolling(window=lookback, min_periods=20).std()
        alpha = (alpha - alpha_mean) / (alpha_std + 1e-10)
        alpha = alpha.clip(-3, 3)
    
    # Compute returns (Open-to-Close next day)
    returns = (data['Close'].shift(-1) - data['Open'].shift(-1)) / data['Open'].shift(-1)
    
    # Strategy returns
    strategy_returns = alpha * returns
    strategy_returns = strategy_returns.dropna()
    
    if len(strategy_returns) < 30:
        return {'error': 'Insufficient data'}
    
    # Metrics
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
    
    # Cumulative returns for drawdown
    cum_returns = (1 + strategy_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # Turnover (how often signal changes sign)
    signal_changes = (np.sign(alpha) != np.sign(alpha.shift(1))).sum()
    turnover = signal_changes / len(alpha)
    
    return {
        'sharpe_ratio': float(sharpe),
        'mean_return': float(mean_ret * 252),  # Annualized
        'std_return': float(std_ret * np.sqrt(252)),  # Annualized
        'max_drawdown': float(max_drawdown),
        'turnover': float(turnover),
        'n_observations': len(strategy_returns)
    }