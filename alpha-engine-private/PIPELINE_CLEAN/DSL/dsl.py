"""
Domain-Specific Language (DSL) for Alpha Expression Synthesis

This module defines the building blocks for constructing financial alpha expressions:
- Operands: Raw price/volume data (open, high, low, close, volume)
- Operators: Mathematical operations to combine and transform data
- Expression Tree: Structure to represent complete alpha formulas

The RL agent will use these components to build alpha expressions step-by-step.
"""

import numpy as np
import pandas as pd
import sys
import os
from abc import ABC, abstractmethod
from typing import Union, List, Optional
from enum import Enum, auto
from .utils import (
    Operand, Constant, UnaryOperator, BinaryOperator, TimeSeriesOperator, Token,
    safe_div, safe_log, sign,
    ts_delay, ts_delta, ts_mean, ts_std, ts_max, ts_min, ts_rank,
    ts_zscore, returns,
    # New alpha primitives
    momentum, mean_reversion, volatility_adjusted_momentum,
    price_acceleration, volume_price_trend, relative_strength,
    bollinger_position, overnight_gap, intraday_range, close_location,
    volume_surge, price_volume_divergence
)
from .expression import Expression
from .node import ExpressionNode

# =============================================================================
# NEW: COMPOUND ALPHA OPERATOR (takes multiple inputs from data)
# =============================================================================

class CompoundAlphaOperator(Token):
    """
    Pre-built alpha factors that operate on multiple OHLCV columns.
    These encapsulate known-good alpha logic.
    """
    
    def __init__(self, name: str, func, required_columns: List[str], default_window: int = 20):
        from .utils import TokenType
        super().__init__(name, TokenType.TS_OP)  # Treat as TS_OP for action space
        self.func = func
        self.required_columns = required_columns
        self.default_window = default_window
    
    def __call__(self, data: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
        w = window if window is not None else self.default_window
        # Extract required columns and call the function
        args = [data[col].astype(float) for col in self.required_columns]
        return self.func(*args, w)


# =============================================================================
# SECTION 9: DSL CLASS (Putting It All Together)
# =============================================================================

class DSL:
    """
    The complete Domain-Specific Language for alpha expressions.
    
    This class holds all available tokens (operands, constants, operators)
    and provides methods to access them.
    
    Usage:
        dsl = DSL()
        print(dsl.get_all_tokens())  # See all available building blocks
    """
    
    def __init__(self):
        """Initialize all tokens in our DSL."""
        
        # ----- OPERANDS -----
        # These extract raw price/volume data from the DataFrame
        self.operands = {
            'open': Operand('Open'),      # Opening price
            'high': Operand('High'),      # Highest price of the day
            'low': Operand('Low'),        # Lowest price of the day
            'close': Operand('Close'),    # Closing price
            'volume': Operand('Volume'),  # Trading volume
        }
        
        # ----- CONSTANTS -----
        # Commonly used numeric values
        self.constants = {
            '-1': Constant(-1),
            '0': Constant(0),
            '0.5': Constant(0.5),
            '1': Constant(1),
            '2': Constant(2),
        }
        
        # ----- UNARY OPERATORS -----
        # Transform a single input
        self.unary_ops = {
            'neg': UnaryOperator('neg', lambda x: -x),           # Negate
            'abs': UnaryOperator('abs', lambda x: np.abs(x)),    # Absolute value
            'log': UnaryOperator('log', safe_log),               # Safe logarithm
            'sign': UnaryOperator('sign', sign),                 # Sign (-1, 0, +1)
        }
        
        # ----- BINARY OPERATORS -----
        # Combine two inputs
        self.binary_ops = {
            'add': BinaryOperator('add', lambda x, y: x + y),    # Addition
            'sub': BinaryOperator('sub', lambda x, y: x - y),    # Subtraction
            'mul': BinaryOperator('mul', lambda x, y: x * y),    # Multiplication
            'div': BinaryOperator('div', safe_div),              # Safe division
            'max': BinaryOperator('max', lambda x, y: np.maximum(x, y)),  # Element-wise max
            'min': BinaryOperator('min', lambda x, y: np.minimum(x, y)),  # Element-wise min
        }
        
        # ----- TIME-SERIES OPERATORS -----
        self.ts_ops = {
            'delay': TimeSeriesOperator('delay', ts_delay, default_window=1),
            'delta': TimeSeriesOperator('delta', ts_delta, default_window=1),
            'ts_mean': TimeSeriesOperator('ts_mean', ts_mean, default_window=5),
            'ts_std': TimeSeriesOperator('ts_std', ts_std, default_window=5),
            'ts_max': TimeSeriesOperator('ts_max', ts_max, default_window=5),
            'ts_min': TimeSeriesOperator('ts_min', ts_min, default_window=5),
            'ts_rank': TimeSeriesOperator('ts_rank', ts_rank, default_window=5),
            'ts_zscore': TimeSeriesOperator('ts_zscore', ts_zscore, default_window=20),
            'returns': TimeSeriesOperator('returns', returns, default_window=5),
        }
        
        # ----- KNOWN-GOOD ALPHA PRIMITIVES -----
        # These are pre-built factors that work well in practice
        self.alpha_primitives = {
            # Momentum family
            'alpha_mom': CompoundAlphaOperator(
                'alpha_mom', 
                lambda c, d: momentum(c, d),
                ['Close'], default_window=20
            ),
            'alpha_mom_vol_adj': CompoundAlphaOperator(
                'alpha_mom_vol_adj',
                lambda c, d: volatility_adjusted_momentum(c, d),
                ['Close'], default_window=20
            ),
            'alpha_acceleration': CompoundAlphaOperator(
                'alpha_acceleration',
                lambda c, d: price_acceleration(c, d),
                ['Close'], default_window=10
            ),
            
            # Mean reversion family
            'alpha_mean_rev': CompoundAlphaOperator(
                'alpha_mean_rev',
                lambda c, d: mean_reversion(c, d),
                ['Close'], default_window=20
            ),
            'alpha_rsi': CompoundAlphaOperator(
                'alpha_rsi',
                lambda c, d: relative_strength(c, d) - 0.5,  # Center at 0
                ['Close'], default_window=14
            ),
            'alpha_bollinger': CompoundAlphaOperator(
                'alpha_bollinger',
                lambda c, d: -bollinger_position(c, d),  # Contrarian
                ['Close'], default_window=20
            ),
            
            # Volume family
            'alpha_vpt': CompoundAlphaOperator(
                'alpha_vpt',
                lambda c, v, d: volume_price_trend(c, v, d),
                ['Close', 'Volume'], default_window=20
            ),
            'alpha_vol_surge': CompoundAlphaOperator(
                'alpha_vol_surge',
                lambda v, d: volume_surge(v, d),
                ['Volume'], default_window=20
            ),
            'alpha_pv_diverge': CompoundAlphaOperator(
                'alpha_pv_diverge',
                lambda c, v, d: price_volume_divergence(c, v, d),
                ['Close', 'Volume'], default_window=20
            ),
            
            # Price action family
            'alpha_gap': CompoundAlphaOperator(
                'alpha_gap',
                lambda o, c, d: overnight_gap(o, c, d),
                ['Open', 'Close'], default_window=5
            ),
            'alpha_range': CompoundAlphaOperator(
                'alpha_range',
                lambda h, l, c, d: -intraday_range(h, l, c, d),  # Low range = bullish
                ['High', 'Low', 'Close'], default_window=10
            ),
            'alpha_close_loc': CompoundAlphaOperator(
                'alpha_close_loc',
                lambda h, l, c, d: close_location(h, l, c, d) - 0.5,
                ['High', 'Low', 'Close'], default_window=5
            ),
        }

    def parse(self, expr_str: str) -> Expression:
        """
        Parse a string expression into an Expression object.
        
        Handles:
            - Operands: 'Close', 'Open', 'High', 'Low', 'Volume'
            - Constants: '-1', '0', '0.5', '1', '2'
            - Unary ops: 'neg(x)', 'abs(x)', 'log(x)', 'sign(x)'
            - Binary ops: 'add(x, y)', 'sub(x, y)', 'mul(x, y)', etc.
            - TS ops: 'ts_mean(x, 5)', 'ts_rank(x, 10)', etc.
            - Alpha primitives: 'alpha_bollinger(20)', 'alpha_mean_rev(5)', etc.
        
        Parameters:
            expr_str: String representation of the expression
            
        Returns:
            Expression object that can be evaluated on data
        """
        
        expr_str = expr_str.strip()
        root = self._parse_node(expr_str)
        return Expression(root)
    
    def _parse_node(self, s: str) -> ExpressionNode:
        """
        Recursively parse a string into an ExpressionNode.
        """
        from .node import ExpressionNode
        
        s = s.strip()
        
        # --- Case 1: Numeric constant (including negative numbers like -1) ---
        try:
            val = float(s)
            # Find matching constant token or create one
            const_token = self.constants.get(s)
            if const_token is None:
                # Try common representations
                if val == int(val):
                    const_token = self.constants.get(str(int(val)))
                if const_token is None:
                    const_token = self.constants.get(str(val))
                if const_token is None:
                    # Create a new constant for this value
                    from .utils import Constant
                    const_token = Constant(val)
            return ExpressionNode(const_token)
        except ValueError:
            pass
        
        # --- Case 2: Simple operand (no parentheses) ---
        if '(' not in s:
            # Check operands (case-insensitive lookup)
            operand_token = self.operands.get(s.lower())
            if operand_token is None:
                # Try exact match (e.g., 'Close', 'Volume')
                for name, token in self.operands.items():
                    if token.name == s:
                        operand_token = token
                        break
            if operand_token is not None:
                return ExpressionNode(operand_token)
            
            # Unknown token - treat as constant 0
            from .utils import Constant
            return ExpressionNode(Constant(0))
        
        # --- Case 3: Function call: name(args...) ---
        paren_pos = s.index('(')
        func_name = s[:paren_pos].strip()
        # Extract inner content (strip outer parentheses)
        inner = s[paren_pos + 1:-1].strip()
        
        # --- Case 3a: Alpha primitive: alpha_xxx(window) ---
        if func_name in self.alpha_primitives:
            alpha_token = self.alpha_primitives[func_name]
            try:
                window = int(inner)
            except ValueError:
                window = alpha_token.default_window
            return ExpressionNode(alpha_token, [], window)
        
        # --- Split arguments at top-level commas ---
        args = self._split_args(inner)
        
        # --- Case 3b: Unary operator: func(x) ---
        if func_name in self.unary_ops and len(args) == 1:
            child = self._parse_node(args[0])
            return ExpressionNode(self.unary_ops[func_name], [child])
        
        # --- Case 3c: Binary operator: func(x, y) ---
        if func_name in self.binary_ops and len(args) == 2:
            left = self._parse_node(args[0])
            right = self._parse_node(args[1])
            return ExpressionNode(self.binary_ops[func_name], [left, right])
        
        # --- Case 3d: Time-series operator: func(x, window) ---
        if func_name in self.ts_ops and len(args) == 2:
            child = self._parse_node(args[0])
            try:
                window = int(args[1].strip())
            except ValueError:
                window = self.ts_ops[func_name].default_window
            return ExpressionNode(self.ts_ops[func_name], [child], window)
        
        # --- Case 3e: TS op with default window (1 arg) ---
        if func_name in self.ts_ops and len(args) == 1:
            child = self._parse_node(args[0])
            return ExpressionNode(
                self.ts_ops[func_name], [child], 
                self.ts_ops[func_name].default_window
            )
        
        # --- Fallback: try as unary with first arg ---
        if func_name in self.unary_ops:
            child = self._parse_node(args[0])
            return ExpressionNode(self.unary_ops[func_name], [child])
        
        raise ValueError(f"Cannot parse expression: '{s}' (unknown function: '{func_name}')")
    
    def _split_args(self, s: str) -> List[str]:
        """
        Split a string by top-level commas (respecting nested parentheses).
        
        Example:
            "add(x, y), z" -> ["add(x, y)", "z"]
            "x, 5" -> ["x", "5"]
        """
        args = []
        depth = 0
        current = []
        
        for char in s:
            if char == '(':
                depth += 1
                current.append(char)
            elif char == ')':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                args.append(''.join(current).strip())
                current = []
            else:
                current.append(char)
        
        # Don't forget the last argument
        remaining = ''.join(current).strip()
        if remaining:
            args.append(remaining)
        
        return args
    
    
    def get_all_tokens(self) -> dict:
        """
        Return all tokens organized by category.
        
        Returns:
            Dictionary with keys: 'operands', 'constants', 'unary_ops', 
                                  'binary_ops', 'ts_ops'
        """
        return {
            'operands': self.operands,
            'constants': self.constants,
            'unary_ops': self.unary_ops,
            'binary_ops': self.binary_ops,
            'ts_ops': self.ts_ops,
            'alpha_primitives': self.alpha_primitives,

        }
    
    def get_token(self, name: str) -> Optional[Token]:
        """
        Get a token by its name.
        
        Parameters:
            name: Token name (e.g., 'close', 'add', 'ts_mean')
        
        Returns:
            The Token object, or None if not found
        """
        all_tokens = {
            **self.operands,
            **self.constants, 
            **self.unary_ops,
            **self.binary_ops,
            **self.ts_ops,
            **self.alpha_primitives

        }
        return all_tokens.get(name)
    
    def get_action_space(self) -> List[str]:
        """
        Get list of all possible actions (token names) for the RL agent.
        
        This will be used to define the agent's action space.
        
        Returns:
            List of all token names
        """
        actions = []
        actions.extend(self.operands.keys())
        actions.extend(self.constants.keys())
        actions.extend(self.unary_ops.keys())
        actions.extend(self.binary_ops.keys())
        actions.extend(self.ts_ops.keys())
        actions.extend(self.alpha_primitives.keys())

        return actions
    
    def get_token_info(self) -> pd.DataFrame:
        """
        Get a summary table of all tokens.
        
        Useful for understanding what's available in the DSL.
        """
        info = []
        
        for name, token in self.operands.items():
            info.append({'name': name, 'type': 'operand', 'inputs': 0})
        
        for name, token in self.constants.items():
            info.append({'name': name, 'type': 'constant', 'inputs': 0})
        
        for name, token in self.unary_ops.items():
            info.append({'name': name, 'type': 'unary', 'inputs': 1})
        
        for name, token in self.binary_ops.items():
            info.append({'name': name, 'type': 'binary', 'inputs': 2})
        
        for name, token in self.ts_ops.items():
            info.append({'name': name, 'type': 'time-series', 'inputs': '1 + window'})

        for name, token in self.alpha_primitives.items():
            info.append({'name': name, 'type': 'alpha-primitive', 'inputs': 'auto + window'})
               
        return pd.DataFrame(info)

