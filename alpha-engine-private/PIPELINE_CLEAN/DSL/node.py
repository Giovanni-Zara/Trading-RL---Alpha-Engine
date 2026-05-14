"""
Expression Tree for Alpha Formulas

This module provides a simple tree structure to represent and evaluate
alpha expressions built from DSL tokens.

Key concepts:
- ExpressionNode: A single node in the tree (operator or operand)
- Expression: The complete tree that can be evaluated on data
"""

import pandas as pd
import sys
import os
from typing import List, Optional, Union
from .utils import Token, TokenType


# =============================================================================
# SECTION 1: EXPRESSION NODE
# =============================================================================

class ExpressionNode:
    """
    A single node in the expression tree.
    
    Each node contains:
    - A token (operator or operand)
    - Children nodes (if it's an operator)
    - Optional window parameter (for time-series operators)
    
    Examples:
        Leaf node:     ExpressionNode(close_token)        -> represents 'close'
        Unary node:    ExpressionNode(neg_token, [child]) -> represents 'neg(child)'
        Binary node:   ExpressionNode(add_token, [a, b])  -> represents 'add(a, b)'
    """
    
    def __init__(
        self, 
        token: Token, 
        children: Optional[List['ExpressionNode']] = None,
        window: Optional[int] = None
    ):
        """
        Create an expression node.
        
        Parameters:
            token: The DSL token (operand, constant, or operator)
            children: List of child nodes (None for leaf nodes)
            window: Window size for time-series operators (optional)
        """
        self.token = token
        self.children = children if children is not None else []
        self.window = window
    
    def is_leaf(self) -> bool:
        """
        Check if this is a leaf node (no children).
        
        Leaf nodes are operands (close, open, etc.) or constants.
        """
        return len(self.children) == 0
    
    def is_alpha_primitive(self) -> bool:
        """Check if this is a pre-built alpha primitive."""
        # Import here to avoid circular imports
        try:
            from .dsl import CompoundAlphaOperator
            return isinstance(self.token, CompoundAlphaOperator)
        except ImportError:
            return False
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Recursively evaluate this node and its children on the data.
        
        This is the core method that computes the actual values.
        
        Parameters:
            data: DataFrame with OHLCV columns
            
        Returns:
            pd.Series with the computed values
        """
        token_type = self.token.token_type

        # ----- CASE 0: Alpha Primitive (pulls data directly) -----
        if self.is_alpha_primitive():
            window = self.window if self.window else self.token.default_window
            return self.token(data, window)
        
        # ----- CASE 1: Leaf node (operand or constant) -----
        # Just apply the token directly to get the data
        if token_type in (TokenType.OPERAND, TokenType.CONSTANT):
            return self.token(data)
        
        # ----- CASE 2: Unary operator -----
        # Evaluate the single child, then apply the operator
        elif token_type == TokenType.UNARY_OP:
            child_result = self.children[0].evaluate(data)
            return self.token(data, child_result)
        
        # ----- CASE 3: Binary operator -----
        # Evaluate both children, then apply the operator
        elif token_type == TokenType.BINARY_OP:
            left_result = self.children[0].evaluate(data)
            right_result = self.children[1].evaluate(data)
            return self.token(data, left_result, right_result)
        
        # ----- CASE 4: Time-series operator -----
        # Evaluate the child, then apply with window parameter
        elif token_type == TokenType.TS_OP:
            child_result = self.children[0].evaluate(data)
            return self.token(data, child_result, self.window)
        
        else:
            raise ValueError(f"Unknown token type: {token_type}")
    
    def to_string(self) -> str:
        """
        Convert the expression to a human-readable string.
        
        Examples:
            'close'
            'neg(close)'
            'add(close, open)'
            'ts_mean(close, 5)'
        """
        name = self.token.name

        # Alpha primitive: show with window
        if self.is_alpha_primitive():
            window = self.window if self.window else self.token.default_window
            return f"{name}({window})"
        
        # Leaf node: just the name
        if self.is_leaf():
            return name
        
        # Time-series operator: include window
        if self.token.token_type == TokenType.TS_OP:
            child_str = self.children[0].to_string()
            window = self.window if self.window else self.token.default_window
            return f"{name}({child_str}, {window})"
        
        # Other operators: list children
        children_str = ", ".join(child.to_string() for child in self.children)
        return f"{name}({children_str})"
    
    def __repr__(self) -> str:
        return f"ExpressionNode({self.to_string()})"
    
    def depth(self) -> int:
        """
        Calculate the depth of this subtree.
        
        Useful for limiting expression complexity.
        """
        if self.is_leaf() or self.is_alpha_primitive():
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def size(self) -> int:
        """
        Count total number of nodes in this subtree.
        
        Useful for measuring expression complexity.
        """
        if self.is_leaf() or self.is_alpha_primitive():
            return 1
        return 1 + sum(child.size() for child in self.children)

