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
from .node import *

# =============================================================================
# SECTION 2: EXPRESSION CLASS (Complete Formula)
# =============================================================================

class Expression:
    """
    A complete alpha expression (wrapper around the root node).
    
    Provides a clean interface to:
    - Evaluate the expression on stock data
    - Convert to string representation
    - Measure complexity
    
    Usage:
        expr = Expression(root_node)
        alpha_values = expr.evaluate(stock_data)
        print(expr)  # Shows the formula
    """
    
    def __init__(self, root: ExpressionNode):
        """
        Create an expression from a root node.
        
        Parameters:
            root: The root ExpressionNode of the tree
        """
        self.root = root
    
    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """
        Evaluate the expression on stock data.
        
        Parameters:
            data: DataFrame with columns: Open, High, Low, Close, Volume
            
        Returns:
            pd.Series with alpha values for each row
        """
        result = self.root.evaluate(data)
        
        # Handle any NaN or infinite values
        result = result.replace([float('inf'), float('-inf')], 0)
        result = result.fillna(0)
        
        return result
    
    def to_string(self) -> str:
        """Get string representation of the formula."""
        return self.root.to_string()
    
    def __repr__(self) -> str:
        return f"Expression({self.to_string()})"
    
    def __str__(self) -> str:
        return self.to_string()
    
    def depth(self) -> int:
        """Get the depth of the expression tree."""
        return self.root.depth()
    
    def size(self) -> int:
        """Get the total number of nodes."""
        return self.root.size()
    
    def complexity(self) -> float:
        """
        Simple complexity measure for the expression.
        
        Can be used as a penalty in the reward function
        to prefer simpler expressions.
        """
        return self.size()


# =============================================================================
# SECTION 3: HELPER FUNCTION TO BUILD EXPRESSIONS
# =============================================================================

def build_expression(dsl, token_name: str, 
                     children: Optional[List[ExpressionNode]] = None,
                     window: Optional[int] = None) -> ExpressionNode:
    """
    Helper function to easily build expression nodes.
    
    Parameters:
        dsl: The DSL instance
        token_name: Name of the token (e.g., 'close', 'add', 'ts_mean')
        children: Child nodes for operators
        window: Window size for time-series operators
        
    Returns:
        ExpressionNode
        
    Example:
        close_node = build_expression(dsl, 'close')
        open_node = build_expression(dsl, 'open')
        sub_node = build_expression(dsl, 'sub', [close_node, open_node])
    """
    token = dsl.get_token(token_name)
    if token is None:
        raise ValueError(f"Unknown token: {token_name}")
    
    return ExpressionNode(token, children, window)

