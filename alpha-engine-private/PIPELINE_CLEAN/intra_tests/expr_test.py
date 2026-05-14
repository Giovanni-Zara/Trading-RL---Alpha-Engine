
# =============================================================================
# SECTION 4: TESTING
# =============================================================================
import pandas as pd
import sys
import os
import numpy as np

_PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
    

from typing import List, Optional, Union
from DSL.utils import Token, TokenType
from DSL.node import *
from DSL.expression import *
from DSL.dsl import *




if __name__ == "__main__":
    
    #from dsl import DSL
    
    # Create DSL and sample data
    dsl = DSL()
    
    sample_data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    print("=" * 60)
    print("EXPRESSION TREE - Examples")
    print("=" * 60)
    
    # ----- Example 1: Simple operand -----
    print("\n--- Example 1: Simple operand ---")
    close_node = ExpressionNode(dsl.get_token('close'))
    expr1 = Expression(close_node)
    print(f"Expression: {expr1}")
    print(f"Result: {expr1.evaluate(sample_data).values}")
    
    # ----- Example 2: Unary operator -----
    print("\n--- Example 2: neg(close) ---")
    neg_close = ExpressionNode(
        dsl.get_token('neg'),
        children=[close_node]
    )
    expr2 = Expression(neg_close)
    print(f"Expression: {expr2}")
    print(f"Result: {expr2.evaluate(sample_data).values}")
    
    # ----- Example 3: Binary operator -----
    print("\n--- Example 3: sub(close, open) ---")
    open_node = ExpressionNode(dsl.get_token('open'))
    sub_node = ExpressionNode(
        dsl.get_token('sub'),
        children=[close_node, open_node]
    )
    expr3 = Expression(sub_node)
    print(f"Expression: {expr3}")
    print(f"Result: {expr3.evaluate(sample_data).values}")
    print("(This is the daily price change: Close - Open)")
    
    # ----- Example 4: Time-series operator -----
    print("\n--- Example 4: ts_mean(close, 3) ---")
    ts_mean_node = ExpressionNode(
        dsl.get_token('ts_mean'),
        children=[close_node],
        window=3
    )
    expr4 = Expression(ts_mean_node)
    print(f"Expression: {expr4}")
    print(f"Result: {expr4.evaluate(sample_data).values}")
    print("(3-day moving average of Close)")
    
    # ----- Example 5: Complex nested expression -----
    print("\n--- Example 5: div(sub(close, open), volume) ---")
    # Formula: (close - open) / volume
    # This measures price change normalized by volume
    
    volume_node = ExpressionNode(dsl.get_token('volume'))
    div_node = ExpressionNode(
        dsl.get_token('div'),
        children=[sub_node, volume_node]  # reuse sub_node from Example 3
    )
    expr5 = Expression(div_node)
    print(f"Expression: {expr5}")
    print(f"Result: {expr5.evaluate(sample_data).values}")
    print(f"Depth: {expr5.depth()}, Size: {expr5.size()}")
    
    # ----- Example 6: Even more complex -----
    print("\n--- Example 6: sub(close, ts_mean(close, 5)) ---")
    # Formula: close - 5-day moving average
    # This measures deviation from the trend (mean reversion signal)
    
    ts_mean_5 = ExpressionNode(
        dsl.get_token('ts_mean'),
        children=[ExpressionNode(dsl.get_token('close'))],
        window=5
    )
    deviation = ExpressionNode(
        dsl.get_token('sub'),
        children=[ExpressionNode(dsl.get_token('close')), ts_mean_5]
    )
    expr6 = Expression(deviation)
    print(f"Expression: {expr6}")
    print(f"Result: {expr6.evaluate(sample_data).values}")
    print(f"Depth: {expr6.depth()}, Size: {expr6.size()}")
    
    # ----- Using the helper function -----
    print("\n--- Example 7: Using build_expression helper ---")
    close_n = build_expression(dsl, 'close')
    open_n = build_expression(dsl, 'open')
    high_n = build_expression(dsl, 'high')
    low_n = build_expression(dsl, 'low')
    
    # (high + low) / 2 = midpoint price
    add_hl = build_expression(dsl, 'add', [high_n, low_n])
    const_2 = build_expression(dsl, '2')
    midpoint = build_expression(dsl, 'div', [add_hl, const_2])
    
    expr7 = Expression(midpoint)
    print(f"Expression: {expr7}")
    print(f"Result: {expr7.evaluate(sample_data).values}")
    print("(Midpoint price: (High + Low) / 2)")
    
    print("\n" + "=" * 60)
    print("Expression Tree is ready!")
    print("=" * 60)