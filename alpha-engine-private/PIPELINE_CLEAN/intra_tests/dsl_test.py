
# =============================================================================
# SECTION 10: TESTING THE DSL
# =============================================================================
import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


print("----")
print(sys.path)
print("----")

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from enum import Enum, auto
from DSL.utils import *
from DSL.dsl import DSL


if __name__ == "__main__":
    # Create the DSL
    dsl = DSL()
    
    # Show all available tokens
    print("=" * 60)
    print("ALPHA EXPRESSION DSL - Token Summary")
    print("=" * 60)
    print(dsl.get_token_info().to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Action Space (for RL agent):")
    print("=" * 60)
    actions = dsl.get_action_space()
    print(f"Total actions: {len(actions)}")
    print(f"Actions: {actions}")
    
    # ----- Demo with sample data -----
    print("\n" + "=" * 60)
    print("Demo: Applying tokens to sample data")
    print("=" * 60)
    
    # Create sample OHLCV data
    sample_data = pd.DataFrame({
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'Low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'Close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'Volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    })
    
    print("\nSample Data:")
    print(sample_data)
    
    # Test operand
    close = dsl.get_token('close')
    close_prices = close(sample_data)
    print(f"\nclose operand:\n{close_prices.values}")
    
    # Test unary operator
    neg = dsl.get_token('neg')
    neg_close = neg(sample_data, close_prices)
    print(f"\nneg(close):\n{neg_close.values}")
    
    # Test binary operator
    add = dsl.get_token('add')
    high = dsl.get_token('high')
    low = dsl.get_token('low')
    high_prices = high(sample_data)
    low_prices = low(sample_data)
    hl_sum = add(sample_data, high_prices, low_prices)
    print(f"\nadd(high, low):\n{hl_sum.values}")
    
    # Test time-series operator
    ts_mean_op = dsl.get_token('ts_mean')
    ma_close = ts_mean_op(sample_data, close_prices, window=3)
    print(f"\nts_mean(close, 3) [3-day moving average]:\n{ma_close.values}")
    
    # Test delay
    delay_op = dsl.get_token('delay')
    delayed_close = delay_op(sample_data, close_prices, window=1)
    print(f"\ndelay(close, 1) [yesterday's close]:\n{delayed_close.values}")

    rank_op = dsl.get_token('ts_rank')
    print(f"rank of the last 3 days\n{rank_op(sample_data, close_prices, window=3).values}")
    

    print("\n" + "=" * 60)
    print("TESTING RANK WITH VOLATILE DATA")
    print("=" * 60)

    # Data that goes Up, Down, then Middle
    volatile_data = pd.DataFrame({
        'Close': [100, 110, 90, 105, 100], 
        # 100: Initial
        # 110: Max (Rank 1.0)
        # 90:  Min (Rank 0.0)
        # 105: High but not max (Rank ~0.75 or 0.66)
        # 100: Middle
    })
    
    rank_op = dsl.get_token('ts_rank')
    close_token = dsl.get_token('close')
    
    v_close = close_token(volatile_data)
    v_rank = rank_op(volatile_data, v_close, window=3)
    
    print("Prices:", v_close.values)
    print("Ranks: ", v_rank.values)


    print("\n" + "=" * 60)
    print("DSL is ready for use!")
    print("=" * 60)