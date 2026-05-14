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
from abc import ABC, abstractmethod
from typing import Union, List, Optional
from enum import Enum, auto


# =============================================================================
# SECTION 1: TOKEN TYPES
# =============================================================================

class TokenType(Enum):
    """
    Enumeration of all possible token types in our DSL.
    
    Think of tokens as the "words" in our mini-language.
    Each token type represents a different kind of building block.
    
    The 'auto()' function automatically assigns unique integer values.
    """
    # Operands - the raw data we work with
    OPERAND = auto()      # e.g., 'close', 'open', 'volume'
    CONSTANT = auto()     # e.g., -1, 0, 0.5, 1, 2
    
    # Operators - the operations we can perform
    UNARY_OP = auto()     # Takes 1 input  (e.g., neg(x), abs(x))
    BINARY_OP = auto()    # Takes 2 inputs (e.g., add(x, y), mul(x, y))
    TS_OP = auto()        # Time-series operators (e.g., ts_mean(x, 5))


# =============================================================================
# SECTION 2: BASE CLASSES (Abstract Templates)
# =============================================================================

class Token(ABC):
    """
    Abstract base class for all tokens in our DSL.
    
    ABC = Abstract Base Class
    This is like a "template" that all tokens must follow.
    We can't create a Token directly - we must create specific types like Operand.
    
    Attributes:
        name (str): Human-readable name of the token (e.g., 'close', 'add')
        token_type (TokenType): What kind of token this is
    """
    
    def __init__(self, name: str, token_type: TokenType):
        self.name = name
        self.token_type = token_type
    
    @abstractmethod
    def __call__(self, data: pd.DataFrame, *args) -> pd.Series:
        """
        Execute this token on data.
        
        The @abstractmethod decorator means every subclass MUST implement this.
        
        Parameters:
            data: DataFrame with columns like 'open', 'high', 'low', 'close', 'volume'
            *args: Additional arguments (varies by token type)
        
        Returns:
            pd.Series: The result of applying this token
        """
        pass
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.__class__.__name__}({self.name})"


# =============================================================================
# SECTION 3: OPERANDS (The Raw Data)
# =============================================================================

class Operand(Token):
    """
    Represents a raw data column (open, high, low, close, volume).
    
    When called, it simply extracts that column from the DataFrame.
    
    Example:
        close_operand = Operand('close')
        prices = close_operand(data)  # Returns the 'close' column
    """
    
    def __init__(self, name: str):
        super().__init__(name, TokenType.OPERAND)
    
    def __call__(self, data: pd.DataFrame, *args) -> pd.Series:
        """Extract the column from the DataFrame."""
        return data[self.name].astype(float)


class Constant(Token):
    """
    Represents a numeric constant value.
    
    When called, returns a Series filled with that constant value.
    This allows mixing constants with price data in expressions.
    
    Example:
        const_2 = Constant(2)
        twos = const_2(data)  # Returns Series of 2s, same length as data
    """
    
    def __init__(self, value: float):
        # Use the value as the name for display purposes
        super().__init__(str(value), TokenType.CONSTANT)
        self.value = value
    
    def __call__(self, data: pd.DataFrame, *args) -> pd.Series:
        """Return a Series filled with this constant value."""
        return pd.Series(self.value, index=data.index)


# =============================================================================
# SECTION 4: UNARY OPERATORS (One Input)
# =============================================================================

class UnaryOperator(Token):
    """
    Operators that take exactly ONE input.
    
    Examples: neg(x), abs(x), log(x), sign(x)
    
    The 'func' parameter is the actual Python function to apply.
    """
    
    def __init__(self, name: str, func):
        """
        Parameters:
            name: Display name of the operator (e.g., 'neg')
            func: A callable that takes a Series and returns a Series
        """
        super().__init__(name, TokenType.UNARY_OP)
        self.func = func
    
    def __call__(self, data: pd.DataFrame, x: pd.Series) -> pd.Series:
        """
        Apply the unary operation to input x.
        
        Parameters:
            data: The full DataFrame (needed for index/length)
            x: The input Series to transform
        
        Returns:
            Transformed Series
        """
        return self.func(x)


# =============================================================================
# SECTION 5: BINARY OPERATORS (Two Inputs)
# =============================================================================

class BinaryOperator(Token):
    """
    Operators that take exactly TWO inputs.
    
    Examples: add(x, y), sub(x, y), mul(x, y), div(x, y)
    """
    
    def __init__(self, name: str, func):
        """
        Parameters:
            name: Display name (e.g., 'add')
            func: A callable that takes two Series and returns a Series
        """
        super().__init__(name, TokenType.BINARY_OP)
        self.func = func
    
    def __call__(self, data: pd.DataFrame, x: pd.Series, y: pd.Series) -> pd.Series:
        """
        Apply the binary operation to inputs x and y.
        
        Parameters:
            data: The full DataFrame
            x: First input Series
            y: Second input Series
        
        Returns:
            Combined Series
        """
        return self.func(x, y)


# =============================================================================
# SECTION 6: TIME-SERIES OPERATORS (Look at History)
# =============================================================================

class TimeSeriesOperator(Token):
    """
    Operators that look at historical data using a window.
    
    Examples: 
        ts_mean(x, 5) - 5-day moving average
        ts_std(x, 10) - 10-day standard deviation
        delay(x, 1) - Yesterday's value
    
    These are crucial for finance because trends and patterns
    often depend on what happened in the past.
    """
    
    def __init__(self, name: str, func, default_window: int = 5):
        """
        Parameters:
            name: Display name (e.g., 'ts_mean')
            func: Callable(series, window) -> Series
            default_window: Default lookback period (in days)
        """
        super().__init__(name, TokenType.TS_OP)
        self.func = func
        self.default_window = default_window
    
    def __call__(self, data: pd.DataFrame, x: pd.Series, 
                 window: Optional[int] = None) -> pd.Series:
        """
        Apply time-series operation with given window.
        
        Parameters:
            data: Full DataFrame
            x: Input Series
            window: Lookback period (uses default if not specified)
        
        Returns:
            Time-series transformed Series
        """
        w = window if window is not None else self.default_window
        return self.func(x, w)


# =============================================================================
# SECTION 7: SAFE MATHEMATICAL FUNCTIONS
# =============================================================================

def safe_div(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    Safe division that handles division by zero.
    
    In finance, we often divide by values that might be zero
    (e.g., volume on a day with no trading). This prevents errors.
    
    Returns 0 where y is 0, otherwise x/y.
    """
    result = np.where(y != 0, x / y, 0)
    return pd.Series(result, index = x.index)


def safe_log(x: pd.Series) -> pd.Series:
    """
    Safe logarithm that handles non-positive values.
    
    log(x) is undefined for x <= 0.
    We use the absolute value and avoid zeros.
    
    This is common in finance for computing log-returns.
    """
    return np.log(np.abs(x) + 1e-10)  # 1e-10 prevents log(0)


def sign(x: pd.Series) -> pd.Series:
    """
    Returns the sign of each value: -1, 0, or +1.
    
    Useful for determining direction of price movement.
    """
    return np.sign(x)


# =============================================================================
# SECTION 8: TIME-SERIES FUNCTIONS
# =============================================================================

def ts_delay(x: pd.Series, d: int) -> pd.Series:
    """
    Shift the series by d days (look at past values).
    
    ts_delay(close, 1) gives yesterday's closing price.
    ts_delay(close, 5) gives the closing price from 5 days ago.
    
    This is essential for computing returns, momentum, etc.
    """
    return x.shift(d)


def ts_delta(x: pd.Series, d: int) -> pd.Series:
    """
    Change over d days: x_today - x_{d days ago}
    
    ts_delta(close, 1) = today's close - yesterday's close
    
    Useful for measuring momentum/change.
    """
    return x - x.shift(d)


def ts_mean(x: pd.Series, d: int) -> pd.Series:
    """
    Moving average over past d days.
    
    ts_mean(close, 5) = average of last 5 closing prices
    
    Smooths out noise and reveals trends.
    """
    return x.rolling(window=d, min_periods=1).mean()


def ts_std(x: pd.Series, d: int) -> pd.Series:
    """
    Rolling standard deviation over past d days.
    
    Measures volatility/uncertainty in recent prices.
    High ts_std = high volatility
    """
    return x.rolling(window=d, min_periods=1).std()


def ts_max(x: pd.Series, d: int) -> pd.Series:
    """
    Maximum value over past d days.
    
    Useful for identifying recent highs.
    """
    return x.rolling(window=d, min_periods=1).max()


def ts_min(x: pd.Series, d: int) -> pd.Series:
    """
    Minimum value over past d days.
    
    Useful for identifying recent lows.
    """
    return x.rolling(window=d, min_periods=1).min()


def ts_rank(x: pd.Series, d: int) -> pd.Series:
    """
    Rank of current value among past d days (percentile).
    
    Returns a value between 0 and 1.
    ts_rank(close, 10) = 1.0 means today's close is the highest in 10 days.
    
    Useful for relative position analysis.
    """
    def rank_pct(arr):
        if len(arr) < 2:
            return 0.5
        # Where does the last value rank among all values?
        return (arr.argsort().argsort()[-1]) / (len(arr) - 1)
    
    return x.rolling(window=d, min_periods=1).apply(rank_pct, raw=True)

def ts_corr(x: pd.Series, y: pd.Series, d: int) -> pd.Series:
    """Rolling correlation between two series."""
    return x.rolling(window=d, min_periods=d//2).corr(y)

def ts_zscore(x: pd.Series, d: int) -> pd.Series:
    """Z-score normalization over window - crucial for alphas."""
    mean = x.rolling(window=d, min_periods=1).mean()
    std = x.rolling(window=d, min_periods=1).std()
    return (x - mean) / (std + 1e-8)

def returns(x: pd.Series, d: int) -> pd.Series:
    """Percentage returns over d days."""
    return x.pct_change(d)


# =============================================================================
# SECTION 9: KNOWN-GOOD ALPHA PRIMITIVES
# =============================================================================

def momentum(close: pd.Series, d: int) -> pd.Series:
    """
    Classic momentum: return over d days.
    Known to work in equities with d=20, 60, 120, 252.
    """
    return close.pct_change(d)


def mean_reversion(close: pd.Series, d: int) -> pd.Series:
    """
    Mean reversion: z-score of price vs moving average.
    Negative = oversold, Positive = overbought.
    Works best with d=5, 10, 20.
    """
    ma = close.rolling(window=d, min_periods=1).mean()
    std = close.rolling(window=d, min_periods=1).std()
    return -((close - ma) / (std + 1e-8))  # Negative because we BUY oversold


def volatility_adjusted_momentum(close: pd.Series, d: int) -> pd.Series:
    """
    Risk-adjusted momentum: return / volatility.
    More robust than raw momentum.
    """
    ret = close.pct_change(d)
    vol = close.pct_change().rolling(window=d, min_periods=1).std()
    return ret / (vol * np.sqrt(d) + 1e-8)


def price_acceleration(close: pd.Series, d: int) -> pd.Series:
    """
    Second derivative of price: momentum of momentum.
    Captures trend acceleration/deceleration.
    """
    mom1 = close.pct_change(d)
    mom2 = mom1.shift(d)
    return mom1 - mom2


def volume_price_trend(close: pd.Series, volume: pd.Series, d: int) -> pd.Series:
    """
    Volume-weighted price trend.
    High volume on up days = bullish, high volume on down days = bearish.
    """
    ret = close.pct_change()
    vpt = (ret * volume).rolling(window=d, min_periods=1).sum()
    vol_ma = volume.rolling(window=d, min_periods=1).mean()
    return vpt / (vol_ma + 1e-8)


def relative_strength(close: pd.Series, d: int) -> pd.Series:
    """
    RSI-style relative strength (0 to 1 scale).
    Below 0.3 = oversold, Above 0.7 = overbought.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=d, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=d, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    return rs / (1 + rs)  # Normalize to 0-1


def bollinger_position(close: pd.Series, d: int) -> pd.Series:
    """
    Position within Bollinger Bands (-1 to +1).
    -1 = at lower band, +1 = at upper band.
    """
    ma = close.rolling(window=d, min_periods=1).mean()
    std = close.rolling(window=d, min_periods=1).std()
    return (close - ma) / (2 * std + 1e-8)


def overnight_gap(open_price: pd.Series, close: pd.Series, d: int) -> pd.Series:
    """
    Average overnight gap over d days.
    Captures overnight sentiment.
    """
    gap = (open_price - close.shift(1)) / (close.shift(1) + 1e-8)
    return gap.rolling(window=d, min_periods=1).mean()


def intraday_range(high: pd.Series, low: pd.Series, close: pd.Series, d: int) -> pd.Series:
    """
    Average true range normalized by price.
    Measures intraday volatility.
    """
    tr = (high - low) / (close + 1e-8)
    return tr.rolling(window=d, min_periods=1).mean()


def close_location(high: pd.Series, low: pd.Series, close: pd.Series, d: int) -> pd.Series:
    """
    Where did price close within the day's range? (0 to 1)
    High values = closed near high (bullish), Low values = closed near low (bearish).
    Averaged over d days for smoothing.
    """
    loc = (close - low) / (high - low + 1e-8)
    return loc.rolling(window=d, min_periods=1).mean()


def volume_surge(volume: pd.Series, d: int) -> pd.Series:
    """
    Current volume vs average volume.
    High values = unusual activity.
    """
    vol_ma = volume.rolling(window=d, min_periods=1).mean()
    return volume / (vol_ma + 1e-8) - 1


def price_volume_divergence(close: pd.Series, volume: pd.Series, d: int) -> pd.Series:
    """
    Divergence between price momentum and volume momentum.
    Price up + volume down = weakening trend.
    """
    price_mom = close.pct_change(d)
    vol_mom = volume.pct_change(d)
    return price_mom - vol_mom