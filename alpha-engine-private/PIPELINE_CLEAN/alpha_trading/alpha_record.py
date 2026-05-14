from dataclasses import dataclass, asdict

# =============================================================================
# ALPHA RECORD: Metadata for a single alpha
# =============================================================================

@dataclass
class AlphaRecord:
    """
    Complete record of a discovered alpha expression.
    
    Stores both the expression string AND performance metrics,
    so we can filter/rank alphas later without re-evaluating.
    """
    # Identity
    name: str                          # Unique identifier (e.g., "alpha_001")
    expression: str                    # The formula (e.g., "sub(close, open)")
    
    # Performance metrics (from training)
    sharpe_ratio: float = 0.0          # Risk-adjusted return
    mean_return: float = 0.0           # Average daily return
    std_return: float = 0.0            # Volatility of returns
    max_drawdown: float = 0.0          # Worst peak-to-trough decline
    
    # Metadata
    created_at: str = ""               # When discovered
    training_ticker: str = ""          # Which ticker was used for training
    training_period: str = ""          # Date range used
    complexity: int = 0                # Number of nodes in expression tree
    
    # Additional metrics (can be computed later)
    turnover: float = 0.0              # How often the signal changes
    correlation_to_market: float = 0.0 # Beta to SPY/market
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> 'AlphaRecord':
        """Create from dictionary."""
        return cls(**d)
