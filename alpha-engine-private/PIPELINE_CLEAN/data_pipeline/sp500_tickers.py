"""
S&P 500 Historical Components Downloader

Downloads the S&P 500 constituents for a specific historical date.
This is important for avoiding SURVIVORSHIP BIAS - we want to know 
which stocks were in the S&P 500 at the time we're backtesting,
not which stocks are in it today.

Usage:
    python sp500_tickers.py                    # Uses default date (2016-02-15)
    python sp500_tickers.py --date 2018-06-30  # Uses specified date
    
Environment variable:
    SP500_SNAPSHOT_DATE=2016-02-15 python sp500_tickers.py
"""

import os
import sys
import re
import pandas as pd
from datetime import datetime
from typing import List, Optional, Tuple

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default snapshot date (can be overridden by env var or command line)
DEFAULT_SNAPSHOT_DATE = "2016-02-09"

# Path to the historical components CSV file
# Download from: https://github.com/fja05680/sp500
HISTORICAL_COMPONENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "S&P 500 Historical Components & Changes.csv"
)

# Output file path
OUTPUT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sp500_tickers.csv"
)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_historical_components(filepath: str) -> pd.DataFrame:
    """
    Load the S&P 500 historical components CSV file.
    
    Expected format:
        date,tickers
        2020-10-12,"A,AAL,AAP,AAPL,..."
        
    Returns:
        DataFrame with date index and tickers column (as comma-separated string)
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(
            f"Historical components file not found: {filepath}\n"
            f"Please download it from: https://github.com/fja05680/sp500\n"
            f"And place it in the data/ folder."
        )
    
    df = pd.read_csv(filepath, index_col='date')
    df.index = pd.to_datetime(df.index)
    return df


def get_sp500_tickers_at_date(df: pd.DataFrame, snapshot_date: str) -> List[str]:
    """
    Get the list of S&P 500 tickers that were in the index on a specific date.
    
    This finds the most recent record on or before the snapshot date.
    
    Parameters:
        df: DataFrame with historical components
        snapshot_date: Date string in YYYY-MM-DD format
        
    Returns:
        List of ticker symbols (sorted)
    """
    snapshot_dt = pd.to_datetime(snapshot_date)
    
    # Filter to rows on or before the snapshot date
    df_filtered = df[df.index <= snapshot_dt]
    
    if len(df_filtered) == 0:
        raise ValueError(
            f"No data available for date {snapshot_date}. "
            f"Earliest available date: {df.index.min()}"
        )
    
    # Get the most recent row
    last_row = df_filtered.iloc[-1]
    tickers_str = last_row['tickers']
    
    # Parse the comma-separated string into a sorted list
    tickers = sorted([t.strip() for t in tickers_str.split(',') if t.strip()])
    
    return tickers


def parse_ticker_with_date(ticker: str) -> Tuple[str, Optional[str]]:
    """
    Parse a ticker that may have a date suffix.
    
    The historical data file uses formats like:
    - 'AAMRQ-201312' -> ticker 'AAMRQ', removed Dec 2013
    - 'ABI-200811' -> ticker 'ABI', removed Nov 2008
    - 'BF-B' -> ticker 'BF-B' (Class B shares, no date suffix)
    - 'AAPL' -> ticker 'AAPL' (no suffix)
    
    Date suffix pattern: -YYYYMM (6 digits after dash)
    
    Returns:
        (clean_ticker, removal_date or None)
    """
    # Pattern: ticker ending with -YYYYMM (year 19xx or 20xx, month 01-12)
    pattern = r'^(.+)-((19|20)\d{2})(0[1-9]|1[0-2])$'
    match = re.match(pattern, ticker)
    
    if match:
        clean_ticker = match.group(1)
        year = match.group(2)
        month = match.group(4)
        removal_date = f"{year}-{month}"
        return clean_ticker, removal_date
    else:
        return ticker, None


def clean_tickers_for_yfinance(tickers: List[str], verbose: bool = True) -> Tuple[List[str], dict]:
    """
    Clean ticker symbols for use with yfinance.
    
    This function:
    1. Strips date suffixes (e.g., 'AAMRQ-201312' -> 'AAMRQ')
    2. Converts dots to dashes for yfinance (e.g., 'BF.B' -> 'BF-B')
    3. Tracks which tickers have removal dates (for survivorship bias awareness)
    
    IMPORTANT: We do NOT skip any tickers! We want to try downloading
    ALL of them to minimize survivorship bias. yfinance will fail for
    truly unavailable tickers, and we'll report those separately.
    
    Returns:
        (cleaned_tickers, metadata_dict)
        
        metadata_dict contains:
        - 'original_to_clean': mapping of original -> cleaned ticker
        - 'tickers_with_removal_dates': list of (ticker, removal_date)
    """
    cleaned = []
    metadata = {
        'original_to_clean': {},
        'tickers_with_removal_dates': [],
        'clean_to_original': {}
    }
    
    for ticker in tickers:
        # Parse out any date suffix
        clean_ticker, removal_date = parse_ticker_with_date(ticker)
        
        # Track tickers with removal dates (useful info for survivorship bias)
        if removal_date:
            metadata['tickers_with_removal_dates'].append((clean_ticker, removal_date))
        
        # Convert dots to dashes for yfinance
        yf_ticker = clean_ticker.replace('.', '-')
        
        # Handle duplicates (same underlying ticker with different date suffixes)
        if yf_ticker not in cleaned:
            cleaned.append(yf_ticker)
            metadata['original_to_clean'][ticker] = yf_ticker
            metadata['clean_to_original'][yf_ticker] = ticker
        else:
            if verbose:
                print(f"  Note: Duplicate ticker {yf_ticker} (from {ticker}), keeping first occurrence")
    
    if verbose and metadata['tickers_with_removal_dates']:
        print(f"\n  Tickers with known removal dates: {len(metadata['tickers_with_removal_dates'])}")
        print(f"  (These were later removed from S&P 500 - important for survivorship bias)")
    
    return cleaned, metadata


def save_tickers_to_csv(tickers: List[str], output_path: str, snapshot_date: str, metadata: dict = None):
    """
    Save the tickers to a CSV file.
    
    Format: Single row with all tickers comma-separated (matching input format)
    Also saves metadata about when this was generated.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as comma-separated string in a single cell (matching original format)
    tickers_str = ",".join(tickers)
    
    with open(output_path, 'w') as f:
        f.write(f'"{tickers_str}"')
    
    print(f"Saved {len(tickers)} tickers to: {output_path}")
    
    # Also save a more detailed version with metadata
    detailed_path = output_path.replace('.csv', '_detailed.csv')
    
    # Build detailed dataframe
    rows = []
    for ticker in tickers:
        row = {
            'ticker': ticker,
            'snapshot_date': snapshot_date,
            'generated_at': datetime.now().isoformat()
        }
        
        # Add removal date if known
        if metadata and 'tickers_with_removal_dates' in metadata:
            removal_dates = dict(metadata['tickers_with_removal_dates'])
            # The ticker in metadata is the clean version without dots->dashes conversion
            original_ticker = ticker.replace('-', '.')  # Reverse the conversion to look up
            if ticker in removal_dates:
                row['removal_date'] = removal_dates[ticker]
            elif original_ticker in removal_dates:
                row['removal_date'] = removal_dates[original_ticker]
            else:
                row['removal_date'] = None
        
        rows.append(row)
    
    df_out = pd.DataFrame(rows)
    df_out.to_csv(detailed_path, index=False)
    print(f"Saved detailed version to: {detailed_path}")


def main(snapshot_date: Optional[str] = None):
    """
    Main function to download and save S&P 500 tickers for a given date.
    """
    # Determine the snapshot date (priority: argument > env var > default)
    if snapshot_date is None:
        snapshot_date = os.environ.get('SP500_SNAPSHOT_DATE', DEFAULT_SNAPSHOT_DATE)
    
    print("=" * 60)
    print("S&P 500 HISTORICAL COMPONENTS DOWNLOADER")
    print("=" * 60)
    print(f"Snapshot date: {snapshot_date}")
    print(f"Historical data file: {HISTORICAL_COMPONENTS_FILE}")
    print(f"Output file: {OUTPUT_CSV_PATH}")
    print("=" * 60)
    
    # Load historical data
    print("\nLoading historical components...")
    df = load_historical_components(HISTORICAL_COMPONENTS_FILE)
    print(f"Loaded data from {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total records: {len(df)}")
    
    # Get tickers at snapshot date
    print(f"\nFinding S&P 500 components on {snapshot_date}...")
    tickers_raw = get_sp500_tickers_at_date(df, snapshot_date)
    print(f"Found {len(tickers_raw)} tickers (raw, with date suffixes)")
    
    # Show some examples of raw tickers
    tickers_with_dates = [t for t in tickers_raw if re.match(r'.*-\d{6}$', t)]
    if tickers_with_dates:
        print(f"\nExamples of tickers with removal date suffixes:")
        for t in tickers_with_dates[:5]:
            clean, date = parse_ticker_with_date(t)
            print(f"  {t} -> {clean} (removed {date})")
    
    # Clean for yfinance compatibility
    print("\nCleaning tickers for yfinance compatibility...")
    clean_tickers, metadata = clean_tickers_for_yfinance(tickers_raw)
    print(f"Unique tickers after cleaning: {len(clean_tickers)}")
    
    # Show first/last few tickers
    print(f"\nFirst 10 tickers: {clean_tickers[:10]}")
    print(f"Last 10 tickers: {clean_tickers[-10:]}")
    
    # Save to CSV
    print("\nSaving to CSV...")
    save_tickers_to_csv(clean_tickers, OUTPUT_CSV_PATH, snapshot_date, metadata)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    
    return clean_tickers


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download S&P 500 components for a historical date"
    )
    parser.add_argument(
        '--date', '-d',
        type=str,
        default=None,
        help='Snapshot date in YYYY-MM-DD format (default: 2016-02-09)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV path (default: data/sp500_tickers.csv)'
    )
    
    args = parser.parse_args()
    
    if args.output:
        OUTPUT_CSV_PATH = args.output
    
    tickers = main(args.date)