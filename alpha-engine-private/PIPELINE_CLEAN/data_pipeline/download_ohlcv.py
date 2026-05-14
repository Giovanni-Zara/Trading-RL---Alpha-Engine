"""
OHLCV Data Downloader for S&P 500 Stocks

Downloads daily OHLCV data for all tickers in the sp500_tickers.csv file
using yfinance library.

Usage:
    python download_ohlcv.py
    python download_ohlcv.py --start 2016-02-09 --end 2022-02-09
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import warnings
import logging

# Suppress yfinance warnings
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

try:
    import yfinance as yf
except ImportError:
    print("yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default date range
DEFAULT_START_DATE = "2016-02-09"
DEFAULT_END_DATE = "2022-02-09"

# Paths
DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKERS_CSV = os.path.join(DATA_DIR, "data", "sp500_tickers.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "PIPELINE_CLEAN", "data", "ohlcv_data.csv")

# Download settings
SLEEP_BETWEEN_TICKERS = 0.1  # Seconds between individual downloads
SLEEP_BETWEEN_BATCHES = 2    # Seconds between batches of 50


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_tickers(filepath: str) -> List[str]:
    """Load tickers from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Tickers file not found: {filepath}\n"
            f"Run sp500_tickers.py first to generate it."
        )
    
    with open(filepath, 'r') as f:
        content = f.read().strip()
    
    content = content.strip('"\'')
    tickers = [t.strip() for t in content.split(',') if t.strip()]
    
    return tickers


def download_single_ticker(
    ticker: str,
    start_date: str,
    end_date: str,
    verbose: bool = False
) -> Tuple[pd.DataFrame, bool, str]:
    """
    Download data for a single ticker.
    
    Returns:
        (DataFrame, success_flag, error_message)
    """
    try:
        # Use Ticker object directly for better control
        tk = yf.Ticker(ticker)
        data = tk.history(
            start=start_date,
            end=end_date,
            auto_adjust=True,
            actions=False
        )
        
        if data is None or data.empty:
            return pd.DataFrame(), False, "No data returned"
        
        # Reset index to get Date as column
        data = data.reset_index()
        
        # Standardize column names
        data = data.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Add ticker column
        data['Ticker'] = ticker
        
        # Select only needed columns (some tickers have extra columns)
        cols = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']
        data = data[[c for c in cols if c in data.columns]]
        
        # Ensure we have the essential columns
        if 'Close' not in data.columns or len(data) == 0:
            return pd.DataFrame(), False, "Missing Close column or empty data"
        
        return data, True, ""
        
    except Exception as e:
        error_msg = str(e)
        # Simplify common error messages
        if "possibly delisted" in error_msg.lower():
            error_msg = "Delisted"
        elif "no timezone" in error_msg.lower():
            error_msg = "No data (delisted)"
        elif "no price data" in error_msg.lower():
            error_msg = "No price data"
        return pd.DataFrame(), False, error_msg


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Validate and clean the downloaded data."""
    report = {
        'total_rows': len(df),
        'tickers': df['Ticker'].nunique() if len(df) > 0 else 0,
        'date_range': (df['Date'].min(), df['Date'].max()) if len(df) > 0 else (None, None),
        'missing_values': {},
        'removed_rows': 0
    }
    
    if len(df) == 0:
        return df, report
    
    # Check missing values per column
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            report['missing_values'][col] = df[col].isna().sum()
    
    # Remove rows with missing Close price
    rows_before = len(df)
    df = df.dropna(subset=['Close'])
    report['removed_rows'] = rows_before - len(df)
    
    # Remove rows with zero or negative prices
    df = df[(df['Close'] > 0) & (df['Open'] > 0)]
    
    # Forward fill missing OHLV values within each ticker
    df = df.sort_values(['Ticker', 'Date'])
    for col in ['Open', 'High', 'Low', 'Volume']:
        if col in df.columns:
            df[col] = df.groupby('Ticker')[col].ffill()
    
    return df, report


def check_alignment(df: pd.DataFrame) -> dict:
    """Check that all tickers have data for the same dates."""
    if len(df) == 0:
        return {'total_unique_dates': 0, 'common_dates': 0, 'tickers_with_gaps': []}
    
    ticker_dates = df.groupby('Ticker')['Date'].apply(set)
    
    all_dates = set()
    for dates in ticker_dates:
        if len(all_dates) == 0:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)
    
    report = {
        'total_unique_dates': df['Date'].nunique(),
        'common_dates': len(all_dates),
        'tickers_with_gaps': []
    }
    
    if len(ticker_dates) > 0:
        median_dates = ticker_dates.apply(len).median()
        for ticker, dates in ticker_dates.items():
            if len(dates) < median_dates * 0.9:
                report['tickers_with_gaps'].append((ticker, len(dates)))
    
    return report


# =============================================================================
# MAIN DOWNLOAD FUNCTION
# =============================================================================

def download_ohlcv(
    tickers: List[str],
    start_date: str,
    end_date: str,
    output_path: str,
    resume: bool = True
) -> pd.DataFrame:
    """
    Download OHLCV data for all tickers and save to CSV.
    
    Parameters:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Path to save CSV
        resume: If True, skip tickers already downloaded (checks partial file)
    """
    print("=" * 60)
    print("OHLCV DATA DOWNLOADER")
    print("=" * 60)
    print(f"Tickers to download: {len(tickers)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Check for existing partial download
    already_downloaded = set()
    existing_data = []
    
    if resume and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            already_downloaded = set(existing_df['Ticker'].unique())
            existing_data.append(existing_df)
            print(f"\nResuming: Found {len(already_downloaded)} already downloaded tickers")
        except Exception as e:
            print(f"Could not read existing file: {e}")
    
    # Filter out already downloaded
    tickers_to_download = [t for t in tickers if t not in already_downloaded]
    print(f"Tickers remaining: {len(tickers_to_download)}")
    
    if len(tickers_to_download) == 0:
        print("All tickers already downloaded!")
        if existing_data:
            return existing_data[0]
        return pd.DataFrame()
    
    # Track results
    successful_tickers = []
    failed_tickers = {}  # ticker -> error message
    all_data = existing_data.copy()
    
    # Download tickers
    print(f"\nDownloading {len(tickers_to_download)} tickers...")
    print("-" * 40)
    
    for i, ticker in enumerate(tickers_to_download):
        # Progress indicator
        if (i + 1) % 25 == 0 or i == 0:
            pct = (i + 1) / len(tickers_to_download) * 100
            print(f"Progress: {i + 1}/{len(tickers_to_download)} ({pct:.1f}%) - "
                  f"Success: {len(successful_tickers)}, Failed: {len(failed_tickers)}")
        
        # Download
        data, success, error = download_single_ticker(ticker, start_date, end_date)
        
        if success and len(data) > 0:
            all_data.append(data)
            successful_tickers.append(ticker)
        else:
            failed_tickers[ticker] = error
        
        # Rate limiting
        time.sleep(SLEEP_BETWEEN_TICKERS)
        
        # Longer pause every 50 tickers
        if (i + 1) % 50 == 0:
            time.sleep(SLEEP_BETWEEN_BATCHES)
        
        # Save checkpoint every 100 tickers
        if (i + 1) % 100 == 0 and all_data:
            print(f"  Saving checkpoint...")
            checkpoint_df = pd.concat(all_data, ignore_index=True)
            checkpoint_df = checkpoint_df.sort_values(['Ticker', 'Date'])
            checkpoint_df['Date'] = pd.to_datetime(checkpoint_df['Date']).dt.strftime('%Y-%m-%d')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            checkpoint_df.to_csv(output_path, index=False)
    
    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    total_successful = len(successful_tickers) + len(already_downloaded)
    print(f"Successfully downloaded: {len(successful_tickers)} (new) + {len(already_downloaded)} (existing) = {total_successful}")
    print(f"Failed: {len(failed_tickers)}")
    
    # Group failures by error type
    if failed_tickers:
        error_groups = {}
        for ticker, error in failed_tickers.items():
            if error not in error_groups:
                error_groups[error] = []
            error_groups[error].append(ticker)
        
        print(f"\n⚠️  SURVIVORSHIP BIAS WARNING:")
        print(f"The following {len(failed_tickers)} tickers could not be downloaded.")
        print(f"These are likely delisted/acquired companies.\n")
        
        for error, tickers_list in sorted(error_groups.items(), key=lambda x: -len(x[1])):
            print(f"  {error}: {len(tickers_list)} tickers")
            if len(tickers_list) <= 5:
                print(f"    {tickers_list}")
            else:
                print(f"    {tickers_list[:5]} + {len(tickers_list)-5} more")
        
        # Save failed tickers
        failed_path = output_path.replace('.csv', '_failed_tickers.txt')
        with open(failed_path, 'w') as f:
            f.write("# Tickers that could not be downloaded\n")
            f.write("# These are likely delisted/acquired/bankrupted companies\n")
            f.write("# Their absence introduces survivorship bias\n\n")
            for ticker in sorted(failed_tickers.keys()):
                f.write(f"{ticker}: {failed_tickers[ticker]}\n")
        print(f"\nFailed tickers saved to: {failed_path}")
    
    # Combine and save final data
    if not all_data:
        print("ERROR: No data downloaded!")
        return pd.DataFrame()
    
    print("\nCombining data...")
    df = pd.concat(all_data, ignore_index=True)
    
    # Remove duplicates (in case of resume)
    df = df.drop_duplicates(subset=['Date', 'Ticker'])
    
    print(f"Total rows: {len(df)}")
    
    # Validate and clean
    print("\nValidating data...")
    df, validation_report = validate_data(df)
    print(f"  Unique tickers: {validation_report['tickers']}")
    print(f"  Date range: {validation_report['date_range']}")
    print(f"  Removed rows (missing Close): {validation_report['removed_rows']}")
    
    # Check alignment
    print("\nChecking date alignment...")
    alignment_report = check_alignment(df)
    print(f"  Total unique trading days: {alignment_report['total_unique_dates']}")
    print(f"  Common dates (all tickers): {alignment_report['common_dates']}")
    if alignment_report['tickers_with_gaps']:
        print(f"  Tickers with significant gaps: {len(alignment_report['tickers_with_gaps'])}")
    
    # Sort and save
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    
    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Final dataset: {len(df):,} rows, {df['Ticker'].nunique()} tickers")
    print(f"Saved to: {output_path}")
    
    # Calculate survivorship bias metric
    bias_pct = len(failed_tickers) / len(tickers) * 100
    print(f"\n📊 Survivorship bias: {bias_pct:.1f}% of tickers unavailable")
    if bias_pct > 10:
        print("   ⚠️  This is significant - backtest results may be optimistic")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download OHLCV data for S&P 500")
    parser.add_argument('--start', '-s', type=str, default=DEFAULT_START_DATE,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', type=str, default=DEFAULT_END_DATE,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--tickers', '-t', type=str, default=TICKERS_CSV,
                        help='Path to tickers CSV file')
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_CSV,
                        help='Output CSV path')
    parser.add_argument('--no-resume', action='store_true',
                        help='Start fresh, ignore existing partial download')
    
    args = parser.parse_args()
    
    # Load tickers
    print(f"Loading tickers from {args.tickers}...")
    tickers = load_tickers(args.tickers)
    print(f"Loaded {len(tickers)} tickers")
    
    # Download
    df = download_ohlcv(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        output_path=args.output,
        resume=not args.no_resume
    )