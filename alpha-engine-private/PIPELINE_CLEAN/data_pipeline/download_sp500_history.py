"""
Download S&P 500 Historical Components Data

Downloads the historical S&P 500 components file from GitHub.
"""

import os
import urllib.request

# URL to the historical components file
GITHUB_URL = "https://raw.githubusercontent.com/fja05680/sp500/refs/heads/master/S%26P%20500%20Historical%20Components%20%26%20Changes(01-17-2026).csv"

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "S&P 500 Historical Components & Changes.csv"
)


def download_file(url: str, output_path: str):
    """Download a file from URL to local path."""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False


def main():
    print("=" * 60)
    print("DOWNLOADING S&P 500 HISTORICAL COMPONENTS")
    print("=" * 60)
    
    # Try main URL first
    if download_file(GITHUB_URL, OUTPUT_PATH):
        return

    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("Please download the file manually from:")
    print("  https://github.com/fja05680/sp500")
    print(f"\nAnd save it as:\n  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()