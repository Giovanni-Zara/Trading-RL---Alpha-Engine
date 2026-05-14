"""
Backtest: Test the best RL-discovered alpha on out-of-sample data.
Compare with baseline strategies on INTC and GE.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PIPELINE_DIR = os.path.dirname(_SCRIPT_DIR)
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from DSL import DSL

# === BEST ALPHA FROM TRAINING ===
BEST_ALPHA = "add(alpha_bollinger(5), delta(add(add(1, 1), add(0, alpha_pv_diverge(50))), 3))"
INITIAL_CAPITAL = 10000.0
TX_COST_BPS = 5.0
TICKERS = ['INTC', 'GE']


def evaluate_alpha(expr_str: str, df: pd.DataFrame, dsl: DSL) -> pd.Series:
    expr = dsl.parse(expr_str)
    return expr.evaluate(df)


def rank_normalize(alpha: pd.Series, window: int = 60) -> pd.Series:
    ranked = alpha.rolling(window, min_periods=30).rank(pct=True)
    return (ranked - 0.5) * 2.0


def backtest(signal: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Run backtest matching environment logic."""
    fwd_ret = (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
    cost = signal.diff().abs() * (TX_COST_BPS / 10000)
    net_ret = signal * fwd_ret - cost

    result = pd.DataFrame(index=df.index)
    result['date'] = pd.to_datetime(df['Date'])
    result['net_return'] = net_ret
    result['equity'] = INITIAL_CAPITAL * (1 + net_ret.fillna(0)).cumprod()
    return result.dropna(subset=['net_return'])


def metrics(result: pd.DataFrame) -> Dict[str, float]:
    r = result['net_return'].dropna()
    if len(r) == 0 or r.std() == 0:
        return {'sharpe': 0, 'total_return': 0, 'max_dd': 0, 'win_rate': 0, 'final': INITIAL_CAPITAL}
    eq = result['equity']
    dd = (eq - eq.cummax()) / eq.cummax()
    return {
        'sharpe': r.mean() / r.std() * np.sqrt(252),
        'total_return': eq.iloc[-1] / eq.iloc[0] - 1,
        'max_dd': dd.min(),
        'win_rate': (r > 0).mean(),
        'final': eq.iloc[-1],
    }


# === BASELINES ===
def baseline_buy_hold(df):
    ret = df['Close'].pct_change()
    eq = INITIAL_CAPITAL * (1 + ret.fillna(0)).cumprod()
    return pd.DataFrame({'date': pd.to_datetime(df['Date']), 'net_return': ret, 'equity': eq}, index=df.index)

def baseline_sma(df, short=20, long=50):
    sig = (df['Close'].rolling(short).mean() > df['Close'].rolling(long).mean()).astype(float)
    fwd = df['Close'].pct_change().shift(-1)
    ret = sig.shift(1) * fwd
    eq = INITIAL_CAPITAL * (1 + ret.fillna(0)).cumprod()
    return pd.DataFrame({'date': pd.to_datetime(df['Date']), 'net_return': ret, 'equity': eq}, index=df.index)

def baseline_mean_rev(df, w=20):
    ma, std = df['Close'].rolling(w).mean(), df['Close'].rolling(w).std()
    sig = (-(df['Close'] - ma) / (std + 1e-8)).clip(-1, 1)
    fwd = (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
    ret = sig.shift(1) * fwd
    cost = sig.shift(1).diff().abs() * (TX_COST_BPS / 10000)
    net = ret - cost
    eq = INITIAL_CAPITAL * (1 + net.fillna(0)).cumprod()
    return pd.DataFrame({'date': pd.to_datetime(df['Date']), 'net_return': net, 'equity': eq}, index=df.index)

def baseline_random(df, seed=42):
    rng = np.random.RandomState(seed)
    sig = pd.Series(rng.uniform(-1, 1, len(df)), index=df.index)
    fwd = (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
    cost = sig.diff().abs() * (TX_COST_BPS / 10000)
    net = sig * fwd - cost
    eq = INITIAL_CAPITAL * (1 + net.fillna(0)).cumprod()
    return pd.DataFrame({'date': pd.to_datetime(df['Date']), 'net_return': net, 'equity': eq}, index=df.index)

def baseline_momentum(df, lookback=20):
    sig = np.sign(df['Close'].pct_change(lookback))
    fwd = (df['Close'].shift(-1) - df['Open'].shift(-1)) / df['Open'].shift(-1)
    cost = sig.shift(1).diff().abs() * (TX_COST_BPS / 10000)
    net = sig.shift(1) * fwd - cost
    eq = INITIAL_CAPITAL * (1 + net.fillna(0)).cumprod()
    return pd.DataFrame({'date': pd.to_datetime(df['Date']), 'net_return': net, 'equity': eq}, index=df.index)


# =============================================================================
if __name__ == "__main__":

    # --- Load data ---
    data_path = os.path.join(_PIPELINE_DIR, "data", "test", "TEST_ohlcv_data.csv")
    print(f"Loading: {data_path}")
    data = pd.read_csv(data_path)
    data['Date'] = pd.to_datetime(data['Date'])
    test_data = data[data['Date'] >= '2022-01-01'].copy()
    print(f"Period: {test_data['Date'].min().date()} → {test_data['Date'].max().date()}")

    available = test_data['Ticker'].unique()
    tickers = [t for t in TICKERS if t in available]
    if not tickers:
        print(f"ERROR: None of {TICKERS} found in data. Available: {list(available)}")
        sys.exit(1)
    print(f"Tickers: {tickers}")

    dsl = DSL()

    # ── Step 1: Backtest alpha on INTC & GE ──
    print(f"\n{'='*80}")
    print(f"ALPHA: {BEST_ALPHA}")
    print(f"{'='*80}\n")

    ticker_results = {}
    for ticker in tickers:
        df = test_data[test_data['Ticker'] == ticker].sort_values('Date').reset_index(drop=True)
        if len(df) < 120:
            print(f"  ✗ {ticker}: Skipped (only {len(df)} rows)")
            continue
        try:
            alpha = rank_normalize(evaluate_alpha(BEST_ALPHA, df, dsl))
            res = backtest(alpha, df)
            m = metrics(res)
            ticker_results[ticker] = {'result': res, 'metrics': m, 'df': df}
            s = "✓" if m['sharpe'] > 0 else "✗"
            print(f"  {s} {ticker:6s}  Sharpe={m['sharpe']:+.3f}  Return={m['total_return']:+.1%}  "
                  f"Final=${m['final']:,.0f}  MaxDD={m['max_dd']:.1%}  WinRate={m['win_rate']:.1%}")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    if not ticker_results:
        print("No valid results. Exiting.")
        sys.exit(1)

    # Best ticker
    best_ticker = max(ticker_results, key=lambda t: ticker_results[t]['metrics']['sharpe'])
    bm = ticker_results[best_ticker]['metrics']
    print(f"\n{'─'*80}")
    print(f"★ BEST TICKER: {best_ticker}")
    print(f"  Sharpe:       {bm['sharpe']:+.3f}")
    print(f"  Total Return: {bm['total_return']:+.1%}")
    print(f"  Final Equity: ${bm['final']:,.0f}")
    print(f"  Max Drawdown: {bm['max_dd']:.1%}")
    print(f"  Win Rate:     {bm['win_rate']:.1%}")
    print(f"{'─'*80}")

    # ── Step 2: Compute baselines for both tickers ──
    print(f"\n{'='*80}")
    print("BASELINES")
    print(f"{'='*80}")

    baseline_fns = {
        'Buy & Hold':           baseline_buy_hold,
        'SMA Crossover (20/50)': baseline_sma,
        'Mean Reversion (20)':  baseline_mean_rev,
        'Random Trading':       baseline_random,
        'Momentum (20d)':       baseline_momentum,
    }

    # Store equity series per strategy per ticker
    baseline_equities = {name: [] for name in baseline_fns}

    for ticker in ticker_results:
        df = ticker_results[ticker]['df']
        print(f"\n  {ticker}:")
        for name, fn in baseline_fns.items():
            br = fn(df)
            bm_b = metrics(br)
            baseline_equities[name].append(br.set_index('date')['equity'].dropna())
            print(f"    {name:25s}  Sharpe={bm_b['sharpe']:+.3f}  Return={bm_b['total_return']:+.1%}  "
                  f"Final=${bm_b['final']:,.0f}")

    # ── Step 3: Average equity curves and plot ──
    def avg_equity(eq_list):
        combined = pd.concat(eq_list, axis=1).sort_index().ffill()
        return combined.mean(axis=1).dropna()

    # Alpha average equity
    alpha_eq_list = [ticker_results[t]['result'].set_index('date')['equity'] for t in ticker_results]
    avg_alpha = avg_equity(alpha_eq_list)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Alpha
    ax.plot(avg_alpha.index, avg_alpha.values, color='#e6194B', linewidth=2.5,
            label='RL Alpha (best)', zorder=5)

    # Baselines
    colors = ['#808080', '#4363d8', '#911eb4', '#a9a9a9', '#469990']
    styles = ['--', '-.', ':', '--', '-.']
    for i, (name, eq_list) in enumerate(baseline_equities.items()):
        if eq_list:
            avg_b = avg_equity(eq_list)
            ax.plot(avg_b.index, avg_b.values, color=colors[i], ls=styles[i],
                    lw=1.5, label=name)

    ax.axhline(INITIAL_CAPITAL, color='k', lw=0.7, alpha=0.3)
    ax.set_title(f'Average Portfolio Equity: RL Agent vs Baselines\n'
                 f'${INITIAL_CAPITAL:,.0f} initial · INTC & GE · '
                 f'Best: {best_ticker} (Sharpe {ticker_results[best_ticker]["metrics"]["sharpe"]:+.2f})',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(_SCRIPT_DIR, "backtest_equity_curves.png")
    fig.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")
    plt.show()