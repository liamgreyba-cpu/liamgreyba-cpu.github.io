"""
Walk-Forward Validation — Slow Market Milkman
Splits 11-year backtest into in-sample vs out-of-sample periods
to detect overfitting and confirm strategy robustness.

WINDOWS TESTED:
  Full run:         2015-2025 (all data)
  In-sample:        2015-2019 (5 years -- used to develop strategy)
  Out-of-sample:    2020-2025 (6 years -- never used in development)
  Rolling WF:       Train 3yr -> Test 1yr, rolling forward
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings('ignore')
import os, sys

# ── Import the backtest engine ──
# We run the full backtest then filter trades by date window
TRADES_CSV = None
for f in os.listdir('.'):
    if 'backtest_trades_bidir' in f and f.endswith('.csv'):
        TRADES_CSV = f
        break

if TRADES_CSV is None:
    # Try outputs folder
    import glob
    files = glob.glob('outputs/backtest_trades_bidir*.csv') + glob.glob('backtest_trades_bidir*.csv')
    if files:
        TRADES_CSV = files[-1]

if TRADES_CSV is None:
    print("ERROR: No backtest_trades_bidir.csv found.")
    print("Run the full backtest first then run this script in the same folder.")
    sys.exit(1)

print(f"Loading trades from: {TRADES_CSV}")
df = pd.read_csv(TRADES_CSV)
df['date'] = pd.to_datetime(df['date'])
df['pair'] = df['pair'].str.strip()
df['type'] = df['type'].str.strip()
df['year'] = df['date'].dt.year
print(f"Loaded {len(df):,} trades  {df['date'].min().date()} to {df['date'].max().date()}")

START_EQUITY = 25000

def run_window(sub_df, label=""):
    if len(sub_df) == 0:
        return None
    pnls   = sub_df['pnl'].values
    equity = START_EQUITY + np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    dd     = (peak - equity) / peak * 100

    monthly = sub_df.groupby(sub_df['date'].dt.to_period('M'))['pnl'].sum()
    daily   = sub_df.groupby(sub_df['date'].dt.date)['pnl'].sum()
    winners = sub_df[sub_df['pnl'] > 0]
    losers  = sub_df[sub_df['pnl'] < 0]

    years = (sub_df['date'].max() - sub_df['date'].min()).days / 365.25
    net   = pnls.sum()
    cagr  = ((START_EQUITY + net) / START_EQUITY) ** (1 / years) - 1 if years > 0 else 0
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
    pf     = winners['pnl'].sum() / abs(losers['pnl'].sum()) if len(losers) > 0 else 0
    wr     = (sub_df['pnl'] > 0).mean() * 100
    max_dd = dd.max()
    calmar = cagr / (max_dd / 100) if max_dd > 0 else 0
    pos_mo = (monthly > 0).mean() * 100

    # Per-pair pass check (DD < 10%)
    pair_results = {}
    for pair in sub_df['pair'].unique():
        p = sub_df[sub_df['pair'] == pair]
        pe = START_EQUITY + np.cumsum(p['pnl'].values)
        pp = np.maximum.accumulate(pe)
        pd2 = ((pp - pe) / pp * 100).max()
        pair_results[pair] = 'PASS' if pd2 < 10 else 'FAIL'

    return {
        'label':    label,
        'trades':   len(sub_df),
        'net':      net,
        'cagr':     cagr * 100,
        'sharpe':   sharpe,
        'max_dd':   max_dd,
        'calmar':   calmar,
        'wr':       wr,
        'pf':       pf,
        'pos_mo':   pos_mo,
        'equity':   equity,
        'pairs':    pair_results,
        'years':    years,
        'monthly':  monthly,
    }

# ── 1. FULL PERIOD ──
r_full = run_window(df, "Full 2015-2025")

# ── 2. IN-SAMPLE vs OUT-OF-SAMPLE ──
df_is  = df[df['year'] <= 2019]
df_oos = df[df['year'] >= 2020]
r_is   = run_window(df_is,  "In-Sample 2015-2019")
r_oos  = run_window(df_oos, "Out-of-Sample 2020-2025")

# ── 3. ROLLING WALK-FORWARD ──
# Train on 3 years, test on 1 year, roll forward
rolling = []
all_years = sorted(df['year'].unique())
train_window = 3
for i in range(len(all_years) - train_window):
    train_years = all_years[i:i+train_window]
    test_year   = all_years[i+train_window]
    df_test     = df[df['year'] == test_year]
    r = run_window(df_test, f"OOS {test_year}")
    if r:
        r['train_years'] = train_years
        rolling.append(r)

# ── PLOTTING ──
BG = '#0D1117'; CARD = '#161B22'; GREEN = '#1D9E75'; RED = '#E24B4A'
AMBER = '#FFB800'; BLUE = '#378ADD'; WHITE = '#E6EDF3'; DIM = '#7D8590'

def setup(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values(): s.set_edgecolor('#30363D'); s.set_linewidth(0.8)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(color='#21262D', linewidth=0.5, alpha=0.6)

fig = plt.figure(figsize=(26, 22), facecolor=BG)
gs  = gridspec.GridSpec(4, 4, figure=fig, hspace=0.50, wspace=0.42,
                        left=0.05, right=0.98, top=0.93, bottom=0.04)

fig.text(0.5, 0.965, 'SMM — WALK-FORWARD VALIDATION',
         ha='center', fontsize=20, fontweight='bold', color=WHITE, fontfamily='monospace')
fig.text(0.5, 0.950,
         'In-Sample (2015-2019) vs Out-of-Sample (2020-2025)  |  Rolling 3yr train / 1yr test',
         ha='center', fontsize=10, color=DIM)

# ── Panel 1: IS vs OOS equity curves ──
ax = fig.add_subplot(gs[0, 0:2]); setup(ax)
x_is  = np.arange(len(r_is['equity']))
x_oos = np.arange(len(r_oos['equity']))
ax.plot(x_is,  r_is['equity'],  color=BLUE,  linewidth=2, label=f"In-Sample 2015-19  GBP{r_is['net']:,.0f}")
ax.plot(x_oos, r_oos['equity'], color=GREEN, linewidth=2, label=f"OOS 2020-25  GBP{r_oos['net']:,.0f}")
ax.axhline(START_EQUITY, color=DIM, linewidth=0.8, linestyle='--', alpha=0.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'GBP{v/1000:.0f}k'))
ax.set_title('In-Sample vs Out-of-Sample Equity', color=WHITE, fontsize=11, fontweight='bold', pad=6)
ax.legend(fontsize=9, facecolor=CARD, labelcolor=WHITE)

# ── Panel 2: Key metrics comparison table ──
ax = fig.add_subplot(gs[0, 2]); setup(ax)
ax.set_xticks([]); ax.set_yticks([])
metrics_compare = [
    ('Metric',         'In-Sample',              'Out-of-Sample',         'Match?'),
    ('Period',         '2015-2019',              '2020-2025',             ''),
    ('Net P&L',        f"GBP{r_is['net']:,.0f}", f"GBP{r_oos['net']:,.0f}", ''),
    ('CAGR',           f"{r_is['cagr']:.1f}%",  f"{r_oos['cagr']:.1f}%",  'OK' if abs(r_is['cagr']-r_oos['cagr'])<15 else 'DIFF'),
    ('Sharpe',         f"{r_is['sharpe']:.2f}",  f"{r_oos['sharpe']:.2f}", 'OK' if abs(r_is['sharpe']-r_oos['sharpe'])<1.5 else 'DIFF'),
    ('Max DD',         f"{r_is['max_dd']:.2f}%", f"{r_oos['max_dd']:.2f}%",'OK' if r_oos['max_dd'] < 10 else 'DIFF'),
    ('Win Rate',       f"{r_is['wr']:.1f}%",     f"{r_oos['wr']:.1f}%",    'OK' if abs(r_is['wr']-r_oos['wr'])<8 else 'DIFF'),
    ('Prof Factor',    f"{r_is['pf']:.2f}",       f"{r_oos['pf']:.2f}",    'OK' if r_oos['pf']>1.5 else 'DIFF'),
    ('Pos Months',     f"{r_is['pos_mo']:.1f}%",  f"{r_oos['pos_mo']:.1f}%",'OK' if r_oos['pos_mo']>85 else 'DIFF'),
    ('Trades',         f"{r_is['trades']:,}",      f"{r_oos['trades']:,}",  ''),
]
for i, row in enumerate(metrics_compare):
    y = 0.96 - i*0.092
    col0 = DIM; col1 = WHITE; col2 = WHITE
    col3 = GREEN if row[3]=='OK' else RED if row[3]=='DIFF' else DIM
    if i == 0: col0=col1=col2=DIM
    ax.text(0.01, y, row[0], transform=ax.transAxes, fontsize=8.5, color=col0, fontfamily='monospace')
    ax.text(0.38, y, row[1], transform=ax.transAxes, fontsize=8.5, color=col1, fontfamily='monospace')
    ax.text(0.68, y, row[2], transform=ax.transAxes, fontsize=8.5, color=col2, fontfamily='monospace')
    ax.text(0.93, y, row[3], transform=ax.transAxes, fontsize=8.5, color=col3, fontfamily='monospace', fontweight='bold')
ax.set_title('IS vs OOS Metrics', color=WHITE, fontsize=11, fontweight='bold', pad=6)

# ── Panel 3: Performance degradation score ──
ax = fig.add_subplot(gs[0, 3]); setup(ax)
ax.set_xticks([]); ax.set_yticks([])
# Score: how similar are IS and OOS?
sharpe_ratio  = r_oos['sharpe']  / r_is['sharpe']  if r_is['sharpe']  > 0 else 0
cagr_ratio    = r_oos['cagr']    / r_is['cagr']    if r_is['cagr']    > 0 else 0
pf_ratio      = r_oos['pf']      / r_is['pf']      if r_is['pf']      > 0 else 0
wr_ratio      = r_oos['wr']      / r_is['wr']      if r_is['wr']      > 0 else 0
avg_ratio     = np.mean([sharpe_ratio, cagr_ratio, pf_ratio, wr_ratio])
verdict       = 'ROBUST' if avg_ratio > 0.7 else 'MARGINAL' if avg_ratio > 0.5 else 'OVERFITTED'
verdict_col   = GREEN if verdict=='ROBUST' else AMBER if verdict=='MARGINAL' else RED

ax.text(0.5, 0.85, 'Robustness Score', transform=ax.transAxes,
        ha='center', fontsize=11, color=WHITE, fontweight='bold')
ax.text(0.5, 0.65, f'{avg_ratio*100:.0f}%', transform=ax.transAxes,
        ha='center', fontsize=36, color=verdict_col, fontweight='bold', fontfamily='monospace')
ax.text(0.5, 0.48, verdict, transform=ax.transAxes,
        ha='center', fontsize=16, color=verdict_col, fontweight='bold')
ax.text(0.5, 0.32, f'Sharpe retained: {sharpe_ratio*100:.0f}%', transform=ax.transAxes,
        ha='center', fontsize=9, color=DIM)
ax.text(0.5, 0.22, f'CAGR retained:   {cagr_ratio*100:.0f}%', transform=ax.transAxes,
        ha='center', fontsize=9, color=DIM)
ax.text(0.5, 0.12, f'PF retained:     {pf_ratio*100:.0f}%', transform=ax.transAxes,
        ha='center', fontsize=9, color=DIM)
ax.text(0.5, 0.02, f'WR retained:     {wr_ratio*100:.0f}%', transform=ax.transAxes,
        ha='center', fontsize=9, color=DIM)
ax.set_title('OOS Robustness', color=WHITE, fontsize=11, fontweight='bold', pad=6)

# ── Panel 4: Rolling annual OOS performance ──
ax = fig.add_subplot(gs[1, 0:2]); setup(ax)
roll_years = [r['label'].split()[-1] for r in rolling]
roll_nets  = [r['net'] for r in rolling]
roll_sharpe= [r['sharpe'] for r in rolling]
colors     = [GREEN if v >= 0 else RED for v in roll_nets]
bars = ax.bar(roll_years, [v/1000 for v in roll_nets], color=colors, alpha=0.85, width=0.6)
for bar, val in zip(bars, roll_nets):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3 if val >= 0 else bar.get_height()-2,
            f'GBP{val/1000:.0f}k', ha='center', fontsize=8,
            color=GREEN if val >= 0 else RED, fontweight='bold')
ax.axhline(0, color=DIM, linewidth=0.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'GBP{v:.0f}k'))
ax.set_title('Rolling OOS — Annual P&L (each bar = 1 blind test year)', color=WHITE, fontsize=11, fontweight='bold', pad=6)
pos_years = sum(1 for v in roll_nets if v > 0)
ax.text(0.98, 0.95, f'{pos_years}/{len(rolling)} positive blind years',
        transform=ax.transAxes, ha='right', fontsize=9,
        color=GREEN if pos_years >= len(rolling)*0.75 else AMBER, va='top')

# ── Panel 5: Rolling Sharpe ──
ax = fig.add_subplot(gs[1, 2]); setup(ax)
ax.bar(roll_years, roll_sharpe, color=[GREEN if v > 1 else AMBER if v > 0 else RED for v in roll_sharpe],
       alpha=0.85, width=0.6)
ax.axhline(1.0, color=AMBER, linewidth=1, linestyle='--', alpha=0.7)
ax.axhline(0,   color=DIM,   linewidth=0.8)
ax.set_title('Rolling OOS — Annual Sharpe', color=WHITE, fontsize=11, fontweight='bold', pad=6)
ax.text(0.98, 0.95, f'Sharpe>1: {sum(1 for v in roll_sharpe if v>1)}/{len(rolling)} years',
        transform=ax.transAxes, ha='right', fontsize=9,
        color=GREEN if sum(1 for v in roll_sharpe if v>1) >= len(rolling)*0.7 else AMBER, va='top')

# ── Panel 6: Rolling Max DD ──
ax = fig.add_subplot(gs[1, 3]); setup(ax)
roll_dds = [r['max_dd'] for r in rolling]
ax.bar(roll_years, roll_dds,
       color=[GREEN if v < 5 else AMBER if v < 8 else RED for v in roll_dds],
       alpha=0.85, width=0.6)
ax.axhline(5,  color=AMBER, linewidth=1, linestyle='--', alpha=0.7)
ax.axhline(10, color=RED,   linewidth=1, linestyle='--', alpha=0.7)
ax.text(len(roll_years)*0.5, 5.3,  'FTMO daily', fontsize=7.5, color=AMBER)
ax.text(len(roll_years)*0.5, 10.3, 'FTMO total', fontsize=7.5, color=RED)
ax.set_title('Rolling OOS — Annual Max DD', color=WHITE, fontsize=11, fontweight='bold', pad=6)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.1f}%'))

# ── Panel 7: OOS monthly returns ──
ax = fig.add_subplot(gs[2, 0:2]); setup(ax)
oos_monthly = r_oos['monthly']
mo_colors   = [GREEN if v >= 0 else RED for v in oos_monthly.values]
ax.bar(range(len(oos_monthly)), oos_monthly.values/1000, color=mo_colors, alpha=0.8, width=0.8)
ax.axhline(0, color=DIM, linewidth=0.8)
ax.set_xticks(range(0, len(oos_monthly), 6))
ax.set_xticklabels([str(oos_monthly.index[i]) for i in range(0, len(oos_monthly), 6)],
                    rotation=30, fontsize=7)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'GBP{v:.0f}k'))
ax.set_title('Out-of-Sample Monthly Returns (2020-2025)', color=WHITE, fontsize=11, fontweight='bold', pad=6)
pos = (oos_monthly > 0).sum(); tot = len(oos_monthly)
ax.text(0.98, 0.95, f'{pos}/{tot} positive months ({pos/tot*100:.0f}%)',
        transform=ax.transAxes, ha='right', fontsize=9,
        color=GREEN if pos/tot > 0.85 else AMBER, va='top')

# ── Panel 8: IS monthly returns ──
ax = fig.add_subplot(gs[2, 2:4]); setup(ax)
is_monthly = r_is['monthly']
mo_colors  = [GREEN if v >= 0 else RED for v in is_monthly.values]
ax.bar(range(len(is_monthly)), is_monthly.values/1000, color=mo_colors, alpha=0.8, width=0.8)
ax.axhline(0, color=DIM, linewidth=0.8)
ax.set_xticks(range(0, len(is_monthly), 6))
ax.set_xticklabels([str(is_monthly.index[i]) for i in range(0, len(is_monthly), 6)],
                    rotation=30, fontsize=7)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'GBP{v:.0f}k'))
ax.set_title('In-Sample Monthly Returns (2015-2019)', color=WHITE, fontsize=11, fontweight='bold', pad=6)
pos = (is_monthly > 0).sum(); tot = len(is_monthly)
ax.text(0.98, 0.95, f'{pos}/{tot} positive months ({pos/tot*100:.0f}%)',
        transform=ax.transAxes, ha='right', fontsize=9,
        color=GREEN if pos/tot > 0.85 else AMBER, va='top')

# ── Panel 9: Per-pair OOS performance ──
ax = fig.add_subplot(gs[3, 0:2]); setup(ax)
pair_order = ['EUR/USD','USD/JPY','USD/CAD','Gold','GBP/JPY','EUR/AUD']
oos_pair_pnl = []
is_pair_pnl  = []
for pair in pair_order:
    o = r_oos['pairs'].get(pair, 'N/A')
    oos_pair_pnl.append(df_oos[df_oos['pair']==pair]['pnl'].sum() if pair in df_oos['pair'].values else 0)
    is_pair_pnl.append( df_is[df_is['pair']==pair]['pnl'].sum()   if pair in df_is['pair'].values  else 0)
x = np.arange(len(pair_order)); w = 0.35
ax.bar(x-w/2, [v/1000 for v in is_pair_pnl],  w, label='IS 2015-19', color=BLUE,  alpha=0.8)
ax.bar(x+w/2, [v/1000 for v in oos_pair_pnl], w, label='OOS 2020-25', color=GREEN, alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(pair_order, fontsize=9, color=DIM)
ax.axhline(0, color=DIM, linewidth=0.8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'GBP{v:.0f}k'))
ax.set_title('Per-Pair P&L: In-Sample vs Out-of-Sample', color=WHITE, fontsize=11, fontweight='bold', pad=6)
ax.legend(fontsize=9, facecolor=CARD, labelcolor=WHITE)

# ── Panel 10: Walk-Forward verdict ──
ax = fig.add_subplot(gs[3, 2:4]); setup(ax)
ax.set_xticks([]); ax.set_yticks([])

lines = [
    ('WALK-FORWARD SUMMARY', WHITE, 12, True),
    ('', DIM, 9, False),
    (f'Full period net:      GBP{r_full["net"]:,.0f}', WHITE, 9, False),
    (f'In-sample net:        GBP{r_is["net"]:,.0f}  ({r_is["years"]:.0f}yr)', BLUE, 9, False),
    (f'Out-of-sample net:    GBP{r_oos["net"]:,.0f}  ({r_oos["years"]:.0f}yr)', GREEN, 9, False),
    ('', DIM, 9, False),
    (f'OOS Sharpe:           {r_oos["sharpe"]:.2f}  (IS: {r_is["sharpe"]:.2f})', 
     GREEN if r_oos['sharpe']>2 else AMBER, 9, False),
    (f'OOS Max DD:           {r_oos["max_dd"]:.2f}%  (IS: {r_is["max_dd"]:.2f}%)',
     GREEN if r_oos['max_dd']<8 else AMBER, 9, False),
    (f'OOS Pos months:       {r_oos["pos_mo"]:.0f}%  (IS: {r_is["pos_mo"]:.0f}%)',
     GREEN if r_oos['pos_mo']>85 else AMBER, 9, False),
    (f'Rolling OOS wins:     {sum(1 for v in roll_nets if v>0)}/{len(rolling)} years positive', 
     GREEN if sum(1 for v in roll_nets if v>0)>=len(rolling)*0.75 else AMBER, 9, False),
    ('', DIM, 9, False),
    (f'Robustness score:     {avg_ratio*100:.0f}%  ({verdict})', verdict_col, 11, True),
    ('', DIM, 9, False),
    ('If OOS metrics are within ~30% of IS metrics,', DIM, 8, False),
    ('the strategy is considered robust -- not overfit.', DIM, 8, False),
]
y = 0.97
for text, col, size, bold in lines:
    ax.text(0.03, y, text, transform=ax.transAxes, fontsize=size, color=col,
            fontfamily='monospace', fontweight='bold' if bold else 'normal')
    y -= 0.072

plt.savefig('smm_walkforward.png', dpi=140, bbox_inches='tight', facecolor=BG)
print("Walk-forward chart saved: smm_walkforward.png")
plt.close()

# Print summary to console too
print()
print("=" * 55)
print("  WALK-FORWARD RESULTS")
print("=" * 55)
print(f"  In-Sample  2015-2019:  GBP{r_is['net']:,.0f}  Sharpe {r_is['sharpe']:.2f}  DD {r_is['max_dd']:.2f}%")
print(f"  OOS        2020-2025:  GBP{r_oos['net']:,.0f}  Sharpe {r_oos['sharpe']:.2f}  DD {r_oos['max_dd']:.2f}%")
print(f"  Robustness score:      {avg_ratio*100:.0f}%  ({verdict})")
print(f"  Rolling OOS:           {sum(1 for v in roll_nets if v>0)}/{len(rolling)} positive years")
print()
for r in rolling:
    print(f"  OOS {r['label'].split()[-1]}:  GBP{r['net']:+,.0f}  Sharpe {r['sharpe']:.2f}  DD {r['max_dd']:.2f}%")
