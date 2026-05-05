"""
FRESH STOCK SCANNER + PREDICTOR - Rs 5-6K Budget
Scrapes yfinance for NEW high-momentum stocks, filters fundamentals,
then runs ML predictions on the top candidates.
"""
import os, sys, re, time, json, subprocess
import numpy as np
from datetime import datetime, timedelta

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except: pass

import yfinance as yf
import pandas as pd

BUDGET = 6000
MAX_PRICE = 150  # can buy 40+ shares
MIN_VOLUME = 500000
SCRIPT_FILE = "advanced_stock_predictor.py"
PROGRESS_FILE = "fresh_5k_progress.txt"
REPORT_FILE = "fresh_5k_recommendations.txt"
TODAY = datetime.now()

# ================================================================
# PHASE 1: Discover stocks from NSE indices via yfinance
# ================================================================
def get_nse_symbols():
    """Get NSE symbols from multiple sources."""
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Discovering NSE stocks from yfinance")
    print(f"{'='*60}")
    
    import requests
    symbols = set()
    
    # Method 1: NSE API indices
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': '*/*', 'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com/',
    })
    
    indices = [
        "NIFTY TOTAL MARKET", "NIFTY 500", "NIFTY MIDCAP 150",
        "NIFTY SMALLCAP 250", "NIFTY MICROCAP 250",
        "NIFTY PSU BANK", "NIFTY BANK", "NIFTY METAL",
        "NIFTY PHARMA", "NIFTY AUTO", "NIFTY REALTY",
        "NIFTY MEDIA", "NIFTY ENERGY", "NIFTY INFRA",
        "NIFTY COMMODITIES", "NIFTY FINANCIAL SERVICES",
    ]
    
    try:
        session.get('https://www.nseindia.com', timeout=15)
        time.sleep(2)
        for idx_name in indices:
            try:
                encoded = requests.utils.quote(idx_name)
                url = f"https://www.nseindia.com/api/equity-stockIndices?index={encoded}"
                resp = session.get(url, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    for stock in data.get('data', []):
                        sym = stock.get('symbol', '')
                        if sym and sym != idx_name:
                            symbols.add(sym)
                time.sleep(0.8)
            except:
                pass
        print(f"  NSE API: {len(symbols)} symbols")
    except Exception as e:
        print(f"  NSE API failed: {e}")
    
    # Method 2: Hardcoded extras not in indices (small/microcaps)
    extras = [
        "ALOKINDS","PCJEWELLER","RTNPOWER","HATHWAY","SARVESHWAR",
        "SPCENET","MOTISONS","CELLECOR","KELLTONTEC","AFIL",
        "KAMDHENU","BCLIND","PAISALO","DHANBANK","HILTON",
        "CUPID","VMM","CCAVENUE","NSLNISP","ABLBL",
        "EMIL","TVSSCS","TEXRAIL","AEGISVOPAK","RTNINDIA",
        "LLOYDSENT","LLOYDSENGG","JAIBALAJI","WEBELSOLAR",
        "ELECTCAST","JAYNECOIND","DBREALTY",
    ]
    symbols.update(extras)
    print(f"  Total unique: {len(symbols)} symbols")
    return sorted(list(symbols))


# ================================================================
# PHASE 2: Download prices + apply smart filters
# ================================================================
def screen_stocks(symbols):
    """Download recent data and filter for budget-friendly momentum stocks."""
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Screening {len(symbols)} stocks")
    print(f"  Budget: Rs {BUDGET} | Max price: Rs {MAX_PRICE}")
    print(f"{'='*60}")
    
    tickers = [f"{s}.NS" for s in symbols]
    candidates = []
    batch_size = 80
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        
        try:
            tickers_str = " ".join(batch)
            data = yf.download(tickers_str, period="1mo", progress=False, threads=True)
            
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        if ticker not in data['Close'].columns:
                            continue
                        close_s = data['Close'][ticker].dropna()
                        vol_s = data['Volume'][ticker].dropna()
                        if len(close_s) < 10:
                            continue
                        
                        cmp = float(close_s.iloc[-1])
                        avg_vol = float(vol_s.mean())
                        
                        # Price filter
                        if cmp <= 0 or cmp > MAX_PRICE or avg_vol < MIN_VOLUME:
                            continue
                        
                        # Calculate momentum signals
                        ret_5d = (cmp / float(close_s.iloc[-6]) - 1) * 100 if len(close_s) >= 6 else 0
                        ret_20d = (cmp / float(close_s.iloc[0]) - 1) * 100
                        
                        # Volume trend (recent vs older)
                        recent_vol = float(vol_s.iloc[-5:].mean())
                        older_vol = float(vol_s.iloc[:10].mean()) if len(vol_s) >= 15 else recent_vol
                        vol_surge = recent_vol / max(older_vol, 1)
                        
                        # Shares you can buy
                        shares = int(BUDGET / cmp)
                        
                        # Score: momentum + volume surge + affordability
                        score = ret_5d * 0.4 + ret_20d * 0.3 + (vol_surge - 1) * 20 * 0.3
                        
                        candidates.append({
                            'symbol': ticker.replace('.NS', ''),
                            'ticker': ticker,
                            'cmp': round(cmp, 2),
                            'avg_vol': int(avg_vol),
                            'ret_5d': round(ret_5d, 2),
                            'ret_20d': round(ret_20d, 2),
                            'vol_surge': round(vol_surge, 2),
                            'shares': shares,
                            'invest': round(shares * cmp, 0),
                            'score': round(score, 2),
                        })
                    except:
                        pass
            else:
                # Single ticker in batch
                if len(batch) == 1 and len(data) >= 10:
                    try:
                        cmp = float(data['Close'].dropna().iloc[-1])
                        avg_vol = float(data['Volume'].dropna().mean())
                        if 0 < cmp <= MAX_PRICE and avg_vol >= MIN_VOLUME:
                            close_s = data['Close'].dropna()
                            ret_5d = (cmp / float(close_s.iloc[-6]) - 1) * 100 if len(close_s) >= 6 else 0
                            ret_20d = (cmp / float(close_s.iloc[0]) - 1) * 100
                            shares = int(BUDGET / cmp)
                            candidates.append({
                                'symbol': batch[0].replace('.NS', ''),
                                'ticker': batch[0],
                                'cmp': round(cmp, 2), 'avg_vol': int(avg_vol),
                                'ret_5d': round(ret_5d, 2), 'ret_20d': round(ret_20d, 2),
                                'vol_surge': 1.0, 'shares': shares,
                                'invest': round(shares * cmp, 0), 'score': 0,
                            })
                    except:
                        pass
        except Exception as e:
            pass
        
        print(f"  Batch {batch_num}/{total_batches}: {len(candidates)} qualify so far")
        time.sleep(0.3)
    
    # Sort by score (best momentum + volume first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n  Found {len(candidates)} stocks under Rs {MAX_PRICE}")
    print(f"\n  TOP 25 by momentum score:")
    print(f"  {'#':<4} {'Symbol':<14} {'CMP':<10} {'5d%':<8} {'20d%':<8} {'VolSurge':<10} {'Shares':<8} {'Score'}")
    print(f"  {'-'*75}")
    for i, s in enumerate(candidates[:25], 1):
        print(f"  {i:<4} {s['symbol']:<14} Rs{s['cmp']:<8} {s['ret_5d']:>+6.1f}% {s['ret_20d']:>+6.1f}% {s['vol_surge']:<10} {s['shares']:<8} {s['score']:>+.1f}")
    
    return candidates


# ================================================================
# PHASE 3: Run ML predictions on top candidates
# ================================================================
def run_predictions(candidates, max_stocks=20):
    """Run advanced_stock_predictor on top candidates."""
    stocks = candidates[:max_stocks]
    
    print(f"\n{'='*60}")
    print(f"  PHASE 3: Running ML on top {len(stocks)} stocks")
    print(f"  Est time: ~{len(stocks) * 1}-{len(stocks) * 2} minutes")
    print(f"{'='*60}")
    
    with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
        original_code = f.read()
    
    results = []
    # Clear progress file
    open(PROGRESS_FILE, "w").close()
    
    try:
        for i, stock in enumerate(stocks, 1):
            ticker = stock['ticker']
            name = stock['symbol']
            start_t = time.time()
            
            # Modify predictor config
            new_code = re.sub(r'TICKER\s*=\s*".*?"', f'TICKER         = "{ticker}"', original_code)
            new_code = re.sub(r'TICKER_NAME\s*=\s*".*?"', f'TICKER_NAME    = "{name}"', new_code)
            # Keep intermarket tickers as proper benchmarks
            new_code = re.sub(r'BENCH_TICKER\s*=\s*".*?"', 'BENCH_TICKER = "^NSEI"', new_code)
            new_code = re.sub(r'VIX_TICKER\s*=\s*".*?"', 'VIX_TICKER = "^INDIAVIX"', new_code)
            new_code = re.sub(r'USDINR_TICKER\s*=\s*".*?"', 'USDINR_TICKER = "USDINR=X"', new_code)
            new_code = re.sub(r'GOLD_TICKER\s*=\s*".*?"', 'GOLD_TICKER = "GC=F"', new_code)
            new_code = re.sub(r'plt\.show\(\)', '# plt.show()', new_code)
            
            with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
                f.write(new_code)
            
            output_file = f"run_output_{ticker.replace('.', '_').replace('-', '_')}.txt"
            try:
                with open(output_file, "w", encoding="utf-8") as out_f:
                    subprocess.run(
                        [sys.executable, "-X", "utf8", SCRIPT_FILE],
                        stdout=out_f, stderr=subprocess.STDOUT, timeout=600,
                    )
            except subprocess.TimeoutExpired:
                with open(output_file, "a", encoding="utf-8") as out_f:
                    out_f.write("\nTIMEOUT\n")
            except Exception as e:
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(f"ERROR: {e}\n")
            
            result = parse_output(output_file, stock)
            results.append(result)
            
            elapsed = int(time.time() - start_t)
            sig = result.get('signal', 'N/A')
            tgt = result.get('target', '-')
            remaining = (len(stocks) - i) * elapsed
            
            line = f"{i},{name},{ticker},{stock['cmp']},{sig},{tgt},{elapsed}s"
            print(f"  [{i}/{len(stocks)}] {name:<14} Rs{stock['cmp']:<8} {sig:<16} {tgt:<12} {elapsed}s (~{remaining//60}m left)")
            
            with open(PROGRESS_FILE, "a", encoding="utf-8") as pf:
                pf.write(line + "\n")
    
    except KeyboardInterrupt:
        print("\n  Interrupted! Saving partial results...")
    finally:
        with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(original_code)
        print(f"\n  Original {SCRIPT_FILE} restored.")
    
    return results


def parse_output(output_file, stock):
    """Parse model output."""
    r = dict(stock)  # copy screening data
    r.update({
        'signal': 'N/A', 'confidence': 'N/A',
        'predicted': 'N/A', 'change_pct': 'N/A',
        'target': 'N/A', 'stop_loss': 'N/A',
        'rr_ratio': 'N/A', 'model_agreement': 'N/A',
        'sentiment': 'N/A', 'status': 'OK',
    })
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 20:
            r['status'] = "INSUFFICIENT_DATA"
            return r
        for line in lines:
            ls = line.strip()
            if "Last close price   :" in ls:
                r['last_close'] = ls.split(":")[-1].strip()
            elif "Ensemble prediction" in ls and ":" in ls:
                r['predicted'] = ls.split(":")[-1].strip()
            elif "Sentiment-adjusted" in ls and ":" in ls:
                r['predicted'] = ls.split(":")[-1].strip()
            elif "Expected change" in ls and ":" in ls:
                r['change_pct'] = ls.split(":")[-1].strip()
            elif "SIGNAL:" in ls:
                sp = ls.split("SIGNAL:")[-1]
                r['signal'] = sp.split("(")[0].strip()
                if "Confidence:" in sp:
                    r['confidence'] = sp.split("Confidence:")[-1].replace(")","").replace("%","").strip()
            elif "Model Agreement:" in ls:
                r['model_agreement'] = ls.split("Model Agreement:")[-1].strip()
            elif "Sentiment score" in ls and ":" in ls:
                r['sentiment'] = ls.split(":")[-1].strip()
            elif "20-DAY" in ls and ("SETUP:" in ls or "NO 20-DAY" in ls):
                if "Buy near" in ls or "Sell/Short near" in ls:
                    parts = ls.split("|")
                    for p in parts:
                        p = p.strip()
                        if "Target:" in p:
                            m = re.search(r'[\u20b9]?([\d.]+)', p.split("Target:")[-1])
                            if m: r['target'] = f"Rs{m.group(1)}"
                        if "Stop Loss:" in p:
                            m = re.search(r'[\u20b9]?([\d.]+)', p.split("Stop Loss:")[-1])
                            if m: r['stop_loss'] = f"Rs{m.group(1)}"
                        if "RR:" in p:
                            m = re.search(r'RR:\s*([\d.]+)', p)
                            if m: r['rr_ratio'] = m.group(1)
    except Exception as e:
        r['status'] = f"PARSE_ERROR"
    return r


# ================================================================
# PHASE 4: Generate report
# ================================================================
def generate_report(results):
    """Generate Rs 5-6K budget report."""
    buy_stocks = [r for r in results if "BUY" in str(r.get('signal','')) and r.get('status')=='OK']
    sell_stocks = [r for r in results if "SELL" in str(r.get('signal','')) and r.get('status')=='OK']
    hold_stocks = [r for r in results if "HOLD" in str(r.get('signal','')) and r.get('status')=='OK']
    
    L = []
    L.append("=" * 100)
    L.append(f"  FRESH STOCK PICKS - Rs {BUDGET} BUDGET - {TODAY.strftime('%d-%b-%Y %H:%M')}")
    L.append(f"  Market Opens: 05-May-2026 (Monday) 09:15 IST")
    L.append(f"  Stocks Analyzed: {len(results)}")
    L.append("=" * 100)
    L.append(f"  BUY: {len(buy_stocks)} | SELL: {len(sell_stocks)} | HOLD: {len(hold_stocks)}")
    L.append("")
    
    if buy_stocks:
        # Sort by confidence/score
        L.append("=" * 100)
        L.append("  BUY RECOMMENDATIONS")
        L.append("=" * 100)
        for i, r in enumerate(buy_stocks, 1):
            shares = int(BUDGET / r['cmp'])
            invest = shares * r['cmp']
            sig = "STRONG BUY" if "STRONG" in str(r['signal']) else "BUY"
            L.append(f"\n  #{i}. {r['symbol']} ({r['ticker']}) - {sig}")
            L.append(f"     CMP: Rs{r['cmp']} | Predicted: {r['predicted']} | Change: {r['change_pct']}")
            L.append(f"     Target: {r['target']} | Stop Loss: {r['stop_loss']} | R:R 1:{r['rr_ratio']}")
            L.append(f"     Agreement: {r['model_agreement']} | Sentiment: {r['sentiment']}")
            L.append(f"     Momentum: 5d={r['ret_5d']:+.1f}% | 20d={r['ret_20d']:+.1f}%")
            L.append(f"     WITH Rs{BUDGET}: Buy {shares} shares @ Rs{r['cmp']} = Rs{invest:,.0f}")
            try:
                tgt_val = float(str(r['target']).replace('Rs','').replace(',',''))
                profit = shares * (tgt_val - r['cmp'])
                ret = ((tgt_val - r['cmp']) / r['cmp']) * 100
                L.append(f"     If target hit: Profit = Rs{profit:,.0f} ({ret:+.1f}%)")
            except:
                pass
    
    if sell_stocks:
        L.append(f"\n{'='*100}")
        L.append("  SELL/AVOID")
        L.append("=" * 100)
        for r in sell_stocks:
            L.append(f"  AVOID {r['symbol']} ({r['ticker']}) CMP: Rs{r['cmp']} Target: {r['target']}")
    
    if hold_stocks:
        L.append(f"\n{'='*100}")
        L.append("  HOLD/WAIT")
        L.append("=" * 100)
        for r in hold_stocks:
            L.append(f"  WAIT {r['symbol']} ({r['ticker']}) CMP: Rs{r['cmp']}")
    
    L.append(f"\n{'='*100}")
    L.append("  WARNING: AI/ML predictions != financial advice. Use stop-losses. Consult SEBI advisor.")
    L.append("=" * 100)
    
    report = "\n".join(L)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n  Saved: {REPORT_FILE}")


# ================================================================
# MAIN
# ================================================================
def main():
    print(f"""
{'='*60}
  FRESH STOCK SCANNER + ML PREDICTOR
  Budget: Rs {BUDGET} | Max Price: Rs {MAX_PRICE}
  Started: {TODAY.strftime('%d-%b-%Y %H:%M')}
{'='*60}
""")
    
    # Phase 1: Discover
    symbols = get_nse_symbols()
    if not symbols:
        print("  No symbols found! Check internet.")
        return
    
    # Phase 2: Screen
    candidates = screen_stocks(symbols)
    if not candidates:
        print("  No candidates found!")
        return
    
    # Save screened list
    with open("fresh_5k_screened.json", "w", encoding="utf-8") as f:
        json.dump(candidates[:30], f, indent=2)
    
    # Phase 3: Run ML on top 20 by momentum
    results = run_predictions(candidates, max_stocks=20)
    
    # Phase 4: Report
    if results:
        generate_report(results)
    
    buy_c = len([r for r in results if "BUY" in str(r.get('signal',''))])
    sell_c = len([r for r in results if "SELL" in str(r.get('signal',''))])
    hold_c = len([r for r in results if "HOLD" in str(r.get('signal',''))])
    
    print(f"""
{'='*60}
  COMPLETE!
  Screened: {len(symbols)} | Qualified: {len(candidates)} | ML Run: {len(results)}
  BUY: {buy_c} | SELL: {sell_c} | HOLD: {hold_c}
  Report: {REPORT_FILE}
{'='*60}
""")


if __name__ == '__main__':
    main()
