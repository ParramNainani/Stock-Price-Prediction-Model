"""
══════════════════════════════════════════════════════════════════════════
  ★ BATCH RUNNER — ₹5,000 BUDGET — 10-20% RETURN TARGET ★
  Stocks under ₹100 | Good volume | Fresh ML predictions
  Runs: Attention-LSTM + Transformer + XGBoost ensemble
══════════════════════════════════════════════════════════════════════════
"""
import os, sys, re, time, json, subprocess
import requests
import numpy as np
from datetime import datetime, timedelta

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except: pass

SCRIPT_FILE = "advanced_stock_predictor.py"
REPORT_FILE = "trading_reco_5k_budget.txt"
TODAY = datetime.now()

INDIAN_HOLIDAYS_2026 = [
    datetime(2026,1,26), datetime(2026,3,10), datetime(2026,3,17),
    datetime(2026,3,30), datetime(2026,4,2), datetime(2026,4,3),
    datetime(2026,4,14), datetime(2026,5,1), datetime(2026,5,25),
    datetime(2026,6,5), datetime(2026,7,6), datetime(2026,8,15),
    datetime(2026,8,19), datetime(2026,9,4), datetime(2026,10,2),
    datetime(2026,10,20), datetime(2026,11,9), datetime(2026,11,10),
    datetime(2026,11,30), datetime(2026,12,25),
]

def is_trading_day(dt):
    if dt.weekday() >= 5: return False
    return datetime(dt.year, dt.month, dt.day) not in INDIAN_HOLIDAYS_2026

def next_trading_day(dt):
    dt = dt + timedelta(days=1)
    while not is_trading_day(dt): dt += timedelta(days=1)
    return dt

def add_trading_days(dt, n):
    current = dt; count = 0
    while count < n:
        current += timedelta(days=1)
        if is_trading_day(current): count += 1
    return current

# ═══════════════════════════════════════════════════════════════════════
# STOCK UNIVERSE — Under ₹100, good volume, affordable with ₹5K
# Mix of: previous BUY signals + fresh high-volume picks + ALOKINDS
# ═══════════════════════════════════════════════════════════════════════
STOCKS = [
    # ── YOUR PICK ──
    ("ALOKINDS.NS",    "Alok Industries"),
    # ── PREVIOUS STRONG BUY / BUY SIGNALS (re-validate with fresh data) ──
    ("JPPOWER.NS",     "JP Power"),
    ("RPOWER.NS",      "R Power"),
    ("ZEEL.NS",        "Zee Entertainment"),
    ("ELECTCAST.NS",   "Electrosteel Cast"),
    ("UJJIVANSFB.NS",  "Ujjivan Small Fin"),
    ("JAYNECOIND.NS",  "Jayaswal Neco"),
    ("SBFC.NS",        "SBFC Finance"),
    # ── HIGH VOLUME UNDER ₹100 (fresh candidates) ──
    ("IDEA.NS",        "Vodafone Idea"),
    ("SUZLON.NS",      "Suzlon Energy"),
    ("YESBANK.NS",     "Yes Bank"),
    ("MMTC.NS",        "MMTC"),
    ("PCJEWELLER.NS",  "PC Jeweller"),
    ("HFCL.NS",        "HFCL"),
    ("NHPC.NS",        "NHPC"),
    ("NMDC.NS",        "NMDC"),
    ("HCC.NS",         "Hindustan Const"),
    ("NBCC.NS",        "NBCC India"),
    ("SJVN.NS",        "SJVN"),
    ("IRFC.NS",        "IRFC"),
    ("INOXWIND.NS",    "Inox Wind"),
    # ── BANKING/FINANCE UNDER ₹100 ──
    ("IDFCFIRSTB.NS",  "IDFC First Bank"),
    ("IDBI.NS",        "IDBI Bank"),
    ("MAHABANK.NS",    "Bank of Maha"),
    ("CENTRALBK.NS",   "Central Bank"),
    ("IOB.NS",         "Indian Overseas"),
    ("SOUTHBANK.NS",   "South Indian Bank"),
    ("UCOBANK.NS",     "UCO Bank"),
    ("PSB.NS",         "Punjab Sind Bank"),
    ("EQUITASBNK.NS",  "Equitas SFB"),
    # ── MID-VOLUME INTERESTING PICKS ──
    ("RENUKA.NS",      "Shree Renuka"),
    ("TRIDENT.NS",     "Trident"),
    ("NETWORK18.NS",   "Network18"),
    ("HATHWAY.NS",     "Hathway Cable"),
    ("RTNINDIA.NS",    "RattanIndia Ent"),
    ("RTNPOWER.NS",    "RattanIndia Pwr"),
    ("TTML.NS",        "Tata Teleservices"),
    ("MSUMI.NS",       "Motherson Sumi"),
    ("IFCI.NS",        "IFCI"),
    ("SAGILITY.NS",    "Sagility India"),
    ("NSLNISP.NS",     "NSIL"),
    ("ABFRL.NS",       "Aditya Birla FL"),
    ("IRB.NS",         "IRB Infra"),
    ("LLOYDSENGG.NS",  "Lloyds Engg"),
    ("LLOYDSENT.NS",   "Lloyds Enterprise"),
    ("BAJAJHFL.NS",    "Bajaj Housing"),
    ("JAIBALAJI.NS",   "Jai Balaji"),
    ("RBA.NS",         "Restaurant Brands"),
    ("GMRAIRPORT.NS",  "GMR Airports"),
]


def parse_output(output_file, name, ticker):
    """Parse model output for trading signals."""
    result = {
        'name': name, 'ticker': ticker,
        'last_close': 'N/A', 'predicted': 'N/A',
        'change_pct': 'N/A', 'signal': 'N/A',
        'confidence': 'N/A', 'target': 'N/A',
        'stop_loss': 'N/A', 'rr_ratio': 'N/A',
        'entry_price': 'N/A', 'sentiment_score': 'N/A',
        'model_agreement': 'N/A', 'action_detail': 'N/A',
        'status': 'OK',
    }
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) < 20:
            result['status'] = "INSUFFICIENT_DATA"
            return result
        for line in lines:
            ls = line.strip()
            if "Last close price   :" in ls:
                result['last_close'] = ls.split(":")[-1].strip()
            elif "★ Ensemble prediction" in ls and ":" in ls:
                result['predicted'] = ls.split(":")[-1].strip()
            elif "★ Sentiment-adjusted" in ls and ":" in ls:
                result['predicted'] = ls.split(":")[-1].strip()
            elif "★ Expected change" in ls and ":" in ls:
                result['change_pct'] = ls.split(":")[-1].strip()
            elif "★ SIGNAL:" in ls:
                sp = ls.split("★ SIGNAL:")[-1]
                result['signal'] = sp.split("(")[0].strip()
                if "Confidence:" in sp:
                    result['confidence'] = sp.split("Confidence:")[-1].replace(")", "").replace("%", "").strip()
            elif "★ ACTION:" in ls:
                result['action_detail'] = ls.split("★ ACTION:")[-1].strip()
            elif "★ Model Agreement:" in ls:
                result['model_agreement'] = ls.split("★ Model Agreement:")[-1].strip()
            elif "★ Sentiment score" in ls and ":" in ls:
                result['sentiment_score'] = ls.split(":")[-1].strip()
            elif "20-DAY" in ls and ("SETUP:" in ls or "NO 20-DAY" in ls):
                if "Buy near" in ls or "Sell/Short near" in ls:
                    parts = ls.split("|")
                    for p in parts:
                        p = p.strip()
                        if "Buy near" in p or "Sell/Short near" in p:
                            m = re.search(r'₹([\d.]+)', p)
                            if m: result['entry_price'] = f"₹{m.group(1)}"
                        if "Target:" in p:
                            m = re.search(r'₹([\d.]+)', p)
                            if m: result['target'] = f"₹{m.group(1)}"
                        if "Stop Loss:" in p:
                            m = re.search(r'₹([\d.]+)', p)
                            if m: result['stop_loss'] = f"₹{m.group(1)}"
                        if "RR:" in p:
                            m = re.search(r'RR:\s*([\d.]+)', p)
                            if m: result['rr_ratio'] = m.group(1)
    except Exception as e:
        result['status'] = f"PARSE_ERROR: {str(e)[:50]}"
    return result


def generate_report(results):
    """Generate ₹5K budget trading report."""
    entry_date = next_trading_day(TODAY)
    target_20d = add_trading_days(entry_date, 20)

    buy_strong = [r for r in results if "STRONG BUY" in str(r.get('signal','')) and r['status']=='OK']
    buy = [r for r in results if "BUY" in str(r.get('signal','')) and "STRONG" not in str(r.get('signal','')) and r['status']=='OK']
    sell = [r for r in results if "SELL" in str(r.get('signal','')) and r['status']=='OK']
    hold = [r for r in results if "HOLD" in str(r.get('signal','')) and r['status']=='OK']
    fail = [r for r in results if r['status'] != 'OK']

    all_buy = buy_strong + buy

    L = []
    L.append("=" * 110)
    L.append("  ★★★ TRADING RECOMMENDATIONS — ₹5,000 BUDGET — 10-20% RETURN TARGET ★★★")
    L.append(f"  Generated    : {datetime.now().strftime('%d-%b-%Y %H:%M IST')}")
    L.append(f"  Budget       : ₹5,000")
    L.append(f"  Target Return: 10-20% (₹500-₹1,000 profit)")
    L.append(f"  Stocks Run   : {len(results)}")
    L.append(f"  Next Trading : {entry_date.strftime('%d-%b-%Y (%A)')}")
    L.append(f"  Exit By      : {target_20d.strftime('%d-%b-%Y (%A)')}")
    L.append("=" * 110)
    L.append("")
    L.append(f"  🟢🟢 STRONG BUY: {len(buy_strong)}  |  🟢 BUY: {len(buy)}  |  🔴 SELL: {len(sell)}  |  🟡 HOLD: {len(hold)}  |  ❌ Failed: {len(fail)}")
    L.append("")

    if all_buy:
        L.append("=" * 110)
        L.append(f"  🟢 BUY RECOMMENDATIONS — {len(all_buy)} Stocks")
        L.append("=" * 110)
        L.append(f"  {'#':<4} {'Stock':<22} {'Ticker':<18} {'CMP':<10} {'Signal':<16} {'Target':<10} {'SL':<10} {'R:R':<8} {'Agreement'}")
        L.append(f"  {'─'*105}")

        for idx, r in enumerate(all_buy, 1):
            sig = "⚡STRONG" if "STRONG" in str(r['signal']) else "📈BUY"
            L.append(f"  {idx:<4} {r['name']:<22} {r['ticker']:<18} {r['last_close']:<10} {sig:<16} {r['target']:<10} {r['stop_loss']:<10} {r['rr_ratio']:<8} {r['model_agreement']}")

        L.append("")
        L.append("  ── DETAILED BUY CARDS ──")
        for idx, r in enumerate(all_buy, 1):
            sig_icon = "🔥" if "STRONG" in str(r['signal']) else "📈"
            L.append(f"\n  ┌{'─'*90}┐")
            L.append(f"  │  {sig_icon} #{idx}. {r['name']} ({r['ticker']})")
            L.append(f"  ├{'─'*90}┤")
            L.append(f"  │  CMP: {r['last_close']}  |  Predicted: {r['predicted']}  |  Change: {r['change_pct']}")
            L.append(f"  │  Confidence: {r['confidence']}%  |  Agreement: {r['model_agreement']}  |  Sentiment: {r['sentiment_score']}")
            L.append(f"  │")
            L.append(f"  │  ★ ENTRY    : {r['entry_price'] if r['entry_price'] != 'N/A' else r['last_close']}")
            L.append(f"  │  ★ TARGET   : {r['target']}")
            L.append(f"  │  ★ STOP LOSS: {r['stop_loss']}")
            L.append(f"  │  ★ R:R      : 1:{r['rr_ratio']}")
            # Calculate shares with ₹5K
            try:
                cmp = float(r['last_close'].replace('₹','').replace(',',''))
                shares = int(5000 / cmp)
                invest = shares * cmp
                tgt = float(r['target'].replace('₹','').replace(',',''))
                profit = shares * (tgt - cmp)
                ret_pct = ((tgt - cmp) / cmp) * 100
                L.append(f"  │")
                L.append(f"  │  💰 WITH ₹5,000: Buy {shares} shares @ ₹{cmp:.2f} = ₹{invest:,.0f}")
                L.append(f"  │     If target hit: Profit = ₹{profit:,.0f} ({ret_pct:+.1f}%)")
            except:
                pass
            L.append(f"  │")
            L.append(f"  │  📅 Buy: {entry_date.strftime('%d-%b-%Y')}  →  Exit by: {target_20d.strftime('%d-%b-%Y')}")
            L.append(f"  └{'─'*90}┘")

    if sell:
        L.append(f"\n{'='*110}")
        L.append(f"  🔴 SELL/AVOID — {len(sell)} Stocks")
        L.append(f"{'='*110}")
        for r in sell:
            L.append(f"  📉 {r['name']:<22} ({r['ticker']:<16}) CMP: {r['last_close']:<10} Target: {r['target']:<10} SL: {r['stop_loss']}")

    if hold:
        L.append(f"\n{'='*110}")
        L.append(f"  🟡 HOLD/WAIT — {len(hold)} Stocks")
        L.append(f"{'='*110}")
        for r in hold[:20]:
            L.append(f"  ⚖️ {r['name']:<22} ({r['ticker']:<16}) CMP: {r['last_close']}")
        if len(hold) > 20:
            L.append(f"  ... and {len(hold)-20} more")

    L.append(f"\n{'='*110}")
    L.append("  ⚠️ AI/ML predictions ≠ financial advice. Use stop-losses. Consult SEBI advisor.")
    L.append(f"{'='*110}")

    report = "\n".join(L)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n  ✅ Report saved: {REPORT_FILE}")


def main():
    total = len(STOCKS)
    print(f"""
{'═'*70}
  ★★★ BATCH RUNNER — ₹5,000 BUDGET EDITION ★★★
  {total} stocks | Under ₹100 | Target: 10-20% return
  Started: {datetime.now().strftime('%d-%b-%Y %H:%M:%S')}
  Est Time: ~{total * 3}-{total * 7} minutes
{'═'*70}
""")

    with open(SCRIPT_FILE, "r", encoding="utf-8") as f:
        original_code = f.read()

    hdr = f"  {'#':<4} {'Name':<22} {'Ticker':<18} {'Signal':<16} {'Target':<14} {'Time'}"
    print(hdr)
    print("  " + "─" * 85)

    results = []
    try:
        for i, (ticker, name) in enumerate(STOCKS, 1):
            start_t = time.time()

            new_code = re.sub(r'TICKER\s*=\s*".*?"', f'TICKER         = "{ticker}"', original_code)
            new_code = re.sub(r'TICKER_NAME\s*=\s*".*?"', f'TICKER_NAME    = "{name}"', new_code)
            new_code = re.sub(r'BENCH_TICKER\s*=\s*".*?"', f'BENCH_TICKER = "{ticker}"', new_code)
            new_code = re.sub(r'VIX_TICKER\s*=\s*".*?"', f'VIX_TICKER = "{ticker}"', new_code)
            new_code = re.sub(r'USDINR_TICKER\s*=\s*".*?"', f'USDINR_TICKER = "{ticker}"', new_code)
            new_code = re.sub(r'GOLD_TICKER\s*=\s*".*?"', f'GOLD_TICKER = "{ticker}"', new_code)
            new_code = re.sub(r'plt\.show\(\)', '# plt.show()', new_code)

            with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
                f.write(new_code)

            output_file = f"run_output_{ticker.replace('.', '_').replace('-', '_')}.txt"
            try:
                with open(output_file, "w", encoding="utf-8") as out_f:
                    subprocess.run(
                        [sys.executable, "-X", "utf8", SCRIPT_FILE],
                        stdout=out_f, stderr=subprocess.STDOUT, timeout=900,
                    )
            except subprocess.TimeoutExpired:
                with open(output_file, "a", encoding="utf-8") as out_f:
                    out_f.write("\nTIMEOUT\n")
            except Exception as e:
                with open(output_file, "w", encoding="utf-8") as out_f:
                    out_f.write(f"ERROR: {e}\n")

            result = parse_output(output_file, name, ticker)
            results.append(result)

            elapsed = time.time() - start_t
            sig = result['signal'] if result['status'] == 'OK' else 'FAILED'
            tgt = result['target'] if result['target'] != 'N/A' else '-'
            remaining = (total - i) * elapsed
            print(f"  {i:<4} {name:<22} {ticker:<18} {sig:<16} {tgt:<14} {elapsed:.0f}s (~{remaining/60:.0f}m left)")

            with open("batch_5k_progress.txt", "a", encoding="utf-8") as pf:
                pf.write(f"{i},{name},{ticker},{result['signal']},{result['target']},{result['stop_loss']},{elapsed:.0f}s\n")

    except KeyboardInterrupt:
        print("\n\n  ⚠ Interrupted! Saving partial results...")
    finally:
        with open(SCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(original_code)
        print(f"\n  ✓ Original {SCRIPT_FILE} restored.\n")

    if results:
        generate_report(results)

    buy_c = len([r for r in results if "BUY" in str(r.get('signal',''))])
    sell_c = len([r for r in results if "SELL" in str(r.get('signal',''))])
    hold_c = len([r for r in results if "HOLD" in str(r.get('signal',''))])
    print(f"""
{'═'*70}
  ★ BATCH COMPLETE ★
  Stocks: {len(results)}/{total}
  🟢 BUY: {buy_c}  |  🔴 SELL: {sell_c}  |  🟡 HOLD: {hold_c}
  Report: {REPORT_FILE}
{'═'*70}
""")


if __name__ == "__main__":
    main()
