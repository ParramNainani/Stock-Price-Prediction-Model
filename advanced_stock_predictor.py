"""
══════════════════════════════════════════════════════════════════════════
  ADVANCED MULTI-MODEL INDIAN STOCK PRICE PREDICTION SYSTEM
══════════════════════════════════════════════════════════════════════════
  • Ensemble: Attention-LSTM + Transformer + XGBoost
  • 60+ Features: Technical, Fundamental, Sentiment, Chart Patterns
  • Chart Patterns: Double Top(M), Double Bottom(W), Breakout, H&S, Triangle
  • Social Sentiment: Reddit + Google News real-time scraping
  • Wavelet Denoising for cleaner signals
  • Indian Market Focus (NSE stocks, Nifty, India VIX, USD/INR)
══════════════════════════════════════════════════════════════════════════
"""
import os, sys, warnings, time, json, re, math
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONIOENCODING'] = 'utf-8'
# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.signal import argrelextrema

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge

import yfinance as yf
import pywt
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Bidirectional, BatchNormalization,
    Input, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, Add, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

try:
    import feedparser; HAS_FEED = True
except ImportError: HAS_FEED = False
try:
    from textblob import TextBlob; HAS_BLOB = True
except ImportError: HAS_BLOB = False
try:
    import lightgbm as lgb; HAS_LGBM = True
except ImportError: HAS_LGBM = False

# ━━━━━━━━  CONFIGURATION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKER         = "HINDUNILVR.NS"
TICKER_NAME    = "Hindustan Unilever"
START_DATE     = "2020-01-01"
SEQ_LEN        = 30
EPOCHS         = 30
BATCH_SIZE     = 64
PATIENCE       = 10
LR             = 0.0005
MODEL_DIR      = "models"
# Intermarket (Indian context)
BENCH_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"
USDINR_TICKER = "USDINR=X"
GOLD_TICKER = "GC=F"
# Sentiment
REDDIT_SUBS    = ["IndianStockMarket", "IndianStreetBets"]
# Chart pattern params
PAT_LOOKBACK   = 30
PAT_TOL        = 0.02

os.makedirs(MODEL_DIR, exist_ok=True)
P = lambda msg: print(f"  {msg}")

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════
def fetch_ohlcv(ticker, start):
    """Download OHLCV from Yahoo Finance."""
    print(f"\n{'═'*70}\n  📥 Fetching {ticker} from {start}\n{'═'*70}")
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    P(f"✓ {len(df)} trading days  |  {df.index.min().date()} → {df.index.max().date()}")
    P(f"✓ Latest close: ₹{df['Close'].iloc[-1]:.2f}")
    return df

def fetch_intermarket(start):
    """Fetch Nifty50, India VIX, USD/INR, Gold."""
    print(f"\n{'═'*70}\n  📥 Fetching intermarket data\n{'═'*70}")
    data = {}
    for name, t in [("Nifty50", BENCH_TICKER), ("IndiaVIX", VIX_TICKER),
                     ("USDINR", USDINR_TICKER), ("Gold", GOLD_TICKER)]:
        try:
            d = yf.download(t, start=start, auto_adjust=True, progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            data[name] = d['Close'].rename(name)
            P(f"✓ {name}: {len(d)} rows")
        except Exception as e:
            P(f"⚠ {name} failed: {e}")
    return data

def fetch_fundamentals(ticker):
    """Get fundamental data from yfinance."""
    print(f"\n{'═'*70}\n  📥 Fetching fundamentals for {ticker}\n{'═'*70}")
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        fundas = {
            'PE_Ratio': info.get('trailingPE', np.nan),
            'Forward_PE': info.get('forwardPE', np.nan),
            'PB_Ratio': info.get('priceToBook', np.nan),
            'Dividend_Yield': info.get('dividendYield', 0) or 0,
            'Debt_to_Equity': info.get('debtToEquity', np.nan),
            'Market_Cap': info.get('marketCap', np.nan),
            'EPS': info.get('trailingEps', np.nan),
            'Book_Value': info.get('bookValue', np.nan),
            'ROE': info.get('returnOnEquity', np.nan),
            'Revenue_Growth': info.get('revenueGrowth', np.nan),
            'Profit_Margin': info.get('profitMargins', np.nan),
        }
        # Promoter/Institutional holdings
        try:
            holders = tk.major_holders
            if holders is not None and len(holders) > 0:
                for _, row in holders.iterrows():
                    val_str = str(row.iloc[0]).replace('%', '')
                    label = str(row.iloc[1]).lower()
                    try:
                        val = float(val_str) / 100
                    except ValueError:
                        continue
                    if 'insider' in label or 'promoter' in label:
                        fundas['Promoter_Holding'] = val
                    elif 'institution' in label:
                        fundas['Institutional_Holding'] = val
        except: pass
        for k, v in fundas.items():
            P(f"  {k}: {v}")
        return fundas
    except Exception as e:
        P(f"⚠ Fundamentals failed: {e}")
        return {}

# ═══════════════════════════════════════════════════════════════════════
# 2. SOCIAL SENTIMENT SCRAPER
# ═══════════════════════════════════════════════════════════════════════
class SentimentScraper:
    def __init__(self, ticker_name):
        self.ticker_name = ticker_name
        self.vader = SentimentIntensityAnalyzer()
        self.headers = {'User-Agent': 'Mozilla/5.0 (StockPredictor/1.0)'}

    def _vader_score(self, text):
        return self.vader.polarity_scores(text)['compound']

    def _combined_score(self, text):
        v = self._vader_score(text)
        if HAS_BLOB:
            b = TextBlob(text).sentiment.polarity
            return 0.6 * v + 0.4 * b
        return v

    def scrape_reddit(self, limit=50):
        """Scrape Reddit for stock mentions."""
        P("🔍 Scraping Reddit...")
        all_posts = []
        for sub in REDDIT_SUBS:
            try:
                url = f"https://www.reddit.com/r/{sub}/search.json"
                params = {'q': self.ticker_name, 'sort': 'new',
                          'limit': limit, 'restrict_sr': True, 't': 'month'}
                r = requests.get(url, headers=self.headers, params=params, timeout=10)
                if r.status_code == 200:
                    posts = r.json().get('data', {}).get('children', [])
                    for p in posts:
                        d = p.get('data', {})
                        text = f"{d.get('title', '')} {d.get('selftext', '')[:300]}"
                        score = self._combined_score(text)
                        all_posts.append({
                            'source': f'r/{sub}', 'text': d.get('title', ''),
                            'score': score, 'upvotes': d.get('ups', 0),
                            'time': datetime.fromtimestamp(d.get('created_utc', 0))
                        })
                    P(f"  r/{sub}: {len(posts)} posts")
                time.sleep(1)  # Rate limit
            except Exception as e:
                P(f"  ⚠ r/{sub} failed: {e}")
        return all_posts

    def scrape_google_news(self, limit=20):
        """Scrape Google News RSS for stock news."""
        if not HAS_FEED:
            P("⚠ feedparser not installed, skipping Google News")
            return []
        P("🔍 Scraping Google News...")
        articles = []
        try:
            query = self.ticker_name.replace(' ', '+')
            url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-IN&gl=IN&ceid=IN:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:limit]:
                text = f"{entry.get('title', '')} {entry.get('summary', '')[:200]}"
                text_clean = BeautifulSoup(text, 'html.parser').get_text()
                score = self._combined_score(text_clean)
                articles.append({
                    'source': 'Google News', 'text': entry.get('title', ''),
                    'score': score, 'time': entry.get('published', '')
                })
            P(f"  Google News: {len(articles)} articles")
        except Exception as e:
            P(f"  ⚠ Google News failed: {e}")
        return articles

    def get_sentiment_summary(self):
        """Get aggregated sentiment from all sources."""
        print(f"\n{'═'*70}\n  🧠 Social Sentiment Analysis\n{'═'*70}")
        reddit = self.scrape_reddit()
        news = self.scrape_google_news()
        all_items = reddit + news
        if not all_items:
            P("⚠ No sentiment data available")
            return {'overall': 0, 'reddit': 0, 'news': 0, 'count': 0,
                    'bullish_pct': 50, 'items': []}
        reddit_scores = [p['score'] for p in reddit] if reddit else [0]
        news_scores = [a['score'] for a in news] if news else [0]
        all_scores = [i['score'] for i in all_items]
        overall = np.mean(all_scores)
        bullish = sum(1 for s in all_scores if s > 0.05) / len(all_scores) * 100
        bearish = sum(1 for s in all_scores if s < -0.05) / len(all_scores) * 100
        signal = "🟢 BULLISH" if overall > 0.1 else "🔴 BEARISH" if overall < -0.1 else "🟡 NEUTRAL"
        P(f"Overall Sentiment: {overall:.3f}  {signal}")
        P(f"Reddit avg: {np.mean(reddit_scores):.3f}  |  News avg: {np.mean(news_scores):.3f}")
        P(f"Bullish: {bullish:.0f}%  |  Bearish: {bearish:.0f}%  |  Neutral: {100-bullish-bearish:.0f}%")
        P(f"Total data points: {len(all_items)}")
        # Top headlines
        sorted_items = sorted(all_items, key=lambda x: abs(x['score']), reverse=True)[:5]
        P("\n  📰 Top Impactful Headlines:")
        for item in sorted_items:
            emoji = "🟢" if item['score'] > 0 else "🔴"
            P(f"  {emoji} [{item['score']:+.2f}] {item['text'][:80]}")
        return {
            'overall': overall, 'reddit': np.mean(reddit_scores),
            'news': np.mean(news_scores), 'count': len(all_items),
            'bullish_pct': bullish, 'items': all_items
        }

# ═══════════════════════════════════════════════════════════════════════
# 3. CHART PATTERN DETECTION (M, W, Breakout, H&S, Triangle)
# ═══════════════════════════════════════════════════════════════════════
class ChartPatternDetector:
    """Detects key chart patterns used in Indian stock trading."""

    @staticmethod
    def _find_extrema(series, order=5):
        max_idx = argrelextrema(series.values, np.greater, order=order)[0]
        min_idx = argrelextrema(series.values, np.less, order=order)[0]
        return max_idx, min_idx

    @staticmethod
    def detect_double_top_M(highs, closes, lookback=40, tol=0.02):
        """Double Top (M pattern) — Bearish reversal."""
        n = len(closes)
        signals = np.zeros(n)
        for i in range(lookback, n):
            window_h = highs.iloc[i-lookback:i]
            window_c = closes.iloc[i-lookback:i]
            peaks, _ = ChartPatternDetector._find_extrema(window_h, order=4)
            if len(peaks) < 2: continue
            # Take last two peaks
            p1, p2 = peaks[-2], peaks[-1]
            v1, v2 = window_h.iloc[p1], window_h.iloc[p2]
            if abs(v1 - v2) / max(v1, v2) < tol:
                trough = window_c.iloc[p1:p2+1].min()
                drop = (max(v1, v2) - trough) / max(v1, v2)
                if drop > 0.025 and window_c.iloc[-1] < trough:
                    signals[i] = -1  # Bearish
        return signals

    @staticmethod
    def detect_double_bottom_W(lows, closes, lookback=40, tol=0.02):
        """Double Bottom (W pattern) — Bullish reversal."""
        n = len(closes)
        signals = np.zeros(n)
        for i in range(lookback, n):
            window_l = lows.iloc[i-lookback:i]
            window_c = closes.iloc[i-lookback:i]
            _, troughs = ChartPatternDetector._find_extrema(window_l, order=4)
            if len(troughs) < 2: continue
            t1, t2 = troughs[-2], troughs[-1]
            v1, v2 = window_l.iloc[t1], window_l.iloc[t2]
            if abs(v1 - v2) / min(v1, v2) < tol:
                peak = window_c.iloc[t1:t2+1].max()
                rise = (peak - min(v1, v2)) / min(v1, v2)
                if rise > 0.025 and window_c.iloc[-1] > peak:
                    signals[i] = 1  # Bullish
        return signals

    @staticmethod
    def detect_head_shoulders(highs, closes, lookback=50, tol=0.10):
        """Head & Shoulders — Bearish / Inverse H&S — Bullish."""
        n = len(closes)
        signals = np.zeros(n)
        for i in range(lookback, n):
            window_h = highs.iloc[i-lookback:i]
            peaks, troughs = ChartPatternDetector._find_extrema(window_h, order=4)
            if len(peaks) >= 3:
                p = peaks[-3:]
                v = [window_h.iloc[j] for j in p]
                # Head is highest, shoulders similar
                if v[1] > v[0] and v[1] > v[2]:
                    if abs(v[0] - v[2]) / max(v[0], v[2]) < tol:
                        signals[i] = -1  # Bearish H&S
            # Inverse H&S
            if len(troughs) >= 3:
                t = troughs[-3:]
                v = [window_h.iloc[j] for j in t]
                if v[1] < v[0] and v[1] < v[2]:
                    if abs(v[0] - v[2]) / max(v[0], v[2]) < tol:
                        signals[i] = 1  # Bullish inverse H&S
        return signals

    @staticmethod
    def detect_breakout(highs, lows, closes, volumes, lookback=20):
        """Breakout/Breakdown with volume confirmation."""
        n = len(closes)
        signals = np.zeros(n)
        vol_ma = volumes.rolling(20).mean()
        for i in range(lookback + 1, n):
            resistance = highs.iloc[i-lookback:i].max()
            support = lows.iloc[i-lookback:i].min()
            high_vol = volumes.iloc[i] > vol_ma.iloc[i] * 1.5
            if closes.iloc[i] > resistance and high_vol:
                signals[i] = 1   # Bullish breakout
            elif closes.iloc[i] < support and high_vol:
                signals[i] = -1  # Bearish breakdown
        return signals

    @staticmethod
    def detect_triangle(highs, lows, lookback=30):
        """Ascending / Descending / Symmetric triangle."""
        n = len(highs)
        signals = np.zeros(n)
        for i in range(lookback, n):
            h = highs.iloc[i-lookback:i].values
            l = lows.iloc[i-lookback:i].values
            x = np.arange(lookback)
            try:
                h_slope = np.polyfit(x, h, 1)[0]
                l_slope = np.polyfit(x, l, 1)[0]
            except: continue
            if l_slope > 0 and abs(h_slope) < abs(l_slope) * 0.3:
                signals[i] = 1   # Ascending triangle (bullish)
            elif h_slope < 0 and abs(l_slope) < abs(h_slope) * 0.3:
                signals[i] = -1  # Descending triangle (bearish)
            elif h_slope < 0 and l_slope > 0:
                signals[i] = 0.5 if l_slope > abs(h_slope) else -0.5  # Symmetric
        return signals

    @staticmethod
    def detect_flag_pennant(closes, volumes, lookback=20):
        """Bull/Bear flag after strong move."""
        n = len(closes)
        signals = np.zeros(n)
        for i in range(lookback + 10, n):
            # Check for strong prior move (pole)
            pole_ret = (closes.iloc[i-lookback] - closes.iloc[i-lookback-10]) / closes.iloc[i-lookback-10]
            # Check for consolidation (flag)
            flag = closes.iloc[i-lookback:i]
            flag_range = (flag.max() - flag.min()) / flag.mean()
            if abs(pole_ret) > 0.05 and flag_range < 0.03:
                signals[i] = 1 if pole_ret > 0 else -1
        return signals

    def detect_all(self, df):
        """Run all pattern detectors, return DataFrame of signals."""
        print(f"\n{'═'*70}\n  📊 Detecting Chart Patterns (M, W, Breakout, H&S, Triangle)\n{'═'*70}")
        patterns = pd.DataFrame(index=df.index)
        patterns['Pat_DoubleTop_M'] = self.detect_double_top_M(df['High'], df['Close'])
        patterns['Pat_DoubleBottom_W'] = self.detect_double_bottom_W(df['Low'], df['Close'])
        patterns['Pat_HeadShoulders'] = self.detect_head_shoulders(df['High'], df['Close'])
        patterns['Pat_Breakout'] = self.detect_breakout(df['High'], df['Low'], df['Close'], df['Volume'])
        patterns['Pat_Triangle'] = self.detect_triangle(df['High'], df['Low'])
        patterns['Pat_Flag'] = self.detect_flag_pennant(df['Close'], df['Volume'])
        # Candlestick patterns
        o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
        body = abs(c - o)
        rng = h - l
        patterns['Cndl_Doji'] = (body / rng.replace(0, np.nan) < 0.1).astype(float)
        patterns['Cndl_Hammer'] = ((rng > 0) & (body / rng < 0.35) &
                                    ((c - l) / rng > 0.6) & (c > o)).astype(float)
        patterns['Cndl_BullEngulf'] = ((c > o) & (c.shift(1) < o.shift(1)) &
                                        (c > o.shift(1)) & (o < c.shift(1))).astype(float)
        patterns['Cndl_BearEngulf'] = ((c < o) & (c.shift(1) > o.shift(1)) &
                                        (c < o.shift(1)) & (o > c.shift(1))).astype(float)
        # Summary
        for col in patterns.columns:
            nz = (patterns[col] != 0).sum()
            if nz > 0: P(f"  {col}: {nz} signals detected")
        P(f"  Total pattern features: {len(patterns.columns)}")
        return patterns

# ═══════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING (60+ features)
# ═══════════════════════════════════════════════════════════════════════
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs = avg_g / avg_l
    return 100 - 100 / (1 + rs)

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return k, d

def compute_cci(high, low, close, period=20):
    tp = (high + low + close) / 3
    sma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad)

def compute_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0).rolling(period).sum()
    mfi = 100 - 100 / (1 + pos_mf / neg_mf.replace(0, 1))
    return mfi

def compute_obv(close, volume):
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv

def compute_adl(high, low, close, volume):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
    return (mfm * volume).cumsum()

def compute_cmf(high, low, close, volume, period=20):
    mfm = ((close - low) - (high - close)) / (high - low).replace(0, 1)
    return (mfm * volume).rolling(period).sum() / volume.rolling(period).sum()

def wavelet_denoise(series, wavelet='db4', level=2):
    """Wavelet denoising for cleaner price signals."""
    arr = series.values
    coeffs = pywt.wavedec(arr, wavelet, level=level)
    # Soft threshold detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    thresh = sigma * np.sqrt(2 * np.log(len(arr)))
    coeffs[1:] = [pywt.threshold(c, thresh, mode='soft') for c in coeffs[1:]]
    denoised = pywt.waverec(coeffs, wavelet)[:len(arr)]
    return pd.Series(denoised, index=series.index, name=f'{series.name}_denoised')

def engineer_features(df_raw, intermarket_data, fundamentals):
    """Build 60+ features from price, volume, patterns, fundamentals, intermarket."""
    print(f"\n{'═'*70}\n  🔧 Engineering 60+ Features\n{'═'*70}")
    d = df_raw.copy()
    c, h, l, o, v = d['Close'], d['High'], d['Low'], d['Open'], d['Volume']

    # --- Wavelet denoised close ---
    d['Close_Denoised'] = wavelet_denoise(c)

    # --- Returns ---
    for p in [1, 2, 3, 5, 10, 20]:
        d[f'Return_{p}d'] = c.pct_change(p)
    d['Log_Return'] = np.log(c / c.shift(1))

    # --- Moving Averages ---
    for w in [5, 10, 20, 50, 100, 200]:
        d[f'SMA_{w}'] = c.rolling(w).mean()
        d[f'Close_SMA{w}_Ratio'] = c / d[f'SMA_{w}']
    for w in [12, 26, 50]:
        d[f'EMA_{w}'] = c.ewm(span=w, adjust=False).mean()

    # --- MACD ---
    d['MACD'] = d['EMA_12'] - d['EMA_26']
    d['MACD_Signal'] = d['MACD'].ewm(span=9, adjust=False).mean()
    d['MACD_Hist'] = d['MACD'] - d['MACD_Signal']

    # --- Oscillators ---
    d['RSI_14'] = compute_rsi(c, 14)
    d['RSI_7'] = compute_rsi(c, 7)
    d['Stoch_K'], d['Stoch_D'] = compute_stochastic(h, l, c)
    d['Williams_R'] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min())
    d['CCI_20'] = compute_cci(h, l, c, 20)
    d['ROC_10'] = c.pct_change(10) * 100
    d['MFI_14'] = compute_mfi(h, l, c, v, 14)

    # --- Trend ---
    d['ADX_proxy'] = abs(d['Return_1d']).rolling(14).mean() * 100  # Simplified ADX proxy

    # --- Volatility ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    d['BB_Upper'] = bb_mid + 2 * bb_std
    d['BB_Lower'] = bb_mid - 2 * bb_std
    d['BB_Width'] = (d['BB_Upper'] - d['BB_Lower']) / bb_mid
    d['BB_PctB'] = (c - d['BB_Lower']) / (d['BB_Upper'] - d['BB_Lower'])
    # ATR
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d['ATR_14'] = tr.rolling(14).mean()
    d['ATR_pct'] = d['ATR_14'] / c * 100
    # Historical volatility
    for w in [10, 20, 60]:
        d[f'HV_{w}'] = d['Log_Return'].rolling(w).std() * np.sqrt(252) * 100

    # --- Volume ---
    d['OBV'] = compute_obv(c, v)
    d['Vol_SMA20_Ratio'] = v / v.rolling(20).mean()
    d['ADL'] = compute_adl(h, l, c, v)
    d['CMF_20'] = compute_cmf(h, l, c, v, 20)
    d['VWAP'] = (v * (h + l + c) / 3).cumsum() / v.cumsum()
    d['Vol_Change'] = v.pct_change()

    # --- Chart Patterns ---
    detector = ChartPatternDetector()
    patterns = detector.detect_all(df_raw)
    d = pd.concat([d, patterns], axis=1)
    # Composite pattern score
    pat_cols = [c for c in patterns.columns if c.startswith('Pat_')]
    d['Pattern_Score'] = patterns[pat_cols].sum(axis=1)

    # --- Calendar ---
    d['DayOfWeek'] = d.index.dayofweek / 4.0  # Normalize 0-1
    d['Month'] = d.index.month / 12.0
    d['Quarter'] = d.index.quarter / 4.0
    d['IsMonthEnd'] = d.index.is_month_end.astype(float)

    # --- Intermarket ---
    for name, series in intermarket_data.items():
        d = d.join(series, how='left')
        d[name] = d[name].ffill()
        d[f'{name}_Return'] = d[name].pct_change()

    # --- Fundamentals (static features, broadcast) ---
    for key, val in fundamentals.items():
        if not np.isnan(val) if isinstance(val, float) else val is not None:
            d[f'Fund_{key}'] = val

    # --- Sentiment proxy (for historical training) ---
    d['Sent_VolSpike'] = (d['Vol_SMA20_Ratio'] - 1).clip(-2, 5)
    d['Sent_RetMag'] = d['Return_1d'].abs().rolling(5).mean()
    d['Sent_Momentum'] = d['Return_5d'].rolling(3).mean()

    # Cleanup
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    d.dropna(inplace=True)

    feature_cols = [c for c in d.columns if c not in ['Open', 'High', 'Low']]
    P(f"\n  ✓ Total features: {len(feature_cols)}")
    P(f"  ✓ Dataset size: {len(d)} rows")
    return d, feature_cols

# ═══════════════════════════════════════════════════════════════════════
# 5. DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════
def prepare_data(df, feature_cols, seq_len, target='Close'):
    print(f"\n{'═'*70}\n  📐 Preparing sequences (len={seq_len})\n{'═'*70}")
    target_idx = feature_cols.index(target)
    data = df[feature_cols].values

    # Treat the difference from previous day as target
    target_prices = df[target].values
    target_diffs = np.zeros_like(target_prices)
    target_diffs[1:] = target_prices[1:] - target_prices[:-1]

    # MinMaxScaler — fit on full data for features
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Separate target scaler (scaled diffs)
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_target = target_scaler.fit_transform(target_diffs.reshape(-1, 1)).flatten()

    # Auto-adjust sequence length if data is too short
    if len(scaled) <= seq_len + 10:
        old_seq = seq_len
        seq_len = max(5, len(scaled) // 3)
        P(f"  ⚠ Only {len(scaled)} rows — reducing SEQ_LEN from {old_seq} to {seq_len}")

    # Build sequences
    X, y, prev_prices = [], [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled_target[i])
        prev_prices.append(target_prices[i-1])
    X, y, prev_prices = np.array(X), np.array(y), np.array(prev_prices)

    # Chronological split 70/15/15
    n = len(X)
    tr_end = int(n * 0.70)
    va_end = int(n * 0.85)

    X_tr, y_tr, prev_tr = X[:tr_end], y[:tr_end], prev_prices[:tr_end]
    X_va, y_va, prev_va = X[tr_end:va_end], y[tr_end:va_end], prev_prices[tr_end:va_end]
    X_te, y_te, prev_te = X[va_end:], y[va_end:], prev_prices[va_end:]
    n_feat = X.shape[2]

    # XGBoost features: last values + trend (avoid excessive dimensionality)
    def seq_to_flat(Xs):
        flat = []
        for seq in Xs:
            last = seq[-1]
            trend = seq[-1] - seq[0]
            flat.append(np.concatenate([last, trend]))
        return np.array(flat)

    X_tr_xgb = seq_to_flat(X_tr)
    X_va_xgb = seq_to_flat(X_va)
    X_te_xgb = seq_to_flat(X_te)

    P(f"  Sequence length: {seq_len}  |  Features: {n_feat}")
    P(f"  Train: {len(X_tr)}  |  Val: {len(X_va)}  |  Test: {len(X_te)}")
    P(f"  XGBoost features: {X_tr_xgb.shape[1]}")

    return {
        'X_tr': X_tr, 'y_tr': y_tr, 'prev_tr': prev_tr,
        'X_va': X_va, 'y_va': y_va, 'prev_va': prev_va,
        'X_te': X_te, 'y_te': y_te, 'prev_te': prev_te,
        'X_tr_xgb': X_tr_xgb, 'X_va_xgb': X_va_xgb, 'X_te_xgb': X_te_xgb,
        'scaler': scaler, 'target_scaler': target_scaler,
        'n_feat': n_feat, 'feature_cols': feature_cols, 'target_idx': target_idx,
        'seq_len': seq_len
    }

# ═══════════════════════════════════════════════════════════════════════
# 6. MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

# --- 6a: Custom Attention Layer ---
class AttentionLayer(Layer):
    """Bahdanau-style attention for LSTM sequences."""
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="att_W")
        self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="att_b")
        self.u = self.add_weight(shape=(self.units,), initializer="glorot_uniform", name="att_u")

    def call(self, x):
        score = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.u, axes=1), axis=1)
        context = tf.reduce_sum(x * tf.expand_dims(attention_weights, -1), axis=1)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# --- 6b: Attention-LSTM Model ---
def build_attention_lstm(seq_len, n_feat, lr=0.0005):
    print(f"\n{'═'*70}\n  🧠 Building Attention-LSTM Model\n{'═'*70}")
    inp = Input(shape=(seq_len, n_feat))
    x = Bidirectional(LSTM(64, return_sequences=True))(inp)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    # Attention
    att = AttentionLayer(units=32)(x)
    # Dense head
    x = Dense(32, activation='relu')(att)
    x = Dropout(0.1)(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="Attention_LSTM")
    model.compile(optimizer=Adam(learning_rate=lr), loss='mae', metrics=['mse'])
    P(f"  Parameters: {model.count_params():,}")
    return model

# --- 6c: Transformer Model ---
class PositionalEncoding(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_enc = self._get_pos_enc(seq_len, d_model)

    def _get_pos_enc(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, :, :], dtype=tf.float32)

    def call(self, x):
        return x + self.pos_enc[:, :tf.shape(x)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return config

def transformer_encoder_block(x, d_model, n_heads, ff_dim, dropout=0.1):
    # Multi-head attention
    attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)(x, x)
    attn = Dropout(dropout)(attn)
    x1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))
    # Feed-forward
    ff = Dense(ff_dim, activation='relu')(x1)
    ff = Dense(d_model)(ff)
    ff = Dropout(dropout)(ff)
    x2 = LayerNormalization(epsilon=1e-6)(Add()([x1, ff]))
    return x2

def build_transformer(seq_len, n_feat, lr=0.0005, d_model=64, n_heads=4, n_blocks=3):
    print(f"\n{'═'*70}\n  🤖 Building Transformer Model\n{'═'*70}")
    inp = Input(shape=(seq_len, n_feat))
    x = Dense(d_model)(inp)  # Project to d_model
    x = PositionalEncoding(seq_len, d_model)(x)
    for _ in range(n_blocks):
        x = transformer_encoder_block(x, d_model, n_heads, ff_dim=128, dropout=0.1)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation='relu')(x)
    out = Dense(1)(x)
    model = Model(inp, out, name="Transformer")
    model.compile(optimizer=Adam(learning_rate=lr), loss='mae', metrics=['mse'])
    P(f"  Parameters: {model.count_params():,}")
    return model

# --- 6d: XGBoost Model ---
def build_xgboost(X_tr, y_tr, X_va, y_va):
    print(f"\n{'═'*70}\n  🌳 Building XGBoost Model\n{'═'*70}")
    params = {
        'objective': 'reg:absoluteerror', 'max_depth': 8, 'learning_rate': 0.05,
        'n_estimators': 800, 'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 3, 'reg_alpha': 0.1, 'reg_lambda': 0.5,
        'gamma': 0.0, 'tree_method': 'hist', 'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              verbose=False)
    P(f"  XGBoost features: {X_tr.shape[1]}")
    return model

# ═══════════════════════════════════════════════════════════════════════
# 7. TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════
def train_dl_model(model, X_tr, y_tr, X_va, y_va, epochs, batch_size, patience):
    """Train a Keras model with callbacks."""
    name = model.name
    print(f"\n{'═'*70}\n  🏋️ Training {name} ...\n{'═'*70}")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1),
    ]
    history = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=callbacks, shuffle=False, verbose=1)
    P(f"  ✓ {name} training complete — {len(history.history['loss'])} epochs")
    return history

# ═══════════════════════════════════════════════════════════════════════
# 8. ENSEMBLE META-LEARNER
# ═══════════════════════════════════════════════════════════════════════
def build_ensemble(models_preds_val, y_val, models_preds_test):
    """Stack model predictions via Ridge regression."""
    print(f"\n{'═'*70}\n  🔗 Building Ensemble Meta-Learner\n{'═'*70}")
    X_meta_val = np.column_stack(models_preds_val)
    X_meta_test = np.column_stack(models_preds_test)
    meta = Ridge(alpha=1.0)
    meta.fit(X_meta_val, y_val)
    weights = meta.coef_ / meta.coef_.sum()
    P(f"  Ensemble weights: {dict(zip(['LSTM', 'Transformer', 'XGBoost'], weights.round(3)))}")
    ensemble_pred = meta.predict(X_meta_test)
    return ensemble_pred, meta, weights

# ═══════════════════════════════════════════════════════════════════════
# 9. EVALUATION
# ═══════════════════════════════════════════════════════════════════════
def evaluate_model(preds_scaled, y_scaled, target_scaler, prev_prices, label="Test"):
    """Evaluate predictions and print metrics."""
    preds_diff = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals_diff = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    preds = prev_prices + preds_diff
    actuals = prev_prices + actuals_diff
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    # Directional accuracy
    if len(actuals) > 1:
        dir_a = np.diff(actuals) > 0
        dir_p = np.diff(preds) > 0
        dir_acc = np.mean(dir_a == dir_p) * 100
    else:
        dir_acc = 0
    print(f"\n  ── {label} Metrics ──────────────────────────────────")
    print(f"  {'R² Score':<25} {r2:>12.4f}")
    print(f"  {'RMSE':<25} ₹{rmse:>11.2f}")
    print(f"  {'MAE':<25} ₹{mae:>11.2f}")
    print(f"  {'MAPE':<25} {mape:>11.2f}%")
    print(f"  {'Directional Accuracy':<25} {dir_acc:>11.2f}%")
    print(f"  {'─'*45}")
    return preds, actuals, {'r2': r2, 'rmse': rmse, 'mae': mae, 'mape': mape, 'dir_acc': dir_acc}

# ═══════════════════════════════════════════════════════════════════════
# 10. VISUALIZATION SUITE
# ═══════════════════════════════════════════════════════════════════════
def plot_all(histories, model_results, ensemble_preds, ensemble_actuals,
             ensemble_metrics, df, feature_cols, sentiment_data, pred_price):
    print(f"\n{'═'*70}\n  📈 Generating Visualizations\n{'═'*70}")
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle(f'{TICKER_NAME} ({TICKER}) — Advanced Multi-Model Prediction\n'
                 f'Ensemble R²={ensemble_metrics["r2"]:.4f}  |  MAPE={ensemble_metrics["mape"]:.2f}%  |  '
                 f'Dir. Acc={ensemble_metrics["dir_acc"]:.1f}%',
                 fontsize=14, fontweight='bold', y=0.98)

    # 1. Loss curves
    ax1 = fig.add_subplot(3, 3, 1)
    colors = ['#2196F3', '#E53935']
    for (name, hist), c in zip(histories.items(), colors):
        ax1.plot(hist.history['loss'], label=f'{name} Train', lw=1.5, alpha=0.8)
        ax1.plot(hist.history['val_loss'], label=f'{name} Val', lw=1.5, ls='--', alpha=0.8)
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.legend(fontsize=7); ax1.grid(alpha=0.3)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE')

    # 2. Model comparison
    ax2 = fig.add_subplot(3, 3, 2)
    model_names = list(model_results.keys()) + ['Ensemble']
    r2_vals = [m['metrics']['r2'] for m in model_results.values()] + [ensemble_metrics['r2']]
    bars = ax2.bar(model_names, r2_vals, color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'])
    ax2.set_title('R² Score Comparison', fontweight='bold')
    ax2.set_ylim(min(r2_vals) - 0.02, 1.01)
    for bar, val in zip(bars, r2_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.4f}', ha='center', fontsize=8)
    ax2.grid(alpha=0.3, axis='y')

    # 3. Actual vs Predicted (Ensemble)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(ensemble_actuals, label='Actual', lw=2, c='#1565C0')
    ax3.plot(ensemble_preds, label='Predicted', lw=2, c='#E53935', alpha=0.85)
    ax3.fill_between(range(len(ensemble_preds)),
                      ensemble_preds * 0.97, ensemble_preds * 1.03,
                      alpha=0.15, color='red', label='±3% Band')
    ax3.set_title('Ensemble: Actual vs Predicted', fontweight='bold')
    ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

    # 4. Error distribution
    ax4 = fig.add_subplot(3, 3, 4)
    pct_err = ((ensemble_preds - ensemble_actuals) / ensemble_actuals) * 100
    ax4.hist(pct_err, bins=50, color='#7B1FA2', alpha=0.75, edgecolor='white')
    ax4.axvline(0, c='red', lw=2, ls='--')
    ax4.set_title(f'Error Distribution (μ={np.mean(pct_err):.2f}%)', fontweight='bold')
    ax4.set_xlabel('Error (%)'); ax4.grid(alpha=0.3)

    # 5. Directional accuracy comparison
    ax5 = fig.add_subplot(3, 3, 5)
    dir_vals = [m['metrics']['dir_acc'] for m in model_results.values()] + [ensemble_metrics['dir_acc']]
    bars = ax5.bar(model_names, dir_vals, color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'])
    ax5.axhline(50, c='red', ls='--', lw=1, label='Random (50%)')
    ax5.set_title('Directional Accuracy %', fontweight='bold')
    for bar, val in zip(bars, dir_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', fontsize=8)
    ax5.legend(); ax5.grid(alpha=0.3, axis='y')

    # 6. MAPE comparison
    ax6 = fig.add_subplot(3, 3, 6)
    mape_vals = [m['metrics']['mape'] for m in model_results.values()] + [ensemble_metrics['mape']]
    bars = ax6.bar(model_names, mape_vals, color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'])
    ax6.set_title('MAPE % (lower = better)', fontweight='bold')
    for bar, val in zip(bars, mape_vals):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}%', ha='center', fontsize=8)
    ax6.grid(alpha=0.3, axis='y')

    # 7. Full timeline
    ax7 = fig.add_subplot(3, 1, 3)
    dates = df.index
    closes = df['Close'].values
    ax7.plot(dates, closes, lw=1.2, c='#90CAF9', alpha=0.6, label='Historical')
    test_n = len(ensemble_preds)
    test_dates = dates[-test_n - min(SEQ_LEN, len(dates) - test_n):][-test_n:]
    if len(test_dates) == len(ensemble_preds):
        ax7.plot(test_dates, ensemble_preds, lw=2, c='#E53935', label='Ensemble Prediction')
    # Next day marker
    if pred_price and len(test_dates) > 0:
        next_date = test_dates[-1] + pd.Timedelta(days=1)
        ax7.scatter([next_date], [pred_price], s=150, c='gold',
                    edgecolors='black', zorder=5, label=f'Next: ₹{pred_price:.2f}')
    ax7.set_title('Full Price Timeline + Forecast', fontweight='bold')
    ax7.legend(fontsize=9, loc='upper left'); ax7.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = os.path.join(MODEL_DIR, f'{TICKER.replace(".", "_")}_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    # plt.show() # Automatically showing graphs is enabled
    P(f"  ✓ Saved → {save_path}")

    # --- Sentiment Dashboard ---
    if sentiment_data and sentiment_data.get('count', 0) > 0:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle(f'{TICKER_NAME} — Live Sentiment Dashboard', fontweight='bold', fontsize=13)
        # Sentiment distribution
        scores = [i['score'] for i in sentiment_data['items']]
        axes2[0].hist(scores, bins=20, color='#00BCD4', alpha=0.8, edgecolor='white')
        axes2[0].axvline(0, c='gray', ls='--', lw=1.5)
        axes2[0].axvline(sentiment_data['overall'], c='red', lw=2, label=f"Avg: {sentiment_data['overall']:.3f}")
        axes2[0].set_title('Sentiment Score Distribution'); axes2[0].legend(); axes2[0].grid(alpha=0.3)
        # Source breakdown
        sources = {}
        for item in sentiment_data['items']:
            src = item['source']
            sources.setdefault(src, []).append(item['score'])
        src_names = list(sources.keys())
        src_avgs = [np.mean(v) for v in sources.values()]
        colors = ['#4CAF50' if v > 0 else '#F44336' for v in src_avgs]
        axes2[1].barh(src_names, src_avgs, color=colors)
        axes2[1].axvline(0, c='gray', ls='--')
        axes2[1].set_title('Sentiment by Source'); axes2[1].grid(alpha=0.3)
        plt.tight_layout()
        sent_path = os.path.join(MODEL_DIR, f'{TICKER.replace(".", "_")}_sentiment.png')
        plt.savefig(sent_path, dpi=150, bbox_inches='tight')
        # plt.show() # Automatically showing graphs is enabled
        P(f"  ✓ Saved → {sent_path}")

# ═══════════════════════════════════════════════════════════════════════
# 11. NEXT-DAY PREDICTION & TRADING SIGNAL
# ═══════════════════════════════════════════════════════════════════════
def predict_next_day(models, xgb_model, meta_learner, df, data_dict, sentiment_data):
    """Ensemble prediction for next trading day."""
    print(f"\n{'═'*70}\n  ★ NEXT TRADING DAY PREDICTION\n{'═'*70}")
    scaler = data_dict['scaler']
    target_scaler = data_dict['target_scaler']
    feature_cols = data_dict['feature_cols']

    actual_seq_len = data_dict.get('seq_len', SEQ_LEN)
    recent = df[feature_cols].values[-actual_seq_len:]
    recent_scaled = scaler.transform(recent)
    X_seq = recent_scaled.reshape(1, actual_seq_len, len(feature_cols))

    # Individual predictions
    preds = []
    for name, model in models.items():
        p = model.predict(X_seq, verbose=0).flatten()[0]
        preds.append(p)
        P(f"  {name}: {target_scaler.inverse_transform([[p]])[0][0]:.2f}")

    # XGBoost prediction
    last = recent_scaled[-1]
    trend = recent_scaled[-1] - recent_scaled[0]
    X_xgb = np.concatenate([last, trend]).reshape(1, -1)
    p_xgb = xgb_model.predict(X_xgb)[0]
    preds.append(p_xgb)
    P(f"  XGBoost: {target_scaler.inverse_transform([[p_xgb]])[0][0]:.2f}")

    # Ensemble
    X_meta = np.array(preds).reshape(1, -1)
    ensemble_pred_scaled = meta_learner.predict(X_meta)[0]
    pred_diff = target_scaler.inverse_transform([[ensemble_pred_scaled]])[0][0]

    last_close = df['Close'].iloc[-1]
    pred_price = last_close + pred_diff

    change_pct = (pred_price - last_close) / last_close * 100

    # Sentiment adjustment
    sent_adj = 0
    if sentiment_data and sentiment_data.get('overall', 0) != 0:
        sent_adj = sentiment_data['overall'] * 0.5  # Mild adjustment
        pred_price_adj = pred_price * (1 + sent_adj / 100)
    else:
        pred_price_adj = pred_price

    change_pct_adj = (pred_price_adj - last_close) / last_close * 100

    # Trading signal — Weighted scoring system
    model_dirs = [1 if p > data_dict['y_te'][-1] else -1 for p in preds]
    agreement = sum(model_dirs) / len(model_dirs)
    
    # Bullish score: combine model agreement + predicted change direction
    bull_score = (agreement * 50) + (change_pct_adj * 30)  # weighted composite
    
    allocate_pct = 0
    if agreement > 0.3 and change_pct_adj > 0.15:
        signal = "🟢 STRONG BUY"
        confidence = min(abs(agreement) * 100 + (change_pct_adj * 15), 95)
        allocate_pct = int(min(confidence, 80))
    elif agreement > 0.0 and change_pct_adj > 0.0:
        signal = "🟢 BUY"
        confidence = min(abs(agreement) * 70 + (change_pct_adj * 20), 85)
        allocate_pct = int(min(confidence, 60))
    elif agreement > 0.3 and change_pct_adj > -0.15:
        signal = "🟢 BUY"
        confidence = min(abs(agreement) * 60, 75)
        allocate_pct = int(min(confidence, 50))
    elif agreement < -0.3 and change_pct_adj < -0.15:
        signal = "🔴 SELL"
        confidence = min(abs(agreement) * 100 + abs(change_pct_adj * 15), 95)
    elif agreement < 0.0 and change_pct_adj < -0.2:
        signal = "🔴 SELL"
        confidence = min(abs(agreement) * 70 + abs(change_pct_adj * 20), 85)
    else:
        signal = "🟡 HOLD"
        confidence = (1 - abs(agreement)) * 60

    allocation_str = f"Invest {allocate_pct}% (${10.0 * (allocate_pct/100.0):.2f} of your $10)" if allocate_pct > 0 else "Do not allocate new capital."

    # --- 20-DAY SWING TRADE TARGET & STOP LOSS ---
    lookback = min(20, len(df) - 1)
    recent_highs = df['High'].values[-lookback:]
    recent_lows = df['Low'].values[-lookback:]
    recent_closes = df['Close'].values[-(lookback+1):-1]
    # Ensure matching sizes
    min_len = min(len(recent_highs), len(recent_lows), len(recent_closes))
    recent_highs = recent_highs[-min_len:]
    recent_lows = recent_lows[-min_len:]
    recent_closes = recent_closes[-min_len:]
    
    tr1 = recent_highs - recent_lows
    tr2 = np.abs(recent_highs - recent_closes)
    tr3 = np.abs(recent_lows - recent_closes)
    atr = np.mean(np.maximum(tr1, np.maximum(tr2, tr3)))
    
    expected_20d_move = atr * 4.47  # approx sqrt(20) for random walk distance
    
    if "BUY" in signal:
        entry = last_close
        target = last_close + expected_20d_move
        stop_loss = last_close - (2.0 * atr)
        rr = (target - entry) / (entry - stop_loss)
        swing_setup = f"📈 20-DAY LONG SETUP: Buy near ₹{entry:.2f} | Target: ₹{target:.2f} | Stop Loss: ₹{stop_loss:.2f} | RR: {rr:.1f}"
    elif "SELL" in signal:
        entry = last_close
        target = max(0.1, last_close - expected_20d_move)
        stop_loss = last_close + (2.0 * atr)
        rr = (entry - target) / (stop_loss - entry)
        swing_setup = f"📉 20-DAY SHORT SETUP: Sell/Short near ₹{entry:.2f} | Target: ₹{target:.2f} | Stop Loss: ₹{stop_loss:.2f} | RR: {rr:.1f}"
    else:
        swing_setup = f"⚖️ NO 20-DAY SETUP: Market neutral/ranging. Keep capital safe."

    last_date = df.index[-1].date()
    print(f"""
  {'═'*50}
  Last trading day   : {last_date}
  Last close price   : ₹{last_close:.2f}
  ─────────────────────────────────────────────
  ★ Ensemble prediction    : ₹{pred_price:.2f}
  ★ Sentiment-adjusted     : ₹{pred_price_adj:.2f}
  ★ Expected change        : {change_pct_adj:+.2f}%
  ★ Sentiment score        : {sentiment_data.get('overall', 0):.3f}
  ─────────────────────────────────────────────
  ★ SIGNAL: {signal}  (Confidence: {confidence:.0f}%)
  ★ ACTION: {allocation_str}
  ★ Model Agreement: {agreement:+.2f}  ({sum(1 for d in model_dirs if d>0)}/{len(model_dirs)} bullish)
  ─────────────────────────────────────────────
  {swing_setup}
  {'═'*50}
  ⚠  This is a MODEL PREDICTION, NOT financial advice!
  ⚠  Always use stop-losses and position sizing!
  {'═'*50}
""")
    return pred_price_adj

# ═══════════════════════════════════════════════════════════════════════
# 12. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════
def main():
    start_time = time.time()
    print(f"""
{'═'*70}
  ★ ADVANCED MULTI-MODEL STOCK PREDICTOR ★
  Ticker: {TICKER} ({TICKER_NAME})
  Models: Attention-LSTM + Transformer + XGBoost (Ensemble)
  Features: 60+ (Technical + Fundamental + Sentiment + Patterns)
{'═'*70}""")

    # 1) Fetch data
    df_raw = fetch_ohlcv(TICKER, START_DATE)
    intermarket = fetch_intermarket(START_DATE)
    fundamentals = fetch_fundamentals(TICKER)

    # 2) Feature engineering
    df, feature_cols = engineer_features(df_raw, intermarket, fundamentals)

    # 3) Prepare data
    data = prepare_data(df, feature_cols, SEQ_LEN)

    # 4) Build models (use dynamic seq_len for short-history stocks)
    actual_seq_len = data.get('seq_len', SEQ_LEN)
    lstm_model = build_attention_lstm(actual_seq_len, data['n_feat'], LR)
    transformer_model = build_transformer(actual_seq_len, data['n_feat'], LR)

    # 5) Train deep learning models
    hist_lstm = train_dl_model(lstm_model, data['X_tr'], data['y_tr'],
                               data['X_va'], data['y_va'], EPOCHS, BATCH_SIZE, PATIENCE)
    hist_trans = train_dl_model(transformer_model, data['X_tr'], data['y_tr'],
                                data['X_va'], data['y_va'], EPOCHS, BATCH_SIZE, PATIENCE)

    # 6) Train XGBoost
    xgb_model = build_xgboost(data['X_tr_xgb'], data['y_tr'], data['X_va_xgb'], data['y_va'])

    # 7) Get validation predictions for ensemble
    val_pred_lstm = lstm_model.predict(data['X_va'], verbose=0).flatten()
    val_pred_trans = transformer_model.predict(data['X_va'], verbose=0).flatten()
    val_pred_xgb = xgb_model.predict(data['X_va_xgb'])

    # 8) Get test predictions
    test_pred_lstm = lstm_model.predict(data['X_te'], verbose=0).flatten()
    test_pred_trans = transformer_model.predict(data['X_te'], verbose=0).flatten()
    test_pred_xgb = xgb_model.predict(data['X_te_xgb'])

    # 9) Evaluate individual models
    model_results = {}
    P("\n" + "═"*70)
    P("📊 Individual Model Results")
    P("═"*70)
    for name, preds in [("Attention-LSTM", test_pred_lstm),
                         ("Transformer", test_pred_trans),
                         ("XGBoost", test_pred_xgb)]:
        p, a, m = evaluate_model(preds, data['y_te'], data['target_scaler'], data['prev_te'], name)
        model_results[name] = {'preds': p, 'actuals': a, 'metrics': m}

    # 10) Build ensemble
    ensemble_preds_scaled, meta_learner, weights = build_ensemble(
        [val_pred_lstm, val_pred_trans, val_pred_xgb], data['y_va'],
        [test_pred_lstm, test_pred_trans, test_pred_xgb]
    )
    P("\n" + "═"*70)
    P("🏆 ENSEMBLE Results")
    P("═"*70)
    ens_preds, ens_actuals, ens_metrics = evaluate_model(
        ensemble_preds_scaled, data['y_te'], data['target_scaler'], data['prev_te'], "ENSEMBLE"
    )

    # 11) Sentiment analysis
    scraper = SentimentScraper(TICKER_NAME)
    sentiment = scraper.get_sentiment_summary()

    # 12) Next-day prediction
    dl_models = {"Attention-LSTM": lstm_model, "Transformer": transformer_model}
    pred_price = predict_next_day(dl_models, xgb_model, meta_learner, df, data, sentiment)

    # 13) Visualizations
    histories = {"LSTM": hist_lstm, "Transformer": hist_trans}
    plot_all(histories, model_results, ens_preds, ens_actuals,
             ens_metrics, df, feature_cols, sentiment, pred_price)

    # 14) Save models
    lstm_model.save(os.path.join(MODEL_DIR, "attention_lstm.keras"))
    transformer_model.save(os.path.join(MODEL_DIR, "transformer.keras"))
    xgb_model.save_model(os.path.join(MODEL_DIR, "xgboost.json"))
    P("✓ All models saved")

    elapsed = time.time() - start_time
    n = len(data['X_tr']) + len(data['X_va']) + len(data['X_te'])
    print(f"""
{'═'*70}
  ★ FINAL SUMMARY — {TICKER_NAME} ({TICKER})
{'═'*70}
  Data source      : Yahoo Finance (yfinance)
  Training period  : {START_DATE} → {df.index.max().date()}
  Total samples    : {n} ({len(data['X_tr'])} train / {len(data['X_va'])} val / {len(data['X_te'])} test)
  Features         : {data['n_feat']}
  Sequence length  : {SEQ_LEN} days

  ── Individual Models ──
  Attention-LSTM   : R²={model_results['Attention-LSTM']['metrics']['r2']:.4f}  MAPE={model_results['Attention-LSTM']['metrics']['mape']:.2f}%  Dir={model_results['Attention-LSTM']['metrics']['dir_acc']:.1f}%
  Transformer      : R²={model_results['Transformer']['metrics']['r2']:.4f}  MAPE={model_results['Transformer']['metrics']['mape']:.2f}%  Dir={model_results['Transformer']['metrics']['dir_acc']:.1f}%
  XGBoost          : R²={model_results['XGBoost']['metrics']['r2']:.4f}  MAPE={model_results['XGBoost']['metrics']['mape']:.2f}%  Dir={model_results['XGBoost']['metrics']['dir_acc']:.1f}%

  ── 🏆 ENSEMBLE ──
  R² Score         : {ens_metrics['r2']:.4f}
  MAPE             : {ens_metrics['mape']:.2f}%
  RMSE             : ₹{ens_metrics['rmse']:.2f}
  MAE              : ₹{ens_metrics['mae']:.2f}
  Directional Acc  : {ens_metrics['dir_acc']:.1f}%

  ── Sentiment ──
  Overall Score    : {sentiment.get('overall', 0):.3f}
  Data Points      : {sentiment.get('count', 0)}

  ── Next-Day Forecast ──
  Predicted        : ₹{pred_price:.2f}

  Time elapsed     : {elapsed:.1f}s
{'═'*70}
  ⚠  Predictions ≠ financial advice. Use at your own risk.
{'═'*70}
""")

if __name__ == "__main__":
    main()
