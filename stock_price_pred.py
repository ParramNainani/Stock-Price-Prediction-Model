"""
==============================================================================
  Tesla (TSLA) Stock Price Prediction — Real-Time LSTM Model
==============================================================================
  • Fetches LIVE data via yfinance (Yahoo Finance)
  • Multivariate LSTM with carefully selected technical indicators
  • Predicts the NEXT TRADING DAY closing price
  • Professional evaluation & visualizations
  • Saves model for reuse
==============================================================================
"""

import os, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # suppress TF info logs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense, LSTM, Dropout, Bidirectional,
    BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TICKER        = "TSLA"
START_DATE    = "2015-01-01"      # enough history for robust training
SEQUENCE_LEN  = 60                # look-back window (trading days)
EPOCHS        = 200
BATCH_SIZE    = 32
PATIENCE      = 20                # early-stopping patience
LR            = 0.001
MODEL_PATH    = "tsla_lstm_model.keras"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. FETCH LIVE DATA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def fetch_data(ticker: str, start: str) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    print(f"\n{'='*70}")
    print(f"  Fetching LIVE data for {ticker} from {start} to today ...")
    print(f"{'='*70}")

    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    # yfinance sometimes returns multi-level columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.sort_index()
    print(f"  ✓ Downloaded {len(df)} trading days")
    print(f"  ✓ Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  ✓ Latest close: ${df['Close'].iloc[-1]:.2f}")
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. FEATURE ENGINEERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)
    avg_g = gain.rolling(period).mean()
    avg_l = loss.rolling(period).mean()
    rs    = avg_g / avg_l
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators — keep it focused to avoid noise."""
    print(f"\n{'='*70}")
    print("  Engineering features ...")
    print(f"{'='*70}")

    d = df.copy()

    # Moving Averages
    d['SMA_10']  = d['Close'].rolling(10).mean()
    d['SMA_20']  = d['Close'].rolling(20).mean()
    d['EMA_12']  = d['Close'].ewm(span=12, adjust=False).mean()
    d['EMA_26']  = d['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    d['MACD']    = d['EMA_12'] - d['EMA_26']

    # RSI
    d['RSI_14']  = compute_rsi(d['Close'], 14)

    # Bollinger Band width (normalised volatility)
    bb_mid       = d['Close'].rolling(20).mean()
    bb_std       = d['Close'].rolling(20).std()
    d['BB_Width'] = (2 * bb_std) / bb_mid

    # Average True Range (volatility)
    hl = d['High'] - d['Low']
    hc = (d['High'] - d['Close'].shift()).abs()
    lc = (d['Low']  - d['Close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d['ATR_14'] = tr.rolling(14).mean()

    # Daily returns & volume change
    d['Return']     = d['Close'].pct_change()
    d['Vol_Change'] = d['Volume'].pct_change()

    # Price relative to SMAs (mean-reversion signals)
    d['Close_SMA10_Ratio'] = d['Close'] / d['SMA_10']
    d['Close_SMA20_Ratio'] = d['Close'] / d['SMA_20']

    d.dropna(inplace=True)
    d.replace([np.inf, -np.inf], 0, inplace=True)

    print(f"  ✓ Added {len(d.columns) - 5} technical indicators")
    print(f"  ✓ Final features: {list(d.columns)}")
    print(f"  ✓ Dataset size after cleaning: {len(d)} rows")
    return d


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. PREPARE DATA  (sequences + train / val / test split)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def prepare_data(df: pd.DataFrame, seq_len: int):
    print(f"\n{'='*70}")
    print("  Preparing sequences ...")
    print(f"{'='*70}")

    feature_cols = list(df.columns)
    target_col   = 'Close'
    target_idx   = feature_cols.index(target_col)

    data = df[feature_cols].values

    # --- Scale ---
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    # Separate scaler for target (inverse-transform later)
    target_scaler = MinMaxScaler()
    target_scaler.fit_transform(df[[target_col]].values)

    # --- Build sequences ---
    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len : i])
        y.append(scaled[i, target_idx])
    X = np.array(X)
    y = np.array(y)

    # --- Chronological split  80 / 10 / 10 ---
    n           = len(X)
    train_end   = int(n * 0.80)
    val_end     = int(n * 0.90)

    X_train, y_train = X[:train_end],          y[:train_end]
    X_val,   y_val   = X[train_end:val_end],   y[train_end:val_end]
    X_test,  y_test  = X[val_end:],            y[val_end:]

    n_features = X.shape[2]
    print(f"  ✓ Sequence length : {seq_len}")
    print(f"  ✓ Num features    : {n_features}")
    print(f"  ✓ Train / Val / Test : {len(X_train)} / {len(X_val)} / {len(X_test)}  samples")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            scaler, target_scaler, n_features, feature_cols, target_idx)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. BUILD MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_model(seq_len: int, n_features: int, lr: float = 0.001):
    print(f"\n{'='*70}")
    print("  Building Bidirectional LSTM model ...")
    print(f"{'='*70}")

    model = Sequential([
        Input(shape=(seq_len, n_features)),

        Bidirectional(LSTM(100, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.25),

        Bidirectional(LSTM(80, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.25),

        LSTM(60, return_sequences=False),
        BatchNormalization(),
        Dropout(0.20),

        Dense(40, activation='relu'),
        Dropout(0.10),
        Dense(20, activation='relu'),
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae'],
    )
    model.summary()
    return model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. TRAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_model(model, X_train, y_train, X_val, y_val,
                epochs, batch_size, patience):
    print(f"\n{'='*70}")
    print("  Training ...")
    print(f"{'='*70}")

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        shuffle=False,   # preserve time order
        verbose=1,
    )
    print(f"\n  ✓ Training done — best epoch ~{len(history.history['loss'])}")
    return history


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. EVALUATE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def evaluate(model, X_test, y_test, target_scaler, label="Test"):
    preds_scaled = model.predict(X_test, verbose=0)
    preds   = target_scaler.inverse_transform(preds_scaled).flatten()
    actuals = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mse  = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(actuals, preds)
    r2   = r2_score(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

    # Directional accuracy
    dir_actual = np.diff(actuals) > 0
    dir_pred   = np.diff(preds) > 0
    dir_acc    = np.mean(dir_actual == dir_pred) * 100

    print(f"\n  ── {label} Set Metrics ──────────────────────────────")
    print(f"  {'MSE':<30} {mse:>12.2f}")
    print(f"  {'RMSE':<30} ${rmse:>11.2f}")
    print(f"  {'MAE':<30} ${mae:>11.2f}")
    print(f"  {'R² Score':<30} {r2:>12.4f}")
    print(f"  {'MAPE':<30} {mape:>11.2f}%")
    print(f"  {'Directional Accuracy':<30} {dir_acc:>11.2f}%")
    print(f"  {'─'*50}")

    return preds, actuals, dict(mse=mse, rmse=rmse, mae=mae,
                                r2=r2, mape=mape, dir_acc=dir_acc)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. PREDICT NEXT TRADING DAY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def predict_next_day(model, df, scaler, target_scaler,
                     feature_cols, seq_len):
    """Use the most recent `seq_len` days to predict tomorrow's close."""
    print(f"\n{'='*70}")
    print("  ★ REAL-TIME PREDICTION — Next Trading Day")
    print(f"{'='*70}")

    recent = df[feature_cols].values[-seq_len:]
    recent_scaled = scaler.transform(recent)
    X_pred = recent_scaled.reshape(1, seq_len, len(feature_cols))

    pred_scaled = model.predict(X_pred, verbose=0)
    pred_price  = target_scaler.inverse_transform(pred_scaled)[0, 0]

    last_close = df['Close'].iloc[-1]
    change_pct = (pred_price - last_close) / last_close * 100
    direction  = "📈 UP" if change_pct > 0 else "📉 DOWN"

    last_date = df.index[-1].date()

    print(f"\n  Last trading day : {last_date}")
    print(f"  Last close price : ${last_close:.2f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  ★ Predicted next close : ${pred_price:.2f}")
    print(f"  ★ Expected change      : {change_pct:+.2f}%  {direction}")
    print(f"  ─────────────────────────────────────────")
    print(f"\n  ⚠  This is a model prediction, NOT financial advice!")

    return pred_price, last_close, change_pct


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. VISUALIZATIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def plot_results(history, preds, actuals, metrics, df,
                 seq_len, train_end, val_end, pred_price):

    print(f"\n{'='*70}")
    print("  Generating visualizations ...")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f'TSLA Stock Prediction — Live LSTM Model  |  '
                 f'R²={metrics["r2"]:.4f}   MAPE={metrics["mape"]:.2f}%',
                 fontsize=15, fontweight='bold')

    # ---- 1. Loss curves ----
    ax = axes[0, 0]
    ax.plot(history.history['loss'],     label='Train Loss', lw=2, c='#2196F3')
    ax.plot(history.history['val_loss'],  label='Val Loss',   lw=2, c='#FF5722')
    ax.set_title('Training & Validation Loss', fontweight='bold')
    ax.set_xlabel('Epoch');  ax.set_ylabel('MSE Loss')
    ax.legend();  ax.grid(alpha=.3)

    # ---- 2. Actual vs Predicted ----
    ax = axes[0, 1]
    ax.plot(actuals, label='Actual',    lw=2, c='#1565C0')
    ax.plot(preds,   label='Predicted', lw=2, c='#E53935', alpha=.85)
    ax.set_title('Actual vs Predicted — Test Set', fontweight='bold')
    ax.set_xlabel('Days (Test Set)');  ax.set_ylabel('Price (USD)')
    ax.legend();  ax.grid(alpha=.3)

    # ---- 3. Error distribution ----
    ax = axes[1, 0]
    pct_err = ((preds - actuals) / actuals) * 100
    ax.hist(pct_err, bins=50, color='#7B1FA2', alpha=.75, edgecolor='white')
    ax.axvline(0, c='red', lw=2, ls='--')
    ax.set_title(f'% Error Distribution  (mean={np.mean(pct_err):.2f}%)',
                 fontweight='bold')
    ax.set_xlabel('Error (%)');  ax.set_ylabel('Count')
    ax.grid(alpha=.3)

    # ---- 4. Full timeline ----
    ax = axes[1, 1]
    dates    = df.index
    closes   = df['Close'].values
    test_start_idx = seq_len + val_end
    test_dates = dates[test_start_idx : test_start_idx + len(preds)]

    ax.plot(dates, closes, lw=1.2, c='#90CAF9', alpha=.6, label='Historical')
    ax.plot(test_dates, preds, lw=2, c='#E53935', label='Test Predictions')

    # Mark next-day prediction
    import matplotlib.dates as mdates
    next_date = test_dates[-1] + pd.Timedelta(days=1)
    ax.scatter([next_date], [pred_price], s=120, c='gold',
               edgecolors='black', zorder=5, label=f'Next day: ${pred_price:.2f}')

    ax.set_title('Full Price Timeline + Next-Day Forecast', fontweight='bold')
    ax.set_xlabel('Date');  ax.set_ylabel('Price (USD)')
    ax.legend(fontsize=9, loc='upper left');  ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig('tsla_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✓ Saved → tsla_prediction_results.png")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main():
    # 1) Fetch live data
    df_raw = fetch_data(TICKER, START_DATE)

    # 2) Feature engineering
    df = add_features(df_raw)

    # 3) Prepare sequences + split
    (X_train, y_train, X_val, y_val, X_test, y_test,
     scaler, target_scaler, n_features, feature_cols, target_idx
    ) = prepare_data(df, SEQUENCE_LEN)

    # Split indices (for plotting)
    n          = len(X_train) + len(X_val) + len(X_test)
    train_end  = len(X_train)
    val_end    = train_end + len(X_val)

    # 4) Build model
    model = build_model(SEQUENCE_LEN, n_features, LR)

    # 5) Train
    history = train_model(model, X_train, y_train, X_val, y_val,
                          EPOCHS, BATCH_SIZE, PATIENCE)

    # 6) Evaluate (test + validation)
    preds, actuals, metrics = evaluate(model, X_test, y_test,
                                        target_scaler, "Test")
    evaluate(model, X_val, y_val, target_scaler, "Validation")

    # 7) Next-day prediction
    pred_price, last_close, change_pct = predict_next_day(
        model, df, scaler, target_scaler, feature_cols, SEQUENCE_LEN
    )

    # 8) Visualize
    plot_results(history, preds, actuals, metrics, df,
                 SEQUENCE_LEN, train_end, val_end, pred_price)

    # 9) Save model
    model.save(MODEL_PATH)
    print(f"\n  ✓ Model saved → {MODEL_PATH}")

    # ── SUMMARY ──
    print(f"""
{'='*70}
  FINAL SUMMARY — TSLA Real-Time LSTM Predictor
{'='*70}
  Data source     : Yahoo Finance (yfinance)
  Ticker          : {TICKER}
  Training period : {START_DATE} → {df.index.max().date()}
  Samples         : {n} total  ({len(X_train)} train / {len(X_val)} val / {len(X_test)} test)
  Features        : {n_features}
  Sequence length : {SEQUENCE_LEN} days
  Model params    : {model.count_params():,}

  Test R² Score          : {metrics['r2']:.4f}
  Test MAPE              : {metrics['mape']:.2f}%
  Test RMSE              : ${metrics['rmse']:.2f}
  Test MAE               : ${metrics['mae']:.2f}
  Directional Accuracy   : {metrics['dir_acc']:.2f}%

  ★ NEXT-DAY PREDICTION ★
  Last close  : ${last_close:.2f}
  Predicted   : ${pred_price:.2f}  ({change_pct:+.2f}%)
{'='*70}
  ⚠  Model predictions ≠ financial advice. Use at your own risk.
{'='*70}
""")


if __name__ == "__main__":
    main()