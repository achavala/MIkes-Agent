#!/usr/bin/env python3
"""
Validate the trained historical model.

- Default: file + load + inference smoke test
- Optional: run a short offline evaluation backtest on recent historical bars
  and report trades/day + worst trade loss (sanity-check the -15% constraint).
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def _load_local_env() -> None:
    """Minimal .env loader (no external deps)."""
    candidates = [
        Path(os.getcwd()) / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    for p in candidates:
        try:
            if not p.exists():
                continue
            for raw in p.read_text(errors="ignore").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k or not v:
                    continue
                if (k not in os.environ) or (not str(os.environ.get(k, "")).strip()):
                    os.environ[k] = v
        except Exception:
            continue

try:
    from stable_baselines3 import PPO
    from gymnasium import spaces
    RL_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error: {e}")
    print("   Install with: pip install stable-baselines3 gymnasium")
    sys.exit(1)


def _load_model(model_path: str):
    """Load PPO/MaskablePPO model from disk."""
    # If model was trained with MaskablePPO, loading with PPO can fail.
    try:
        from sb3_contrib import MaskablePPO  # type: ignore
        return MaskablePPO.load(model_path)
    except Exception:
        return PPO.load(model_path)


def _load_symbol_data(enriched_dir: Path, symbol: str) -> pd.DataFrame:
    """Load OHLCV from latest enriched pickle for symbol."""
    enriched_files = sorted(enriched_dir.glob(f"{symbol}_enriched_*.pkl"))
    if not enriched_files:
        raise FileNotFoundError(f"No enriched data file found for {symbol} in {enriched_dir}")
    enriched_file = enriched_files[-1]
    with open(enriched_file, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"Invalid data format in {enriched_file.name}")
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} missing required cols: {missing}")
    out = df[required_cols].copy()
    return out


def offline_evaluate(
    model_path: str,
    symbols: list[str],
    enriched_dir: str = "data/historical/enriched",
    days: int = 15,
    deterministic: bool = True,
):
    """Run a lightweight offline backtest to sanity-check behavior."""
    from historical_training_system import HistoricalTradingEnv
    ActionMasker = None
    try:
        from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
    except Exception:
        ActionMasker = None

    print("=" * 80)
    print("🧪 OFFLINE EVALUATION (LIGHT BACKTEST)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Symbols: {symbols}")
    print(f"Enriched dir: {enriched_dir}")
    print(f"Days (approx): {days}")
    print()

    model = _load_model(model_path)
    enriched_path = Path(enriched_dir)

    # Minimal VIX series (env will default if empty); daily is fine for this check.
    vix = pd.Series([20.0], index=pd.date_range("2000-01-01", periods=1, freq="1D"))

    all_trades = []
    for sym in symbols:
        df = _load_symbol_data(enriched_path, sym)
        eval_df = None

        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            # Detect daily vs intraday
            is_daily = df.index[0].time() == pd.Timestamp("00:00:00").time()
            if is_daily:
                # For daily data, evaluate on a bigger window (default ~1 year)
                bars = max(int(days), 252)
                eval_df = df.tail(bars).copy()
            else:
                # Intraday: last ~N days (buffer for weekends/holidays)
                last_ts = df.index.max()
                start_ts = last_ts - pd.Timedelta(days=int(days * 2))
                eval_df = df.loc[df.index >= start_ts].copy()
        else:
            # Not time-indexed: just take tail window
            eval_df = df.tail(max(int(days), 2000)).copy()

        min_needed = 20 + 5  # window_size + a little room
        if len(eval_df) < min_needed:
            print(f"⚠️ {sym}: not enough data for evaluation ({len(eval_df)} rows < {min_needed}), skipping")
            continue

        env = HistoricalTradingEnv(
            data=eval_df,
            vix_data=vix,
            symbol=sym,
            window_size=20,
            initial_capital=100000.0,
            use_greeks=True,
            use_features=False,
            human_momentum_mode=True,
        )
        if ActionMasker is not None:
            env = ActionMasker(env, lambda e: e.action_masks())

        obs, _ = env.reset()
        done = False
        truncated = False
        steps = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(int(action))
            steps += 1
            if steps > len(eval_df) + 10:
                break

        # If still holding a position at the end, force-close so we can measure PnL.
        try:
            if getattr(env, "unwrapped", env).position is not None:
                last_price = float(eval_df["close"].iloc[-1])
                getattr(env, "unwrapped", env)._execute_exit(last_price, 20.0)
        except Exception:
            pass

        # Pull trades from env
        trade_hist = getattr(getattr(env, "unwrapped", env), "trade_history", [])
        for t in trade_hist:
            t2 = dict(t)
            t2["symbol"] = sym
            all_trades.append(t2)

        print(f"✅ {sym}: eval steps={steps:,} trades={len(trade_hist)}")

    if not all_trades:
        print("\n❌ No trades recorded in evaluation window.")
        return

    trades_df = pd.DataFrame(all_trades)
    worst = float(trades_df["pnl_pct"].min()) if "pnl_pct" in trades_df.columns else None
    avg = float(trades_df["pnl_pct"].mean()) if "pnl_pct" in trades_df.columns else None

    # Approx trades/day if durations are present; otherwise just print totals.
    print("\n" + "-" * 80)
    print(f"Trades total: {len(trades_df)}")
    if avg is not None:
        print(f"Avg trade pnl_pct: {avg:.2%}")
    if worst is not None:
        print(f"Worst trade pnl_pct: {worst:.2%} (target >= -15.0%)")
    print("-" * 80)

def validate_model(model_path: str):
    """Validate the trained model (file + load + inference smoke test)."""
    print("=" * 80)
    print("🔍 MODEL VALIDATION")
    print("=" * 80)
    print()
    
    # 1. Check file exists
    print("1️⃣ FILE VALIDATION")
    print("-" * 80)
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path) / 1024  # KB
    print(f"✅ Model file exists: {model_path}")
    print(f"   Size: {file_size:.1f} KB")
    print()
    
    # 2. Load model
    print("2️⃣ MODEL LOADING")
    print("-" * 80)
    try:
        print(f"   Loading model from {model_path}...")
        model = _load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"   Policy: {type(model.policy).__name__}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # 3. Check observation space
    print("3️⃣ OBSERVATION SPACE")
    print("-" * 80)
    obs_space = model.observation_space
    print(f"   Type: {type(obs_space).__name__}")
    print(f"   Shape: {obs_space.shape}")
    print(f"   Dtype: {obs_space.dtype}")
    
    if isinstance(obs_space, spaces.Box):
        print(f"   Low: {obs_space.low[0] if len(obs_space.low) > 0 else obs_space.low}")
        print(f"   High: {obs_space.high[0] if len(obs_space.high) > 0 else obs_space.high}")
    print()
    
    # 4. Check action space
    print("4️⃣ ACTION SPACE")
    print("-" * 80)
    action_space = model.action_space
    print(f"   Type: {type(action_space).__name__}")
    print(f"   Shape: {action_space.shape}")
    print(f"   Dtype: {action_space.dtype}")
    
    if isinstance(action_space, spaces.Box):
        print(f"   Low: {action_space.low}")
        print(f"   High: {action_space.high}")
    print()
    
    # 5. Test inference
    print("5️⃣ INFERENCE TEST")
    print("-" * 80)
    try:
        # Create sample observation matching expected shape
        if isinstance(obs_space, spaces.Box):
            obs_shape = obs_space.shape
            if len(obs_shape) == 2:
                # (window_size, features) - e.g., (20, 10) for 20 timesteps, 10 features
                sample_obs = np.random.randn(*obs_shape).astype(np.float32)
            elif len(obs_shape) == 1:
                # Flattened observation
                sample_obs = np.random.randn(*obs_shape).astype(np.float32)
            else:
                # 3D observation (batch, window, features)
                sample_obs = np.random.randn(*obs_shape).astype(np.float32)
        else:
            print("   ⚠️  Unsupported observation space type")
            return False
        
        print(f"   Sample observation shape: {sample_obs.shape}")
        
        # Test prediction
        action, _ = model.predict(sample_obs, deterministic=True)
        print(f"✅ Inference successful!")
        print(f"   Action output: {action}")
        print(f"   Action shape: {action.shape}")
        print(f"   Action type: {type(action)}")
        
        # Check if action is in valid range
        if isinstance(action_space, spaces.Box):
            if np.all(action >= action_space.low) and np.all(action <= action_space.high):
                print(f"   ✅ Action in valid range")
            else:
                print(f"   ⚠️  Action out of valid range!")
        print()
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 6. Summary
    print("6️⃣ SUMMARY")
    print("-" * 80)
    print("✅ MODEL VALIDATION: PASSED")
    print()
    print("Model is ready for integration!")
    print(f"   - File: {model_path}")
    print(f"   - Observation space: {obs_space.shape}")
    print(f"   - Action space: {action_space.shape}")
    print()
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/mike_historical_model.zip", help="Path to model .zip")
    parser.add_argument("--offline-eval", action="store_true", help="Run lightweight offline evaluation")
    parser.add_argument("--symbols", default="SPY,QQQ,SPX", help="Comma-separated symbols for offline eval")
    parser.add_argument("--enriched-dir", default="data/historical/enriched", help="Directory with *_enriched_*.pkl files")
    parser.add_argument("--days", type=int, default=15, help="Approx days of data to evaluate")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions in eval (default deterministic)")
    parser.add_argument("--intraday", action="store_true", help="Evaluate on intraday 1m bars via Massive/Polygon instead of enriched daily pickles")
    parser.add_argument("--intraday-days", type=int, default=10, help="When --intraday: calendar days of 1m bars to load")
    parser.add_argument("--data-source", type=str, default="massive", choices=["massive", "polygon"], help="When --intraday: which source to use (massive/polygon)")
    args = parser.parse_args()

    _load_local_env()

    success = validate_model(args.model)
    if success and args.offline_eval:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
        if args.intraday:
            # Intraday evaluation path: load 1m bars from Massive/Polygon for last N days.
            from historical_training_system import HistoricalDataCollector, HistoricalTradingEnv
            ActionMasker = None
            try:
                from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
            except Exception:
                ActionMasker = None

            collector = HistoricalDataCollector(cache_dir="data/historical")
            end_dt = pd.Timestamp.now()
            start_dt = end_dt - pd.Timedelta(days=int(args.intraday_days))
            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")
            model = _load_model(args.model)

            vix = pd.Series([20.0], index=pd.date_range("2000-01-01", periods=1, freq="1D"))
            all_trades = []
            for sym in symbols:
                print(f"\n📊 Intraday eval load: {sym} (1m) {start_date}→{end_date}")
                try:
                    df = collector.get_historical_data_massive(sym, start_date, end_date, interval="1m", use_cache=True)
                except Exception as e:
                    print(f"   ❌ Failed to load {sym} intraday: {e}")
                    continue
                if df is None or len(df) < 25:
                    print(f"   ⚠️ Not enough bars for {sym}: {0 if df is None else len(df)}")
                    continue
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.between_time("09:30", "16:00")
                req = ["open", "high", "low", "close", "volume"]
                df = df[[c for c in req if c in df.columns]].copy()

                env = HistoricalTradingEnv(
                    data=df,
                    vix_data=vix,
                    symbol=sym,
                    window_size=20,
                    initial_capital=100000.0,
                    use_greeks=True,
                    use_features=False,
                    human_momentum_mode=True,
                )
                if ActionMasker is not None:
                    env = ActionMasker(env, lambda e: e.action_masks())

                obs, _ = env.reset()
                done = False
                truncated = False
                steps = 0
                while not (done or truncated):
                    action, _ = model.predict(obs, deterministic=not args.stochastic)
                    obs, reward, done, truncated, info = env.step(int(action))
                    steps += 1
                    if steps > len(df) + 10:
                        break
                trade_hist = getattr(getattr(env, "unwrapped", env), "trade_history", [])
                for t in trade_hist:
                    t2 = dict(t)
                    t2["symbol"] = sym
                    all_trades.append(t2)
                print(f"   ✅ {sym}: eval steps={steps:,} trades={len(trade_hist)}")

            if not all_trades:
                print("\n❌ No trades recorded in intraday evaluation window.")
            else:
                trades_df = pd.DataFrame(all_trades)
                worst = float(trades_df["pnl_pct"].min()) if "pnl_pct" in trades_df.columns else None
                avg = float(trades_df["pnl_pct"].mean()) if "pnl_pct" in trades_df.columns else None
                print("\n" + "-" * 80)
                print(f"Trades total: {len(trades_df)}")
                if avg is not None:
                    print(f"Avg trade pnl_pct: {avg:.2%}")
                if worst is not None:
                    print(f"Worst trade pnl_pct: {worst:.2%} (target >= -15.0%)")
                print("-" * 80)
        else:
            offline_evaluate(
                model_path=args.model,
                symbols=symbols,
                enriched_dir=args.enriched_dir,
                days=args.days,
                deterministic=not args.stochastic,
            )
    sys.exit(0 if success else 1)

