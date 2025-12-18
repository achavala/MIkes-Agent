#!/usr/bin/env python3
"""
🚀 HISTORICAL MODEL TRAINING SCRIPT

Trains RL model on historical data (2002-present) with regime-aware sampling
Ensures model learns from good, bad, and worst market days

Usage:
    python train_historical_model.py --symbols SPY,QQQ --start-date 2002-01-01 --timesteps 1000000

Author: Mike Agent Institutional Upgrade
Date: December 6, 2025
"""

import argparse
import os
import sys
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from collections import Counter

# Minimal .env loader (no external deps)
def _load_local_env() -> dict:
    """
    Load environment variables from a local .env file if present.
    - Looks in CWD and project root (this file's directory).
    - Does NOT override already-set non-empty env vars.
    """
    candidates = [
        Path(os.getcwd()) / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    loaded_keys: list[str] = []
    checked = [(str(p), p.exists()) for p in candidates]
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
                    loaded_keys.append(k)
        except Exception:
            continue
    return {"checked": checked, "loaded_keys": sorted(set(loaded_keys))}

# -------------------- FAST PRE-FLIGHT (NO HEAVY IMPORTS) --------------------
# This prevents noisy Gym/TA warnings during `--dry-run` by exiting early
# before importing SB3/Gym or the training environment.
if __name__ == "__main__" and "--dry-run" in sys.argv:
    env_info = _load_local_env()

    # Minimal argv scan for data-source (defaults to enriched)
    ds = "enriched"
    if "--data-source" in sys.argv:
        try:
            ds = sys.argv[sys.argv.index("--data-source") + 1].strip()
        except Exception:
            ds = "enriched"

    key_ok = bool(os.getenv("MASSIVE_API_KEY", "").strip())
    print(f"[MASSIVE] Key detected: {key_ok}")
    if not key_ok:
        print(f"[ENV] cwd: {os.getcwd()}")
        print(f"[ENV] .env checked: {env_info.get('checked')}")
        print(f"[ENV] loaded keys: {env_info.get('loaded_keys')}")

    if ds in ("massive", "polygon", "intraday") and not key_ok:
        print("❌ MASSIVE_API_KEY missing. Add .env or export it before training.")
        raise SystemExit(2)

    print("✅ Dry-run OK")
    raise SystemExit(0)

# RL imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

# LSTM Policy imports (custom implementation)
try:
    from custom_lstm_policy import LSTMFeatureExtractor, LSTMPolicy
    CUSTOM_LSTM_AVAILABLE = True
except ImportError:
    CUSTOM_LSTM_AVAILABLE = False
    print("⚠️  Custom LSTM policy not available")

# Import training system
from historical_training_system import (
    HistoricalDataCollector,
    HistoricalTradingEnv,
    create_regime_aware_training_data
)

# Suppress warnings
os.environ['GYM_NO_DEPRECATION_WARN'] = '1'


class RegimeAwareTrainingCallback:
    """Ensures training samples from all market regimes."""

    def __init__(self, vix_data: pd.Series):
        self.vix_data = vix_data
        self.regime_counts = {'calm': 0, 'normal': 0, 'storm': 0, 'crash': 0}

    def classify_regime(self, vix: float) -> str:
        if vix < 18:
            return 'calm'
        if vix < 25:
            return 'normal'
        if vix < 35:
            return 'storm'
        return 'crash'

    def get_regime_for_date(self, date) -> str:
        try:
            if isinstance(date, str):
                date = pd.to_datetime(date).date()
            elif hasattr(date, 'date'):
                date = date.date()

            if isinstance(self.vix_data.index, pd.DatetimeIndex):
                vix_values = self.vix_data[self.vix_data.index.date == date]
                if len(vix_values) > 0:
                    vix = float(vix_values.iloc[-1])
                    regime = self.classify_regime(vix)
                    self.regime_counts[regime] += 1
                    return regime
            return 'normal'
        except Exception:
            return 'normal'


class MomentumDiagnosticsCallback(BaseCallback):
    """Prints action/setup diagnostics during training (human-momentum mode)."""

    def __init__(self, print_every_steps: int = 5000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.print_every_steps = int(print_every_steps)
        self.flat_action_counts = Counter()
        self.strong_setup_count = 0
        self.strong_setup_buy_count = 0
        self.strong_setup_hold_count = 0
        self.good_buy_bonus_count = 0
        self.missed_opportunity_count = 0
        self.bad_chase_penalty_count = 0

    def _on_step(self) -> bool:
        try:
            infos = self.locals.get("infos", [])
            actions = self.locals.get("actions", None)
            if actions is None:
                return True
            acts = list(actions) if hasattr(actions, "__len__") else [actions]

            for i, info in enumerate(infos):
                if not isinstance(info, dict):
                    continue
                a = int(acts[i]) if i < len(acts) else None
                is_flat_pre = bool(info.get("is_flat_pre", False))

                if is_flat_pre and a is not None:
                    self.flat_action_counts[a] += 1

                if bool(info.get("strong_setup", False)) and is_flat_pre and a is not None:
                    self.strong_setup_count += 1
                    if a in (1, 2):
                        self.strong_setup_buy_count += 1
                    elif a == 0:
                        self.strong_setup_hold_count += 1

                if bool(info.get("good_buy_bonus", False)):
                    self.good_buy_bonus_count += 1
                if bool(info.get("missed_opportunity", False)):
                    self.missed_opportunity_count += 1
                if bool(info.get("bad_chase_penalty", False)):
                    self.bad_chase_penalty_count += 1

            if self.num_timesteps > 0 and (self.num_timesteps % self.print_every_steps == 0):
                total_flat = sum(self.flat_action_counts.values())

                def pct(n: int) -> float:
                    return 0.0 if total_flat == 0 else (100.0 * n / total_flat)

                print("\n" + "-" * 70)
                print(f"📊 MomentumDiagnostics @ step={self.num_timesteps:,}")
                print("FLAT action distribution (count / %):")
                for a in sorted(self.flat_action_counts.keys()):
                    print(f"  action {a}: {self.flat_action_counts[a]} ({pct(self.flat_action_counts[a]):.1f}%)")

                if self.strong_setup_count > 0:
                    buy_rate = 100.0 * self.strong_setup_buy_count / self.strong_setup_count
                    hold_rate = 100.0 * self.strong_setup_hold_count / self.strong_setup_count
                    print(f"Strong-setup states: {self.strong_setup_count} | BUY rate={buy_rate:.1f}% | HOLD rate={hold_rate:.1f}%")
                else:
                    print("Strong-setup states: 0")

                print(
                    "Triggers: "
                    f"good_buy_bonus={self.good_buy_bonus_count} | "
                    f"missed_opportunity={self.missed_opportunity_count} | "
                    f"bad_chase_penalty={self.bad_chase_penalty_count}"
                )
                print("-" * 70 + "\n")
        except Exception:
            return True
        return True


def create_env_from_data(
    data: pd.DataFrame,
    vix_data: pd.Series,
    symbol: str,
    use_greeks: bool = True,
    use_features: bool = False,
    human_momentum: bool = False
) -> HistoricalTradingEnv:
    """Create environment from historical data"""
    env = HistoricalTradingEnv(
        data=data,
        vix_data=vix_data,
        symbol=symbol,
        window_size=20,
        initial_capital=100000.0,
        use_greeks=use_greeks,
        use_features=use_features,
        human_momentum_mode=human_momentum
    )
    return env


def split_data_by_regime(
    data: pd.DataFrame,
    vix_data: pd.Series
) -> Dict[str, pd.DataFrame]:
    """
    Split data by market regime to ensure balanced training
    """
    regimes = {
        'calm': [],
        'normal': [],
        'storm': [],
        'crash': []
    }
    
    if not isinstance(data.index, pd.DatetimeIndex):
        return {'all': data}
    
    # Group by date
    data['date'] = data.index.date
    
    for date, day_data in data.groupby('date'):
        try:
            # Get VIX for this date
            vix_values = vix_data[vix_data.index.date == date]
            if len(vix_values) > 0:
                vix = float(vix_values.iloc[-1])
                
                if vix < 18:
                    regime = 'calm'
                elif vix < 25:
                    regime = 'normal'
                elif vix < 35:
                    regime = 'storm'
                else:
                    regime = 'crash'
                
                regimes[regime].append(day_data)
        except:
            regimes['normal'].append(day_data)
    
    # Combine days for each regime
    result = {}
    for regime, day_list in regimes.items():
        if len(day_list) > 0:
            combined = pd.concat(day_list, axis=0)
            combined = combined.drop('date', axis=1) if 'date' in combined.columns else combined
            result[regime] = combined
    
    return result


def train_on_historical_data(
    symbols: List[str],
    start_date: str = "2002-01-01",
    end_date: Optional[str] = None,
    total_timesteps: int = 1000000,
    model_name: str = "mike_historical_model",
    use_greeks: bool = True,
    use_features: bool = False,
    regime_balanced: bool = True,
    human_momentum: bool = False,
    data_source: str = "enriched",
    intraday_days: int = 60,
    learning_rate: Optional[float] = None,
    ent_coef: Optional[float] = None,
    gamma: Optional[float] = None,
    n_steps: Optional[int] = None,
    load_model_path: Optional[str] = None
):
    """
    Train RL model on historical data with regime-aware sampling
    """
    print("=" * 70)
    print("🚀 HISTORICAL MODEL TRAINING")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Date Range: {start_date} to {end_date or 'today'}")
    print(f"Total Timesteps: {total_timesteps:,}")
    if load_model_path:
        print(f"Resuming from model: {load_model_path}")
    print(f"Use Greeks: {use_greeks}")
    print(f"Use Features: {use_features}")
    print(f"Human Momentum Mode: {human_momentum}")
    print(f"Regime Balanced: {regime_balanced}")
    print(f"Data Source: {data_source}")
    print("=" * 70)
    print()
    
    # Initialize data collector
    collector = HistoricalDataCollector(cache_dir="data/historical")
    
    all_data_dict = {}
    vix_data = collector.get_vix_data(start_date, end_date, use_cache=True)

    # If human_momentum is enabled, default to intraday data unless explicitly overridden.
    resolved_source = data_source
    if human_momentum and data_source == "enriched":
        resolved_source = "massive"

    if resolved_source in ("massive", "polygon", "intraday"):
        print("📥 LOADING INTRADAY (1m) HISTORICAL DATA...")
        print("   Source: Massive/Polygon (cached)")
        print(f"   Window: last {intraday_days} days")
        print()

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=int(intraday_days))
        intraday_start = start_dt.strftime("%Y-%m-%d")
        intraday_end = end_dt.strftime("%Y-%m-%d")

        for symbol in symbols:
            print(f"\n📊 Loading {symbol} intraday data (1m)...")
            print(f"   🔑 Data Source Priority: Alpaca → Massive → yfinance")
            print(f"   💰 Using PAID services first (Alpaca/Massive)")
            try:
                df = collector.get_historical_data_massive(
                    symbol=symbol,
                    start_date=intraday_start,
                    end_date=intraday_end,
                    interval="1m",
                    use_cache=True,
                )
                if df is None or len(df) == 0:
                    print(f"   ❌ No intraday data returned for {symbol} from any source")
                    continue

                required_cols = ['open', 'high', 'low', 'close', 'volume']
                df = df[[c for c in required_cols if c in df.columns]].copy()
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.between_time('09:30', '16:00')

                print(f"   ✅ {symbol}: {len(df):,} bars loaded")
                print(f"      Date range: {df.index.min()} to {df.index.max()}")
                all_data_dict[symbol] = df
            except Exception as e:
                # Production-grade: Polygon/Massive is mandatory for intraday training.
                # Do not silently fall back for SPX; fail fast so environment issues are fixed correctly.
                print(f"   ❌ Error loading intraday data for {symbol}: {e}")
                continue
    else:
        # Load enriched data files directly (offline operation)
        print("📥 LOADING ENRICHED HISTORICAL DATA...")
        print("   Using pre-collected enriched data files (works offline)")
        print()

        enriched_dir = Path("data/historical/enriched")

        for symbol in symbols:
            print(f"\n📊 Loading {symbol} enriched data...")

            enriched_files = list(enriched_dir.glob(f"{symbol}_enriched_*.pkl"))
            if not enriched_files:
                print(f"   ⚠️  No enriched data file found for {symbol}")
                print(f"   💡 Looking for: {enriched_dir}/{symbol}_enriched_*.pkl")
                print(f"   💡 To create enriched data, run:")
                print(f"      python collect_quant_features.py --symbols {symbol}")
                continue

            enriched_file = sorted(enriched_files)[-1]
            print(f"   📂 Loading from: {enriched_file.name}")

            try:
                with open(enriched_file, 'rb') as f:
                    data = pickle.load(f)

                if not isinstance(data, pd.DataFrame):
                    print(f"   ⚠️  Invalid data format in {enriched_file.name}")
                    continue

                if isinstance(data.index, pd.DatetimeIndex) and len(data) > 0:
                    sample_time = data.index[0].time()
                    is_daily_data = sample_time == pd.Timestamp('00:00:00').time()

                    if not is_daily_data:
                        data = data.between_time('09:30', '16:00')
                        print(f"      Filtered to trading hours (intraday data)")
                    else:
                        print(f"      Using all daily bars (no time filtering needed)")

                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                if missing_cols:
                    print(f"   ⚠️  Missing required columns: {missing_cols}")
                    print(f"   Available columns: {list(data.columns)[:10]}...")
                    continue

                training_data = data[required_cols].copy()

                print(f"   ✅ {symbol}: {len(training_data):,} bars loaded")
                print(f"      Date range: {training_data.index.min()} to {training_data.index.max()}")
                all_data_dict[symbol] = training_data

            except Exception as e:
                print(f"   ❌ Error loading enriched data: {e}")
                continue
    
    if len(all_data_dict) == 0:
        print("❌ No data collected! Exiting.")
        return
    
    print("\n" + "=" * 70)
    print("📚 PREPARING TRAINING ENVIRONMENTS...")
    print("=" * 70)
    
    # Create environments for each symbol/regime
    envs = []
    
    if regime_balanced:
        # Split by regime for balanced training
        for symbol, data in all_data_dict.items():
            regime_data = split_data_by_regime(data, vix_data)
            
            for regime, regime_df in regime_data.items():
                if len(regime_df) < 100:  # Skip if too little data
                    continue
                
                print(f"  Creating env: {symbol} ({regime}) - {len(regime_df):,} bars")
                env = create_env_from_data(
                    data=regime_df,
                    vix_data=vix_data,
                    symbol=symbol,
                    use_greeks=use_greeks,
                    use_features=use_features,
                    human_momentum=human_momentum
                )
                envs.append(env)
    else:
        # Simple: one env per symbol
        for symbol, data in all_data_dict.items():
            print(f"  Creating env: {symbol} - {len(data):,} bars")
            env = create_env_from_data(
                data=data,
                vix_data=vix_data,
                symbol=symbol,
                use_greeks=use_greeks,
                use_features=use_features,
                human_momentum=human_momentum
            )
            envs.append(env)
    
    if len(envs) == 0:
        print("❌ No environments created! Exiting.")
        return
    
    print(f"\n✅ Created {len(envs)} training environments")
    
    # Create vectorized environment (use first env as template)
    print("\n" + "=" * 70)
    print("🎓 STARTING TRAINING...")
    print("=" * 70)
    
    # For now, train on first environment (can be enhanced for multi-env)
    training_env = envs[0]
    
    # Optional: true action masking (when using sb3-contrib MaskablePPO).
    # This prevents the policy from wasting probability mass on invalid actions (TRIM/EXIT while flat).
    USE_MASKABLE = False
    ActionMasker = None
    MaskablePPO = None
    if human_momentum:
        try:
            from sb3_contrib import MaskablePPO  # type: ignore
            from sb3_contrib.common.wrappers import ActionMasker  # type: ignore
            USE_MASKABLE = True
            print("✅ sb3-contrib available - enabling true action masking (MaskablePPO)")
            training_env = ActionMasker(training_env, lambda env: env.action_masks())
        except Exception:
            print("⚠️  sb3-contrib not installed - using penalty-based masking only")

    # Wrap with Monitor for logging
    log_dir = "logs/training"
    os.makedirs(log_dir, exist_ok=True)
    training_env = Monitor(training_env, log_dir)
    
    # Create vectorized env
    vec_env = DummyVecEnv([lambda: training_env])
    
    # ==================== LSTM POLICY UPGRADE ====================
    # Upgrade from MLP → LSTM for temporal intelligence and state memory
    # This enables: pattern recognition, regime transitions, trend detection
    
    # PHASE 1 FIX: Allow LSTM even with MaskablePPO by using RecurrentPPO (which supports both)
    # RecurrentPPO from sb3-contrib provides LSTM + action masking support
    USE_LSTM = False
    LSTM_TYPE = None
    
    # Try RecurrentPPO first (supports LSTM + action masking)
    if USE_MASKABLE:
        try:
            from sb3_contrib import RecurrentPPO  # type: ignore
            USE_LSTM = True
            LSTM_TYPE = "RecurrentPPO"
            print("✅ RecurrentPPO available - Using LSTM Policy with action masking (temporal intelligence enabled)")
        except ImportError:
            print("ℹ️  MaskablePPO available but RecurrentPPO not found - using MLP with action masking")
            print("   💡 To enable LSTM: pip install --upgrade sb3-contrib")
    
    # If RecurrentPPO not available, try other LSTM options (only if not using MaskablePPO)
    if not USE_LSTM and not USE_MASKABLE:
        # Try multiple LSTM options (in priority order)
        # Option 1: RecurrentPPO (sb3-contrib)
        try:
            from sb3_contrib import RecurrentPPO  # type: ignore
            USE_LSTM = True
            LSTM_TYPE = "RecurrentPPO"
            print("✅ RecurrentPPO available - Using LSTM Policy (temporal intelligence enabled)")
        except ImportError:
            pass
    
        # Option 2: Custom LSTM Policy (if RecurrentPPO not available)
        if not USE_LSTM and CUSTOM_LSTM_AVAILABLE:
            USE_LSTM = True
            LSTM_TYPE = "CustomLSTM"
            print("✅ Custom LSTM Policy available - Using LSTM Policy (temporal intelligence enabled)")

        # Option 3: MlpLstmPolicy (if available in some SB3 versions)
        if not USE_LSTM:
            try:
                from stable_baselines3.common.policies import MlpLstmPolicy
                USE_LSTM = True
                LSTM_TYPE = "MlpLstmPolicy"
                print("✅ MlpLstmPolicy available - Using LSTM Policy (temporal intelligence enabled)")
            except ImportError:
                pass

        if not USE_LSTM:
            print("⚠️  No LSTM policy available, falling back to MLP")
            print("   💡 Options to enable LSTM:")
            print("      1. pip install --upgrade stable-baselines3[extra]")
            print("      2. Use custom_lstm_policy.py (already implemented)")

    # Resolve hyperparameters (allow CLI overrides)
    lr = learning_rate if learning_rate is not None else (3e-5 if USE_LSTM else 3e-4)
    # Anti-collapse: Increase default entropy for human_momentum mode to encourage exploration
    # Previous: 0.02 was too low, causing collapse into HOLD
    # Tune1: 0.06 showed improvement but slow recollapse at 10k → increased to 0.08
    ent = ent_coef if ent_coef is not None else (0.08 if human_momentum else 0.01)
    gam = gamma if gamma is not None else (0.92 if human_momentum else 0.99)
    steps = n_steps if n_steps is not None else (512 if USE_LSTM else 2048)
    
    if USE_LSTM and LSTM_TYPE == "RecurrentPPO":
        # Use RecurrentPPO (official SB3 recurrent policy with LSTM)
        policy_kwargs = {
            "lstm_hidden_size": 256,      # LSTM hidden units (memory capacity)
            "net_arch": [128, 64]         # Post-LSTM layers
        }
        
        # RecurrentPPO-optimized hyperparameters
        model = RecurrentPPO(
            "MlpLstmPolicy",              # ✅ LSTM Policy (temporal memory)
            vec_env,
            verbose=1,
            learning_rate=lr,
            n_steps=steps,
            batch_size=128,               # ✅ MUST be smaller for LSTM (was 64)
            n_epochs=10,
            gamma=gam,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="logs/tensorboard",
            policy_kwargs=policy_kwargs,
            device="cpu"
        )
        print("   ✅ RecurrentPPO Configuration:")
        print(f"      - Hidden Size: {policy_kwargs['lstm_hidden_size']}")
        print(f"      - N Steps: 512 (optimized for LSTM)")
        print(f"      - Batch Size: 128 (optimized for LSTM)")
    elif USE_LSTM and LSTM_TYPE == "CustomLSTM":
        # Use Custom LSTM Policy
        # Get observation space from environment
        obs_space = vec_env.observation_space
        
        policy_kwargs = {
            "features_extractor_class": lambda obs_space: LSTMFeatureExtractor(
                obs_space,
                features_dim=256,
                lstm_hidden_size=256,
                lstm_num_layers=2
            ),
            "net_arch": [128, 64],        # Post-LSTM layers
            "lstm_hidden_size": 256,
            "lstm_num_layers": 2
        }
        
        # LSTM-optimized hyperparameters
        model = PPO(
            LSTMPolicy,                   # ✅ Custom LSTM Policy (temporal memory)
            vec_env,
            verbose=1,
            learning_rate=lr,
            n_steps=steps,
            batch_size=128,               # ✅ MUST be smaller for LSTM (was 64)
            n_epochs=10,
            gamma=gam,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="logs/tensorboard",
            policy_kwargs=policy_kwargs,
            device="cpu"
        )
        print("   ✅ Custom LSTM Configuration:")
        print(f"      - Hidden Size: 256")
        print(f"      - LSTM Layers: 2")
        print(f"      - N Steps: 512 (optimized for LSTM)")
        print(f"      - Batch Size: 128 (optimized for LSTM)")
    elif USE_LSTM and LSTM_TYPE == "MlpLstmPolicy":
        # Use MlpLstmPolicy (if available)
        policy_kwargs = {
            "lstm_hidden_size": 256,      # LSTM hidden units (memory capacity)
            "enable_critic_lstm": True,   # Critic also uses LSTM
            "enable_actor_lstm": True,    # Actor uses LSTM for temporal patterns
            "net_arch": [128, 64]         # Post-LSTM layers
        }
        
        # LSTM-optimized hyperparameters
        model = PPO(
            "MlpLstmPolicy",              # ✅ LSTM Policy (temporal memory)
            vec_env,
            verbose=1,
            learning_rate=lr,
            n_steps=steps,
            batch_size=128,               # ✅ MUST be smaller for LSTM (was 64)
            n_epochs=10,
            gamma=gam,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="logs/tensorboard",
            policy_kwargs=policy_kwargs,
            device="cpu"
        )
        print("   ✅ LSTM Configuration:")
        print(f"      - Hidden Size: {policy_kwargs['lstm_hidden_size']}")
        print(f"      - Actor LSTM: {policy_kwargs['enable_actor_lstm']}")
        print(f"      - Critic LSTM: {policy_kwargs['enable_critic_lstm']}")
        print(f"      - N Steps: 512 (optimized for LSTM)")
        print(f"      - Batch Size: 128 (optimized for LSTM)")
    else:
        # Fallback to MLP (original)
        policy_kwargs = {}
        if use_features:
            policy_kwargs['net_arch'] = [256, 256, 128]  # Larger network for more features
        
        Algo = MaskablePPO if (USE_MASKABLE and MaskablePPO is not None) else PPO
        model = Algo(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=lr,
            n_steps=steps,
            batch_size=64,
            n_epochs=10,
            gamma=gam,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=ent,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="logs/tensorboard",
            policy_kwargs=policy_kwargs,
            device="cpu"
        )
        print("   ⚠️  Using MLP Policy (no temporal memory)")
        print("   💡 To enable LSTM: pip install --upgrade stable-baselines3[extra]")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='models/checkpoints',
        name_prefix=model_name
    )

    diagnostics_callback = MomentumDiagnosticsCallback(print_every_steps=5000) if human_momentum else None
    
    # Start training
    print(f"\n🏋️ Training for {total_timesteps:,} timesteps...")
    print("   (This will take a while - progress shown below)\n")
    
    start_time = time.time()
    
    try:
        use_progress_bar = True
        try:
            import tqdm  # noqa: F401
            import rich  # noqa: F401
        except Exception:
            use_progress_bar = False
        model.learn(
            total_timesteps=total_timesteps,
            callback=[cb for cb in [checkpoint_callback, diagnostics_callback] if cb is not None],
            progress_bar=use_progress_bar
        )
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    
    training_time = time.time() - start_time
    
    # Save final model
    model_path = f"models/{model_name}.zip"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved: {model_path}")
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Total timesteps: {total_timesteps:,}")
    print("=" * 70)


def main():
    # Load .env early so MASSIVE_API_KEY is visible to the process
    env_info = _load_local_env()

    parser = argparse.ArgumentParser(description='Train RL model on historical data')
    parser.add_argument('--symbols', type=str, default='SPY,QQQ', help='Comma-separated symbols')
    parser.add_argument('--start-date', type=str, default='2002-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD), default=today')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--model-name', type=str, default='mike_historical_model', help='Model name')
    parser.add_argument('--use-greeks', action='store_true', default=True, help='Use Greeks in observations')
    parser.add_argument('--use-features', action='store_true', default=False, help='Use institutional features')
    parser.add_argument('--regime-balanced', action='store_true', default=True, help='Balance training across regimes')
    parser.add_argument('--human-momentum', action='store_true', default=False, help='Enable human scalp/momentum training mode (new observation + reward)')
    parser.add_argument('--data-source', type=str, default='enriched',
                        choices=['enriched', 'massive', 'polygon', 'intraday', 'yfinance'],
                        help='Data source: enriched (offline daily), massive/polygon intraday 1m, or yfinance')
    parser.add_argument('--intraday-days', type=int, default=60,
                        help='For intraday sources: calendar days of 1m bars to load (cached)')
    parser.add_argument('--learning-rate', type=float, default=None, help='Override learning rate (e.g. 3e-5)')
    parser.add_argument('--ent-coef', type=float, default=None, help='Override entropy coefficient (e.g. 0.02)')
    parser.add_argument('--gamma', type=float, default=None, help='Override gamma discount (e.g. 0.92)')
    parser.add_argument('--n-steps', type=int, default=None, help='Override n_steps (e.g. 512)')
    parser.add_argument('--load-model', type=str, default=None, help='Path to existing model to resume training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='Validate environment + API key detection, then exit (no training).')
    
    args = parser.parse_args()

    # Dry-run: validate key visibility without leaking it
    if args.dry_run:
        key_ok = bool(os.getenv("MASSIVE_API_KEY", "").strip())
        print(f"[MASSIVE] Key detected: {key_ok}")
        if not key_ok:
            print(f"[ENV] cwd: {os.getcwd()}")
            print(f"[ENV] .env checked: {env_info.get('checked')}")
            print(f"[ENV] loaded keys: {env_info.get('loaded_keys')}")
        # If user requested massive/polygon/intraday sources, this must be True
        if args.data_source in ("massive", "polygon", "intraday"):
            if not key_ok:
                print("❌ MASSIVE_API_KEY missing. Add .env or export it before training.")
                raise SystemExit(2)
        print("✅ Dry-run OK")
        return
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    train_on_historical_data(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        total_timesteps=args.timesteps,
        model_name=args.model_name,
        use_greeks=args.use_greeks,
        use_features=args.use_features,
        regime_balanced=args.regime_balanced,
        human_momentum=args.human_momentum,
        data_source=args.data_source,
        intraday_days=args.intraday_days,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        n_steps=args.n_steps,
        load_model_path=args.load_model
    )


if __name__ == "__main__":
    main()

