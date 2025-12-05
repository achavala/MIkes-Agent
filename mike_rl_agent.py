import yfinance as yf
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import os

class MikeTradingEnv(gym.Env):
    def __init__(self, data, window_size=20):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.window_size = window_size
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 5), dtype=np.float32)
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        start = self.current_step
        end = start + self.window_size
        window = self.data.iloc[start:end]
        if len(window) < self.window_size:
            pad = pd.DataFrame(np.zeros((self.window_size - len(window), 5)), columns=self.data.columns)
            window = pd.concat([window, pad]).iloc[:self.window_size]
        return window[['open','high','low','close','volume']].values.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        done = self.current_step + self.window_size >= len(self.data)
        reward = action[0] * 0.001
        return self._get_obs(), reward, done, False, {}

def train():
    print("Downloading SPY data...")
    data = yf.download("SPY", period="60d", interval="1h", progress=False)
    
    # THE ONLY LINE THAT WORKS 100% IN 2025 — NO MATTER WHAT YFINANCE DOES
    data = data.xs('SPY', axis=1, level=1) if isinstance(data.columns, pd.MultiIndex) else data
    
    # Force lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Ensure we have the columns
    required = ['open','high','low','close','volume']
    for col in required:
        if col not in data.columns:
            print(f"Missing {col} — using previous close as fallback")
            data[col] = data.get(col, data['close'])
    
    data = data[required].copy()
    data = data.dropna().reset_index(drop=True)
    
    print(f"Training on {len(data)} clean bars...")
    
    env = DummyVecEnv([lambda: MikeTradingEnv(data)])
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=50000)
    
    model.save("mike_rl_model.zip")
    print("Training completed successfully!")
    print("Model saved as: mike_rl_model.zip")

if __name__ == "__main__":
    train()
