#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUSTOM LSTM POLICY FOR STABLE-BASELINES3

Implements LSTM backbone for PPO when RecurrentPPO is not available.
This provides temporal intelligence and state memory for the RL agent.

Author: Mike Agent Institutional Upgrade
Date: December 9, 2025
"""

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for temporal pattern recognition
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(observation_space, features_dim)
        
        # Determine input size from observation space
        if isinstance(observation_space, spaces.Box):
            if len(observation_space.shape) == 2:
                # (timesteps, features) shape
                self.input_size = observation_space.shape[1]
                self.timesteps = observation_space.shape[0]
            else:
                # Flatten if needed
                self.input_size = int(np.prod(observation_space.shape))
                self.timesteps = 1
        else:
            self.input_size = observation_space.n
            self.timesteps = 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=False  # (seq_len, batch, features) format for LSTM
        )
        
        # Post-LSTM processing
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        self.activation = nn.ReLU()
        
        # Initialize LSTM hidden state
        self.lstm_hidden = None
        self.last_batch_size = None
        
    def reset_hidden_state(self):
        """Reset LSTM hidden state (called by SB3 automatically at episode start)"""
        self.lstm_hidden = None
        self.last_batch_size = None
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM
        
        Args:
            observations: (batch, timesteps, features) or (batch, features)
        
        Returns:
            Features: (batch, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Handle different input shapes
        if len(observations.shape) == 3:
            # (batch, timesteps, features) - perfect for LSTM
            seq_len, features = observations.shape[1], observations.shape[2]
            # Transpose to (seq_len, batch, features) for LSTM
            obs_reshaped = observations.transpose(0, 1)
        elif len(observations.shape) == 2:
            # (batch, flattened_features) - need to reshape
            if hasattr(self, 'timesteps') and self.timesteps > 1:
                # Reshape assuming timesteps
                seq_len = self.timesteps
                features = observations.shape[1] // seq_len
                obs_reshaped = observations.view(batch_size, seq_len, features).transpose(0, 1)
            else:
                # Single timestep - add sequence dimension
                seq_len = 1
                obs_reshaped = observations.unsqueeze(0)  # (1, batch, features)
        else:
            # Unexpected shape
            obs_reshaped = observations.view(batch_size, -1).unsqueeze(0)
        
        # CRITICAL FIX 1: Reset hidden state if batch size changed
        # This prevents the RuntimeError: Expected hidden[0] size (2, 128, 256), got [2, 1, 256]
        if self.lstm_hidden is not None:
            # Check if batch size matches
            # hidden state shape: (num_layers, batch_size, hidden_size) for both h and c
            if isinstance(self.lstm_hidden, tuple):
                expected_batch_size = self.lstm_hidden[0].shape[1]
            else:
                expected_batch_size = self.lstm_hidden.shape[1]
            
            if expected_batch_size != batch_size:
                # Batch size changed - reset hidden state
                self.lstm_hidden = None
        
        # CRITICAL FIX 2: Detach hidden state from computation graph before reuse
        # This prevents: RuntimeError: Trying to backward through the graph a second time
        # The hidden state should NOT be part of the gradient computation graph
        # We only want gradients for the current forward pass, not previous ones
        if self.lstm_hidden is not None:
            if isinstance(self.lstm_hidden, tuple):
                # Detach both h and c tensors
                self.lstm_hidden = (
                    self.lstm_hidden[0].detach(),
                    self.lstm_hidden[1].detach()
                )
            else:
                self.lstm_hidden = self.lstm_hidden.detach()
        
        # LSTM forward pass
        # Pass None if hidden state is None (PyTorch will initialize)
        lstm_out, self.lstm_hidden = self.lstm(obs_reshaped, self.lstm_hidden)
        
        # CRITICAL FIX 3: Detach hidden state AFTER forward pass
        # This ensures the hidden state is not part of the computation graph for next iteration
        if isinstance(self.lstm_hidden, tuple):
            self.lstm_hidden = (
                self.lstm_hidden[0].detach(),
                self.lstm_hidden[1].detach()
            )
        else:
            self.lstm_hidden = self.lstm_hidden.detach()
        
        # Use last timestep output
        # lstm_out shape: (seq_len, batch, hidden_size)
        last_output = lstm_out[-1]  # (batch, hidden_size)
        
        # Post-LSTM processing
        features = self.linear(last_output)
        features = self.activation(features)
        
        return features


class LSTMPolicy(ActorCriticPolicy):
    """
    Custom PPO policy with LSTM backbone for temporal intelligence
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        # Extract LSTM-specific kwargs
        lstm_hidden_size = kwargs.pop('lstm_hidden_size', 256)
        lstm_num_layers = kwargs.pop('lstm_num_layers', 2)
        
        # Use LSTM feature extractor
        features_extractor_class = kwargs.pop(
            'features_extractor_class',
            lambda obs_space: LSTMFeatureExtractor(
                obs_space,
                features_dim=256,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers
            )
        )
        
        # Default network architecture for post-LSTM layers
        if net_arch is None:
            net_arch = [128, 64]
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            *args,
            **kwargs
        )
    
    def reset_hidden_state(self, batch_size: int = 1):
        """Reset LSTM hidden state (called by SB3 automatically)"""
        if hasattr(self.features_extractor, 'reset_hidden_state'):
            self.features_extractor.reset_hidden_state()
        elif hasattr(self.features_extractor, 'lstm_hidden'):
            self.features_extractor.lstm_hidden = None
            if hasattr(self.features_extractor, 'last_batch_size'):
                self.features_extractor.last_batch_size = None


# Registration function for easy use
def create_lstm_policy(observation_space: gym.Space, action_space: gym.Space):
    """
    Create LSTM policy factory for PPO
    
    Usage:
        policy_kwargs = {
            'features_extractor_class': lambda obs_space: LSTMFeatureExtractor(obs_space, features_dim=256)
        }
        model = PPO("LSTMPolicy", env, policy_kwargs=policy_kwargs, ...)
    """
    def policy_fn(**kwargs):
        return LSTMPolicy(observation_space, action_space, **kwargs)
    return policy_fn

