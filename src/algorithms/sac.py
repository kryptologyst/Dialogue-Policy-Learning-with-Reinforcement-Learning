"""SAC (Soft Actor-Critic) implementation for dialogue policy learning."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass
import gymnasium as gym

from .ppo import TrainingConfig, DialoguePolicyNetwork, DialogueValueNetwork, ReplayBuffer


class SoftQNetwork(nn.Module):
    """Soft Q-network for SAC."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """Initialize the Q-network."""
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = obs_dim + action_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.q_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = torch.cat([obs, action], dim=-1)
        features = self.network(x)
        q_value = self.q_head(features)
        return q_value.squeeze(-1)


class SACPolicyNetwork(nn.Module):
    """Policy network for SAC with continuous actions."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        """Initialize the SAC policy network."""
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Build network layers
        layers = []
        input_dim = obs_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            ])
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        features = self.network(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        
        # Reparameterization trick
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        
        # Compute log probability
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class SACTrainer:
    """Soft Actor-Critic trainer for dialogue policy learning."""
    
    def __init__(
        self,
        env: gym.Env,
        config: TrainingConfig,
        device: str = "auto",
    ):
        """Initialize SAC trainer."""
        self.env = env
        self.config = config
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Get dimensions
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_dim = sum(np.prod(space.shape) for space in obs_space.values())
        else:
            self.obs_dim = np.prod(obs_space.shape)
        
        self.action_dim = env.action_space.n
        
        # For discrete actions, we'll use a categorical policy
        self.discrete_actions = True
        
        # Initialize networks
        self.policy_net = DialoguePolicyNetwork(
            self.obs_dim, self.action_dim
        ).to(self.device)
        
        self.q_net1 = SoftQNetwork(
            self.obs_dim, self.action_dim
        ).to(self.device)
        
        self.q_net2 = SoftQNetwork(
            self.obs_dim, self.action_dim
        ).to(self.device)
        
        # Target networks
        self.target_q_net1 = SoftQNetwork(
            self.obs_dim, self.action_dim
        ).to(self.device)
        
        self.target_q_net2 = SoftQNetwork(
            self.obs_dim, self.action_dim
        ).to(self.device)
        
        # Copy parameters to target networks
        self.target_q_net1.load_state_dict(self.q_net1.state_dict())
        self.target_q_net2.load_state_dict(self.q_net2.state_dict())
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )
        self.q_optimizer = optim.Adam(
            list(self.q_net1.parameters()) + list(self.q_net2.parameters()),
            lr=config.learning_rate
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size, self.obs_dim, self.action_dim
        )
        
        # Entropy coefficient
        if config.auto_entropy:
            self.target_entropy = -self.action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
        else:
            self.alpha = config.alpha
        
        # Training state
        self.total_timesteps = 0
    
    def _flatten_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Flatten observation dictionary to tensor."""
        flattened = []
        for key in sorted(obs.keys()):
            flattened.append(obs[key].flatten())
        return torch.cat(flattened).to(self.device)
    
    def _one_hot_action(self, action: int) -> torch.Tensor:
        """Convert discrete action to one-hot encoding."""
        action_tensor = torch.zeros(self.action_dim, device=self.device)
        action_tensor[action] = 1.0
        return action_tensor
    
    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        """Select action from policy."""
        with torch.no_grad():
            logits = self.policy_net(obs.unsqueeze(0))
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
            return action.item()
    
    def update(self, batch_size: int) -> Dict[str, float]:
        """Update networks."""
        if self.replay_buffer.size < batch_size:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(batch_size)
        
        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)
        
        # Convert actions to one-hot
        actions_one_hot = torch.zeros(batch_size, self.action_dim, device=self.device)
        actions_one_hot.scatter_(1, actions.long().unsqueeze(1), 1)
        
        with torch.no_grad():
            # Compute target Q-values
            next_logits = self.policy_net(next_obs)
            next_dist = torch.distributions.Categorical(logits=next_logits)
            next_actions = next_dist.sample()
            next_actions_one_hot = torch.zeros(batch_size, self.action_dim, device=self.device)
            next_actions_one_hot.scatter_(1, next_actions.unsqueeze(1), 1)
            
            next_log_probs = next_dist.log_prob(next_actions)
            
            target_q1 = self.target_q_net1(next_obs, next_actions_one_hot)
            target_q2 = self.target_q_net2(next_obs, next_actions_one_hot)
            target_q = torch.min(target_q1, target_q2)
            
            alpha = self.log_alpha.exp() if self.config.auto_entropy else self.alpha
            target_q = target_q - alpha * next_log_probs
            
            target_q_values = rewards + self.config.gamma * (1 - dones) * target_q
        
        # Update Q-networks
        current_q1 = self.q_net1(obs, actions_one_hot)
        current_q2 = self.q_net2(obs, actions_one_hot)
        
        q1_loss = F.mse_loss(current_q1, target_q_values)
        q2_loss = F.mse_loss(current_q2, target_q_values)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        
        # Update policy network
        logits = self.policy_net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        new_actions = dist.sample()
        new_actions_one_hot = torch.zeros(batch_size, self.action_dim, device=self.device)
        new_actions_one_hot.scatter_(1, new_actions.unsqueeze(1), 1)
        
        log_probs = dist.log_prob(new_actions)
        
        q1_new = self.q_net1(obs, new_actions_one_hot)
        q2_new = self.q_net2(obs, new_actions_one_hot)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp() if self.config.auto_entropy else self.alpha
        policy_loss = (alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update alpha
        if self.config.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # Update target networks
        self._soft_update(self.target_q_net1, self.q_net1)
        self._soft_update(self.target_q_net2, self.q_net2)
        
        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha": alpha.item() if self.config.auto_entropy else self.alpha,
        }
    
    def _soft_update(self, target_net: nn.Module, source_net: nn.Module) -> None:
        """Soft update target network."""
        for target_param, param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    
    def train(self, total_timesteps: int, eval_freq: int = 10000) -> None:
        """Train the agent."""
        obs, _ = self.env.reset()
        obs_tensor = self._flatten_obs(obs)
        
        episode_reward = 0
        episode_length = 0
        
        while self.total_timesteps < total_timesteps:
            # Select action
            action = self.select_action(obs_tensor)
            
            # Take action
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.replay_buffer.add(
                obs_tensor.cpu().numpy(),
                np.array([action]),
                reward,
                self._flatten_obs(next_obs).cpu().numpy(),
                done,
            )
            
            # Update networks
            if self.total_timesteps > 1000:  # Start training after some experience
                losses = self.update(64)
                
                if self.total_timesteps % 1000 == 0 and losses:
                    print(f"Timesteps: {self.total_timesteps}")
                    print(f"Q1 Loss: {losses['q1_loss']:.4f}")
                    print(f"Q2 Loss: {losses['q2_loss']:.4f}")
                    print(f"Policy Loss: {losses['policy_loss']:.4f}")
                    print(f"Alpha: {losses['alpha']:.4f}")
                    print("-" * 50)
            
            # Update state
            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1
            
            if done:
                obs, _ = self.env.reset()
                obs_tensor = self._flatten_obs(obs)
                
                if self.total_timesteps % eval_freq == 0:
                    print(f"Episode Reward: {episode_reward:.2f}")
                    print(f"Episode Length: {episode_length}")
                
                episode_reward = 0
                episode_length = 0
            else:
                obs = next_obs
                obs_tensor = self._flatten_obs(obs)
