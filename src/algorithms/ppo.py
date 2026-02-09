"""Modern RL algorithms for dialogue policy learning."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from dataclasses import dataclass
import gymnasium as gym


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 10000
    target_entropy: Optional[float] = None
    tau: float = 0.005
    alpha: float = 0.2
    auto_entropy: bool = True


class DialoguePolicyNetwork(nn.Module):
    """Policy network for dialogue actions."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """Initialize the policy network.
        
        Args:
            obs_dim: Dimension of flattened observation
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
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
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.network(obs)
        logits = self.action_head(features)
        return logits
    
    def get_action_and_log_prob(
        self, 
        obs: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action and log probability."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action, log_prob


class DialogueValueNetwork(nn.Module):
    """Value network for dialogue state values."""
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        """Initialize the value network.
        
        Args:
            obs_dim: Dimension of flattened observation
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        
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
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        features = self.network(obs)
        value = self.value_head(features)
        return value.squeeze(-1)


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms."""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Initialize buffers
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to buffer."""
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample batch from buffer."""
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "obs": torch.FloatTensor(self.obs[idx]),
            "actions": torch.FloatTensor(self.actions[idx]),
            "rewards": torch.FloatTensor(self.rewards[idx]),
            "next_obs": torch.FloatTensor(self.next_obs[idx]),
            "dones": torch.FloatTensor(self.dones[idx]),
        }


class PPOTrainer:
    """Proximal Policy Optimization trainer for dialogue policy learning."""
    
    def __init__(
        self,
        env: gym.Env,
        config: TrainingConfig,
        device: str = "auto",
    ):
        """Initialize PPO trainer.
        
        Args:
            env: Training environment
            config: Training configuration
            device: Device to use for training
        """
        self.env = env
        self.config = config
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Get observation and action dimensions
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            self.obs_dim = sum(
                np.prod(space.shape) for space in obs_space.values()
            )
        else:
            self.obs_dim = np.prod(obs_space.shape)
        
        self.action_dim = env.action_space.n
        
        # Initialize networks
        self.policy_net = DialoguePolicyNetwork(
            self.obs_dim, self.action_dim
        ).to(self.device)
        
        self.value_net = DialogueValueNetwork(
            self.obs_dim
        ).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=config.learning_rate
        )
        
        # Training state
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _flatten_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Flatten observation dictionary to tensor."""
        flattened = []
        for key in sorted(obs.keys()):
            flattened.append(obs[key].flatten())
        return torch.cat(flattened).to(self.device)
    
    def collect_rollouts(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Collect rollout data."""
        obs_list = []
        action_list = []
        reward_list = []
        value_list = []
        log_prob_list = []
        done_list = []
        
        obs, _ = self.env.reset()
        obs_tensor = self._flatten_obs(obs)
        
        for _ in range(num_steps):
            # Get action from policy
            with torch.no_grad():
                action, log_prob = self.policy_net.get_action_and_log_prob(
                    obs_tensor.unsqueeze(0)
                )
                value = self.value_net(obs_tensor.unsqueeze(0))
            
            # Take action
            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            # Store data
            obs_list.append(obs_tensor.cpu())
            action_list.append(action.cpu())
            reward_list.append(reward)
            value_list.append(value.cpu())
            log_prob_list.append(log_prob.cpu())
            done_list.append(done)
            
            # Update observation
            if done:
                obs, _ = self.env.reset()
            else:
                obs = next_obs
            obs_tensor = self._flatten_obs(obs)
        
        return {
            "obs": torch.stack(obs_list),
            "actions": torch.stack(action_list),
            "rewards": torch.tensor(reward_list, dtype=torch.float32),
            "values": torch.stack(value_list).squeeze(),
            "log_probs": torch.stack(log_prob_list).squeeze(),
            "dones": torch.tensor(done_list, dtype=torch.float32),
        }
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_advantage = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_advantage
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, rollouts: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update policy and value networks."""
        # Move data to device
        for key in rollouts:
            rollouts[key] = rollouts[key].to(self.device)
        
        # Compute advantages and returns
        with torch.no_grad():
            next_obs = rollouts["obs"][-1]
            next_value = self.value_net(next_obs.unsqueeze(0)).squeeze()
        
        advantages, returns = self.compute_gae(
            rollouts["rewards"],
            rollouts["values"],
            rollouts["dones"],
            next_value,
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.config.num_epochs):
            # Create batches
            batch_size = min(self.config.batch_size, len(rollouts["obs"]))
            indices = torch.randperm(len(rollouts["obs"]))
            
            for start_idx in range(0, len(rollouts["obs"]), batch_size):
                end_idx = min(start_idx + batch_size, len(rollouts["obs"]))
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = rollouts["obs"][batch_indices]
                batch_actions = rollouts["actions"][batch_indices]
                batch_log_probs = rollouts["log_probs"][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Policy update
                _, new_log_probs = self.policy_net.get_action_and_log_prob(
                    batch_obs, batch_actions
                )
                
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value update
                values = self.value_net(batch_obs)
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                logits = self.policy_net(batch_obs)
                dist = Categorical(logits=logits)
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.config.value_loss_coef * value_loss + 
                    self.config.entropy_coef * entropy_loss
                )
                
                # Update networks
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), self.config.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.config.max_grad_norm
                )
                
                self.policy_optimizer.step()
                self.value_optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy_loss": np.mean(entropy_losses),
        }
    
    def train(self, total_timesteps: int, eval_freq: int = 10000) -> None:
        """Train the agent."""
        timesteps = 0
        episode = 0
        
        while timesteps < total_timesteps:
            # Collect rollouts
            rollouts = self.collect_rollouts(2048)  # Standard PPO rollout length
            timesteps += len(rollouts["obs"])
            
            # Update networks
            losses = self.update(rollouts)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(rollouts["rewards"])
                print(f"Episode {episode}, Timesteps {timesteps}")
                print(f"Average Reward: {avg_reward:.2f}")
                print(f"Policy Loss: {losses['policy_loss']:.4f}")
                print(f"Value Loss: {losses['value_loss']:.4f}")
                print(f"Entropy Loss: {losses['entropy_loss']:.4f}")
                print("-" * 50)
            
            episode += 1


class PolicyGradientTrainer:
    """Simple policy gradient trainer (baseline)."""
    
    def __init__(
        self,
        env: gym.Env,
        config: TrainingConfig,
        device: str = "auto",
    ):
        """Initialize policy gradient trainer."""
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
        
        # Initialize network
        self.policy_net = DialoguePolicyNetwork(
            self.obs_dim, self.action_dim
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=config.learning_rate
        )
    
    def _flatten_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        """Flatten observation dictionary to tensor."""
        flattened = []
        for key in sorted(obs.keys()):
            flattened.append(obs[key].flatten())
        return torch.cat(flattened).to(self.device)
    
    def train_episode(self) -> Dict[str, float]:
        """Train for one episode."""
        obs, _ = self.env.reset()
        obs_tensor = self._flatten_obs(obs)
        
        episode_rewards = []
        episode_log_probs = []
        
        done = False
        while not done:
            # Get action
            action, log_prob = self.policy_net.get_action_and_log_prob(
                obs_tensor.unsqueeze(0)
            )
            
            # Take action
            next_obs, reward, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated
            
            # Store data
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            
            # Update observation
            obs = next_obs
            obs_tensor = self._flatten_obs(obs)
        
        # Compute returns
        returns = []
        discounted_return = 0
        for reward in reversed(episode_rewards):
            discounted_return = reward + self.config.gamma * discounted_return
            returns.insert(0, discounted_return)
        
        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        log_probs = torch.stack(episode_log_probs).squeeze()
        loss = -(log_probs * returns).mean()
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "episode_reward": sum(episode_rewards),
            "episode_length": len(episode_rewards),
        }
    
    def train(self, num_episodes: int) -> None:
        """Train the agent."""
        for episode in range(num_episodes):
            metrics = self.train_episode()
            
            if episode % 100 == 0:
                print(f"Episode {episode}")
                print(f"Loss: {metrics['loss']:.4f}")
                print(f"Episode Reward: {metrics['episode_reward']:.2f}")
                print(f"Episode Length: {metrics['episode_length']}")
                print("-" * 50)
