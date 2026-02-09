"""Tests for RL algorithms."""

import pytest
import torch
import numpy as np
from src.algorithms import (
    PPOTrainer,
    SACTrainer, 
    PolicyGradientTrainer,
    TrainingConfig,
    DialoguePolicyNetwork,
    DialogueValueNetwork,
)


class TestTrainingConfig:
    """Test cases for TrainingConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TrainingConfig()
        
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.gae_lambda == 0.95
        assert config.clip_ratio == 0.2
        assert config.value_loss_coef == 0.5
        assert config.entropy_coef == 0.01
        assert config.max_grad_norm == 0.5
        assert config.num_epochs == 4
        assert config.batch_size == 64
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TrainingConfig(
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=128,
        )
        
        assert config.learning_rate == 1e-3
        assert config.gamma == 0.95
        assert config.batch_size == 128


class TestNetworks:
    """Test cases for neural networks."""
    
    def test_policy_network(self):
        """Test DialoguePolicyNetwork."""
        obs_dim = 128
        action_dim = 6
        hidden_dim = 256
        
        network = DialoguePolicyNetwork(obs_dim, action_dim, hidden_dim)
        
        # Test forward pass
        obs = torch.randn(1, obs_dim)
        logits = network(obs)
        
        assert logits.shape == (1, action_dim)
        
        # Test action selection
        action, log_prob = network.get_action_and_log_prob(obs)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert 0 <= action.item() < action_dim
    
    def test_value_network(self):
        """Test DialogueValueNetwork."""
        obs_dim = 128
        hidden_dim = 256
        
        network = DialogueValueNetwork(obs_dim, hidden_dim)
        
        # Test forward pass
        obs = torch.randn(1, obs_dim)
        value = network(obs)
        
        assert value.shape == (1,)
        assert isinstance(value.item(), float)
    
    def test_network_parameters(self):
        """Test network parameters."""
        obs_dim = 128
        action_dim = 6
        
        policy_net = DialoguePolicyNetwork(obs_dim, action_dim)
        value_net = DialogueValueNetwork(obs_dim)
        
        # Check that networks have parameters
        assert len(list(policy_net.parameters())) > 0
        assert len(list(value_net.parameters())) > 0
        
        # Check parameter shapes
        policy_params = list(policy_net.parameters())
        value_params = list(value_net.parameters())
        
        assert len(policy_params) > 0
        assert len(value_params) > 0


class TestPPOTrainer:
    """Test cases for PPOTrainer."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        import gymnasium as gym
        from gymnasium import spaces
        
        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Dict({
                    "dialogue_state": spaces.Discrete(7),
                    "turn_count": spaces.Discrete(21),
                    "context_embedding": spaces.Box(-1, 1, (128,)),
                    "required_info_mask": spaces.Box(0, 1, (10,)),
                    "provided_info_mask": spaces.Box(0, 1, (10,)),
                })
                self.action_space = spaces.Discrete(6)
            
            def reset(self, seed=None, options=None):
                obs = {
                    "dialogue_state": 0,
                    "turn_count": 0,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.zeros(10),
                    "provided_info_mask": np.zeros(10),
                }
                return obs, {}
            
            def step(self, action):
                obs = {
                    "dialogue_state": 1,
                    "turn_count": 1,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.ones(10),
                    "provided_info_mask": np.zeros(10),
                }
                return obs, 1.0, False, False, {}
        
        return MockEnv()
    
    def test_ppo_initialization(self, mock_env):
        """Test PPO trainer initialization."""
        config = TrainingConfig()
        trainer = PPOTrainer(mock_env, config)
        
        assert trainer.env == mock_env
        assert trainer.config == config
        assert trainer.device.type in ["cpu", "cuda", "mps"]
    
    def test_ppo_networks(self, mock_env):
        """Test PPO network creation."""
        config = TrainingConfig()
        trainer = PPOTrainer(mock_env, config)
        
        assert isinstance(trainer.policy_net, DialoguePolicyNetwork)
        assert isinstance(trainer.value_net, DialogueValueNetwork)
    
    def test_ppo_optimizers(self, mock_env):
        """Test PPO optimizer creation."""
        config = TrainingConfig()
        trainer = PPOTrainer(mock_env, config)
        
        assert isinstance(trainer.policy_optimizer, torch.optim.Adam)
        assert isinstance(trainer.value_optimizer, torch.optim.Adam)


class TestSACTrainer:
    """Test cases for SACTrainer."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        import gymnasium as gym
        from gymnasium import spaces
        
        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Dict({
                    "dialogue_state": spaces.Discrete(7),
                    "turn_count": spaces.Discrete(21),
                    "context_embedding": spaces.Box(-1, 1, (128,)),
                    "required_info_mask": spaces.Box(0, 1, (10,)),
                    "provided_info_mask": spaces.Box(0, 1, (10,)),
                })
                self.action_space = spaces.Discrete(6)
            
            def reset(self, seed=None, options=None):
                obs = {
                    "dialogue_state": 0,
                    "turn_count": 0,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.zeros(10),
                    "provided_info_mask": np.zeros(10),
                }
                return obs, {}
            
            def step(self, action):
                obs = {
                    "dialogue_state": 1,
                    "turn_count": 1,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.ones(10),
                    "provided_info_mask": np.zeros(10),
                }
                return obs, 1.0, False, False, {}
        
        return MockEnv()
    
    def test_sac_initialization(self, mock_env):
        """Test SAC trainer initialization."""
        config = TrainingConfig()
        trainer = SACTrainer(mock_env, config)
        
        assert trainer.env == mock_env
        assert trainer.config == config
        assert trainer.device.type in ["cpu", "cuda", "mps"]
    
    def test_sac_networks(self, mock_env):
        """Test SAC network creation."""
        config = TrainingConfig()
        trainer = SACTrainer(mock_env, config)
        
        assert isinstance(trainer.policy_net, DialoguePolicyNetwork)
        assert isinstance(trainer.q_net1, trainer.q_net1.__class__)
        assert isinstance(trainer.q_net2, trainer.q_net2.__class__)
    
    def test_sac_replay_buffer(self, mock_env):
        """Test SAC replay buffer."""
        config = TrainingConfig()
        trainer = SACTrainer(mock_env, config)
        
        assert isinstance(trainer.replay_buffer, trainer.replay_buffer.__class__)
        assert trainer.replay_buffer.capacity == config.buffer_size


class TestPolicyGradientTrainer:
    """Test cases for PolicyGradientTrainer."""
    
    @pytest.fixture
    def mock_env(self):
        """Create a mock environment."""
        import gymnasium as gym
        from gymnasium import spaces
        
        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Dict({
                    "dialogue_state": spaces.Discrete(7),
                    "turn_count": spaces.Discrete(21),
                    "context_embedding": spaces.Box(-1, 1, (128,)),
                    "required_info_mask": spaces.Box(0, 1, (10,)),
                    "provided_info_mask": spaces.Box(0, 1, (10,)),
                })
                self.action_space = spaces.Discrete(6)
            
            def reset(self, seed=None, options=None):
                obs = {
                    "dialogue_state": 0,
                    "turn_count": 0,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.zeros(10),
                    "provided_info_mask": np.zeros(10),
                }
                return obs, {}
            
            def step(self, action):
                obs = {
                    "dialogue_state": 1,
                    "turn_count": 1,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.ones(10),
                    "provided_info_mask": np.zeros(10),
                }
                return obs, 1.0, True, False, {}  # Terminate immediately
        
        return MockEnv()
    
    def test_policy_gradient_initialization(self, mock_env):
        """Test Policy Gradient trainer initialization."""
        config = TrainingConfig()
        trainer = PolicyGradientTrainer(mock_env, config)
        
        assert trainer.env == mock_env
        assert trainer.config == config
        assert trainer.device.type in ["cpu", "cuda", "mps"]
    
    def test_policy_gradient_network(self, mock_env):
        """Test Policy Gradient network creation."""
        config = TrainingConfig()
        trainer = PolicyGradientTrainer(mock_env, config)
        
        assert isinstance(trainer.policy_net, DialoguePolicyNetwork)
    
    def test_policy_gradient_optimizer(self, mock_env):
        """Test Policy Gradient optimizer creation."""
        config = TrainingConfig()
        trainer = PolicyGradientTrainer(mock_env, config)
        
        assert isinstance(trainer.optimizer, torch.optim.Adam)


if __name__ == "__main__":
    pytest.main([__file__])
