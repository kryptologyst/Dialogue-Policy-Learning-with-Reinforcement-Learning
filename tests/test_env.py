"""Tests for dialogue environment."""

import pytest
import numpy as np
import gymnasium as gym
from src.envs import DialogueEnvironment, make_dialogue_env, DialogueAction, DialogueState


class TestDialogueEnvironment:
    """Test cases for DialogueEnvironment."""
    
    def test_environment_creation(self):
        """Test environment creation."""
        env = DialogueEnvironment(seed=42)
        assert env.max_turns == 20
        assert env.vocab_size == 1000
        assert env.embedding_dim == 128
    
    def test_reset(self):
        """Test environment reset."""
        env = DialogueEnvironment(seed=42)
        obs, info = env.reset()
        
        assert isinstance(obs, dict)
        assert "dialogue_state" in obs
        assert "turn_count" in obs
        assert "context_embedding" in obs
        assert "required_info_mask" in obs
        assert "provided_info_mask" in obs
        
        assert isinstance(info, dict)
        assert "user_intent" in info
        assert "required_info" in info
        assert "provided_info" in info
    
    def test_step(self):
        """Test environment step."""
        env = DialogueEnvironment(seed=42)
        obs, _ = env.reset()
        
        action = DialogueAction.GREET.value
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(next_obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_action_space(self):
        """Test action space."""
        env = DialogueEnvironment()
        assert env.action_space.n == len(DialogueAction)
    
    def test_observation_space(self):
        """Test observation space."""
        env = DialogueEnvironment()
        obs_space = env.observation_space
        
        assert isinstance(obs_space, gym.spaces.Dict)
        assert "dialogue_state" in obs_space
        assert "turn_count" in obs_space
        assert "context_embedding" in obs_space
        assert "required_info_mask" in obs_space
        assert "provided_info_mask" in obs_space
    
    def test_episode_termination(self):
        """Test episode termination."""
        env = DialogueEnvironment(max_turns=5, seed=42)
        obs, _ = env.reset()
        
        done = False
        step_count = 0
        
        while not done and step_count < 10:
            action = DialogueAction.END_CONVERSATION.value
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            step_count += 1
        
        assert done
        assert step_count <= 5  # Should terminate due to max_turns
    
    def test_render(self):
        """Test environment rendering."""
        env = DialogueEnvironment(seed=42)
        obs, _ = env.reset()
        
        # Should not raise an error
        result = env.render()
        assert result is None
    
    def test_close(self):
        """Test environment close."""
        env = DialogueEnvironment()
        # Should not raise an error
        env.close()


class TestDialogueWrappers:
    """Test cases for dialogue environment wrappers."""
    
    def test_make_dialogue_env(self):
        """Test make_dialogue_env function."""
        env = make_dialogue_env(seed=42)
        
        assert isinstance(env, gym.Env)
        
        # Test reset
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
        
        # Test step
        action = 0
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(next_obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_wrapper_configuration(self):
        """Test wrapper configuration."""
        env = make_dialogue_env(
            max_turns=10,
            normalize_obs=True,
            reward_shaping=True,
            seed=42
        )
        
        obs, _ = env.reset()
        
        # Check if observations are normalized
        assert obs["dialogue_state"].dtype == np.float32
        assert obs["turn_count"].dtype == np.float32


class TestDialogueActions:
    """Test cases for dialogue actions and states."""
    
    def test_dialogue_actions(self):
        """Test dialogue action enum."""
        assert DialogueAction.GREET.value == 0
        assert DialogueAction.ASK_QUESTION.value == 1
        assert DialogueAction.PROVIDE_INFO.value == 2
        assert DialogueAction.CLARIFY.value == 3
        assert DialogueAction.CONFIRM.value == 4
        assert DialogueAction.END_CONVERSATION.value == 5
    
    def test_dialogue_states(self):
        """Test dialogue state enum."""
        assert DialogueState.INITIAL.value == 0
        assert DialogueState.GREETING.value == 1
        assert DialogueState.QUESTIONING.value == 2
        assert DialogueState.INFORMING.value == 3
        assert DialogueState.CLARIFYING.value == 4
        assert DialogueState.CONFIRMING.value == 5
        assert DialogueState.ENDED.value == 6


if __name__ == "__main__":
    pytest.main([__file__])
