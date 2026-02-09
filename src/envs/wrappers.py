"""Environment wrappers for dialogue RL training."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics


class DialogueRewardWrapper(gym.RewardWrapper):
    """Wrapper to modify dialogue rewards for better RL training."""
    
    def __init__(
        self,
        env: gym.Env,
        success_reward: float = 10.0,
        failure_penalty: float = -5.0,
        step_penalty: float = -0.01,
        completion_bonus: float = 5.0,
    ):
        """Initialize the reward wrapper.
        
        Args:
            env: The environment to wrap
            success_reward: Reward for successful dialogue completion
            failure_penalty: Penalty for failed dialogue
            step_penalty: Small penalty per step to encourage efficiency
            completion_bonus: Bonus for high completion rate
        """
        super().__init__(env)
        self.success_reward = success_reward
        self.failure_penalty = failure_penalty
        self.step_penalty = step_penalty
        self.completion_bonus = completion_bonus
    
    def reward(self, reward: float) -> float:
        """Modify the reward based on dialogue performance."""
        # Get episode info
        info = self.env.unwrapped._get_info()
        completion_rate = info.get("completion_rate", 0.0)
        
        # Add step penalty
        modified_reward = reward + self.step_penalty
        
        # Add completion bonus
        if completion_rate > 0.8:
            modified_reward += self.completion_bonus
        
        return modified_reward


class DialogueObservationWrapper(gym.ObservationWrapper):
    """Wrapper to modify dialogue observations for better RL training."""
    
    def __init__(self, env: gym.Env, normalize: bool = True):
        """Initialize the observation wrapper.
        
        Args:
            env: The environment to wrap
            normalize: Whether to normalize observations
        """
        super().__init__(env)
        self.normalize = normalize
        
        # Update observation space if normalizing
        if normalize:
            self.observation_space = spaces.Dict({
                "dialogue_state": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "turn_count": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "context_embedding": spaces.Box(
                    low=-1.0, high=1.0, 
                    shape=env.observation_space["context_embedding"].shape, 
                    dtype=np.float32
                ),
                "required_info_mask": spaces.Box(
                    low=0.0, high=1.0, 
                    shape=env.observation_space["required_info_mask"].shape, 
                    dtype=np.float32
                ),
                "provided_info_mask": spaces.Box(
                    low=0.0, high=1.0, 
                    shape=env.observation_space["provided_info_mask"].shape, 
                    dtype=np.float32
                ),
            })
    
    def observation(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Modify the observation."""
        if not self.normalize:
            return obs
        
        # Normalize dialogue state
        dialogue_state = np.array([obs["dialogue_state"] / 6.0], dtype=np.float32)
        
        # Normalize turn count
        turn_count = np.array([obs["turn_count"] / self.env.max_turns], dtype=np.float32)
        
        return {
            "dialogue_state": dialogue_state,
            "turn_count": turn_count,
            "context_embedding": obs["context_embedding"].astype(np.float32),
            "required_info_mask": obs["required_info_mask"].astype(np.float32),
            "provided_info_mask": obs["provided_info_mask"].astype(np.float32),
        }


class DialogueActionWrapper(gym.ActionWrapper):
    """Wrapper to modify dialogue actions for better RL training."""
    
    def __init__(self, env: gym.Env, action_mapping: Optional[Dict[int, int]] = None):
        """Initialize the action wrapper.
        
        Args:
            env: The environment to wrap
            action_mapping: Optional mapping from new actions to original actions
        """
        super().__init__(env)
        
        if action_mapping is None:
            # Default mapping: keep all actions
            self.action_mapping = {i: i for i in range(env.action_space.n)}
        else:
            self.action_mapping = action_mapping
        
        # Update action space
        self.action_space = spaces.Discrete(len(self.action_mapping))
    
    def action(self, action: int) -> int:
        """Map the action to the original action space."""
        return self.action_mapping[action]


def make_dialogue_env(
    max_turns: int = 20,
    vocab_size: int = 1000,
    embedding_dim: int = 128,
    seed: Optional[int] = None,
    normalize_obs: bool = True,
    reward_shaping: bool = True,
) -> gym.Env:
    """Create a configured dialogue environment.
    
    Args:
        max_turns: Maximum number of conversation turns
        vocab_size: Size of vocabulary for text representation
        embedding_dim: Dimension of text embeddings
        seed: Random seed for reproducibility
        normalize_obs: Whether to normalize observations
        reward_shaping: Whether to apply reward shaping
        
    Returns:
        Configured dialogue environment
    """
    from .dialogue_env import DialogueEnvironment
    
    # Create base environment
    env = DialogueEnvironment(
        max_turns=max_turns,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seed=seed,
    )
    
    # Add time limit
    env = TimeLimit(env, max_episode_steps=max_turns)
    
    # Add episode statistics recording
    env = RecordEpisodeStatistics(env)
    
    # Add observation normalization
    if normalize_obs:
        env = DialogueObservationWrapper(env, normalize=True)
    
    # Add reward shaping
    if reward_shaping:
        env = DialogueRewardWrapper(env)
    
    return env
