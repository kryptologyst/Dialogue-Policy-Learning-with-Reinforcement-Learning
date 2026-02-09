"""Test configuration and setup."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "seed": 42,
        "device": "cpu",
        "max_timesteps": 1000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
    }


def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.envs import DialogueEnvironment, make_dialogue_env
        from src.algorithms import PPOTrainer, SACTrainer, PolicyGradientTrainer
        from src.eval.metrics import DialogueEvaluator, DialogueMetrics
        from src.utils.seeding import set_seed
        from src.utils.logging import TensorBoardLogger, ConsoleLogger
    except ImportError as e:
        pytest.fail(f"Failed to import modules: {e}")


def test_basic_functionality(sample_config):
    """Test basic functionality."""
    from src.envs import make_dialogue_env
    from src.utils.seeding import set_seed
    
    # Test seeding
    seed = set_seed(sample_config["seed"])
    assert seed == sample_config["seed"]
    
    # Test environment creation
    env = make_dialogue_env(seed=sample_config["seed"])
    assert env is not None
    
    # Test environment reset
    obs, info = env.reset()
    assert obs is not None
    assert info is not None
    
    # Test environment step
    action = 0
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs is not None
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info is not None
