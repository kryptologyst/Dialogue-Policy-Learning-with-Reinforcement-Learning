"""Tests for evaluation metrics."""

import pytest
import numpy as np
from src.eval.metrics import (
    DialogueMetrics,
    DialogueEvaluator,
    LearningCurveAnalyzer,
    compare_agents,
    print_comparison_table,
)


class TestDialogueMetrics:
    """Test cases for DialogueMetrics."""
    
    def test_default_metrics(self):
        """Test default metrics values."""
        metrics = DialogueMetrics()
        
        assert metrics.success_rate == 0.0
        assert metrics.completion_rate == 0.0
        assert metrics.average_episode_length == 0.0
        assert metrics.average_return == 0.0
        assert metrics.return_std == 0.0
        assert metrics.return_ci_95 == (0.0, 0.0)
    
    def test_custom_metrics(self):
        """Test custom metrics values."""
        metrics = DialogueMetrics(
            success_rate=0.8,
            completion_rate=0.75,
            average_return=10.5,
            coherence_score=0.7,
        )
        
        assert metrics.success_rate == 0.8
        assert metrics.completion_rate == 0.75
        assert metrics.average_return == 10.5
        assert metrics.coherence_score == 0.7
    
    def test_to_dict(self):
        """Test metrics to dictionary conversion."""
        metrics = DialogueMetrics(
            success_rate=0.8,
            completion_rate=0.75,
            average_return=10.5,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert metrics_dict["success_rate"] == 0.8
        assert metrics_dict["completion_rate"] == 0.75
        assert metrics_dict["average_return"] == 10.5


class TestDialogueEvaluator:
    """Test cases for DialogueEvaluator."""
    
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
                self.context = None
            
            def reset(self, seed=None, options=None):
                obs = {
                    "dialogue_state": 0,
                    "turn_count": 0,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.zeros(10),
                    "provided_info_mask": np.zeros(10),
                }
                
                # Mock context
                class MockContext:
                    def __init__(self):
                        self.user_intent = "test_intent"
                        self.required_info = ["info1", "info2"]
                        self.provided_info = []
                        self.conversation_history = []
                
                self.context = MockContext()
                
                return obs, {}
            
            def step(self, action):
                obs = {
                    "dialogue_state": 1,
                    "turn_count": 1,
                    "context_embedding": np.random.randn(128),
                    "required_info_mask": np.ones(10),
                    "provided_info_mask": np.zeros(10),
                }
                
                # Mock completion rate
                completion_rate = 0.5
                
                info = {
                    "completion_rate": completion_rate,
                    "user_intent": "test_intent",
                    "required_info": ["info1", "info2"],
                    "provided_info": ["info1"],
                }
                
                return obs, 1.0, True, False, info
        
        return MockEnv()
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent."""
        class MockAgent:
            def select_action(self, obs, deterministic=True):
                return 0
            
            def predict(self, obs, deterministic=True):
                return 0, None
        
        return MockAgent()
    
    def test_evaluator_initialization(self, mock_env):
        """Test evaluator initialization."""
        evaluator = DialogueEvaluator(mock_env, num_eval_episodes=10, seed=42)
        
        assert evaluator.env == mock_env
        assert evaluator.num_eval_episodes == 10
        assert evaluator.seed == 42
    
    def test_evaluate_agent(self, mock_env, mock_agent):
        """Test agent evaluation."""
        evaluator = DialogueEvaluator(mock_env, num_eval_episodes=5, seed=42)
        
        metrics = evaluator.evaluate_agent(mock_agent, deterministic=True)
        
        assert isinstance(metrics, DialogueMetrics)
        assert 0.0 <= metrics.success_rate <= 1.0
        assert 0.0 <= metrics.completion_rate <= 1.0
        assert metrics.average_return >= 0.0
        assert metrics.average_episode_length >= 0.0
    
    def test_determine_success(self, mock_env):
        """Test success determination."""
        evaluator = DialogueEvaluator(mock_env, num_eval_episodes=5, seed=42)
        
        # Test successful episode
        info_success = {"completion_rate": 0.9}
        assert evaluator._determine_success(info_success) == True
        
        # Test unsuccessful episode
        info_failure = {"completion_rate": 0.5}
        assert evaluator._determine_success(info_failure) == False
    
    def test_compute_confidence_interval(self, mock_env):
        """Test confidence interval computation."""
        evaluator = DialogueEvaluator(mock_env, num_eval_episodes=5, seed=42)
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = evaluator._compute_confidence_interval(values)
        
        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert ci[0] < ci[1]
        assert ci[0] <= np.mean(values) <= ci[1]
    
    def test_is_valid_transition(self, mock_env):
        """Test valid transition checking."""
        evaluator = DialogueEvaluator(mock_env, num_eval_episodes=5, seed=42)
        
        # Test valid transitions
        assert evaluator._is_valid_transition(0, 1) == True  # GREET -> ASK_QUESTION
        assert evaluator._is_valid_transition(1, 2) == True  # ASK_QUESTION -> PROVIDE_INFO
        assert evaluator._is_valid_transition(4, 5) == True  # CONFIRM -> END_CONVERSATION
        
        # Test invalid transitions
        assert evaluator._is_valid_transition(5, 0) == False  # END_CONVERSATION -> GREET
        assert evaluator._is_valid_transition(0, 3) == False  # GREET -> CLARIFY


class TestLearningCurveAnalyzer:
    """Test cases for LearningCurveAnalyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = LearningCurveAnalyzer(window_size=10)
        
        assert analyzer.window_size == 10
        assert analyzer.episode_rewards == []
        assert analyzer.episode_lengths == []
        assert analyzer.success_rates == []
    
    def test_add_episode(self):
        """Test adding episode data."""
        analyzer = LearningCurveAnalyzer(window_size=10)
        
        analyzer.add_episode(reward=5.0, length=10, success=True)
        analyzer.add_episode(reward=3.0, length=8, success=False)
        
        assert len(analyzer.episode_rewards) == 2
        assert len(analyzer.episode_lengths) == 2
        assert len(analyzer.success_rates) == 2
        
        assert analyzer.episode_rewards == [5.0, 3.0]
        assert analyzer.episode_lengths == [10, 8]
        assert analyzer.success_rates == [True, False]
    
    def test_get_moving_averages(self):
        """Test moving averages computation."""
        analyzer = LearningCurveAnalyzer(window_size=3)
        
        # Add episodes
        for i in range(5):
            analyzer.add_episode(reward=float(i), length=i, success=i % 2 == 0)
        
        moving_averages = analyzer.get_moving_averages()
        
        assert "rewards" in moving_averages
        assert "lengths" in moving_averages
        assert "success_rates" in moving_averages
        
        # Check that moving averages are computed correctly
        assert len(moving_averages["rewards"]) == 3  # 5 - 3 + 1
        assert len(moving_averages["lengths"]) == 3
        assert len(moving_averages["success_rates"]) == 3
    
    def test_find_convergence_point(self):
        """Test convergence point finding."""
        analyzer = LearningCurveAnalyzer(window_size=3)
        
        # Add episodes with improving success rate
        for i in range(10):
            success = i >= 6  # Success rate improves after episode 6
            analyzer.add_episode(reward=float(i), length=10, success=success)
        
        convergence_point = analyzer.find_convergence_point(threshold=0.8)
        
        assert convergence_point is not None
        assert convergence_point >= 6  # Should converge after episode 6
    
    def test_compute_sample_efficiency(self):
        """Test sample efficiency computation."""
        analyzer = LearningCurveAnalyzer(window_size=3)
        
        # Add episodes
        for i in range(10):
            success = i >= 5  # Success rate improves after episode 5
            analyzer.add_episode(reward=float(i), length=10, success=success)
        
        sample_efficiency = analyzer.compute_sample_efficiency(target_success_rate=0.8)
        
        assert isinstance(sample_efficiency, float)
        assert sample_efficiency > 0


class TestComparisonFunctions:
    """Test cases for comparison functions."""
    
    def test_compare_agents(self):
        """Test agent comparison."""
        # Mock agents and evaluator
        class MockAgent:
            def __init__(self, name):
                self.name = name
            
            def select_action(self, obs, deterministic=True):
                return 0
        
        class MockEvaluator:
            def evaluate_agent(self, agent, deterministic=True):
                metrics = DialogueMetrics(
                    success_rate=0.8,
                    completion_rate=0.75,
                    average_return=10.0,
                )
                return metrics
        
        agents = {
            "agent1": MockAgent("agent1"),
            "agent2": MockAgent("agent2"),
        }
        
        evaluator = MockEvaluator()
        
        results = compare_agents(agents, evaluator, deterministic=True)
        
        assert isinstance(results, dict)
        assert "agent1" in results
        assert "agent2" in results
        assert isinstance(results["agent1"], DialogueMetrics)
        assert isinstance(results["agent2"], DialogueMetrics)
    
    def test_print_comparison_table(self, capsys):
        """Test comparison table printing."""
        metrics1 = DialogueMetrics(
            success_rate=0.8,
            completion_rate=0.75,
            average_return=10.0,
            average_episode_length=15.0,
            coherence_score=0.7,
        )
        
        metrics2 = DialogueMetrics(
            success_rate=0.6,
            completion_rate=0.65,
            average_return=8.0,
            average_episode_length=18.0,
            coherence_score=0.6,
        )
        
        results = {
            "PPO": metrics1,
            "SAC": metrics2,
        }
        
        print_comparison_table(results)
        
        captured = capsys.readouterr()
        assert "AGENT COMPARISON RESULTS" in captured.out
        assert "PPO" in captured.out
        assert "SAC" in captured.out


if __name__ == "__main__":
    pytest.main([__file__])
