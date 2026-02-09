"""Evaluation metrics for dialogue policy learning."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gymnasium as gym


@dataclass
class DialogueMetrics:
    """Container for dialogue evaluation metrics."""
    
    # Task completion metrics
    success_rate: float = 0.0
    completion_rate: float = 0.0
    average_episode_length: float = 0.0
    
    # Reward metrics
    average_return: float = 0.0
    return_std: float = 0.0
    return_ci_95: Tuple[float, float] = (0.0, 0.0)
    
    # Dialogue quality metrics
    coherence_score: float = 0.0
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    
    # Action distribution metrics
    action_entropy: float = 0.0
    action_diversity: float = 0.0
    
    # Learning metrics
    sample_efficiency: float = 0.0
    convergence_step: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "success_rate": self.success_rate,
            "completion_rate": self.completion_rate,
            "average_episode_length": self.average_episode_length,
            "average_return": self.average_return,
            "return_std": self.return_std,
            "return_ci_95": self.return_ci_95,
            "coherence_score": self.coherence_score,
            "relevance_score": self.relevance_score,
            "diversity_score": self.diversity_score,
            "action_entropy": self.action_entropy,
            "action_diversity": self.action_diversity,
            "sample_efficiency": self.sample_efficiency,
            "convergence_step": self.convergence_step,
        }


class DialogueEvaluator:
    """Evaluator for dialogue policy learning agents."""
    
    def __init__(
        self,
        env: gym.Env,
        num_eval_episodes: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialize the evaluator.
        
        Args:
            env: Environment for evaluation
            num_eval_episodes: Number of episodes to evaluate
            seed: Random seed for reproducibility
        """
        self.env = env
        self.num_eval_episodes = num_eval_episodes
        self.seed = seed
        
        # Evaluation state
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_flags = []
        self.completion_rates = []
        self.action_sequences = []
        self.conversation_histories = []
    
    def evaluate_agent(
        self, 
        agent: Any, 
        deterministic: bool = True
    ) -> DialogueMetrics:
        """Evaluate an agent and return metrics.
        
        Args:
            agent: The agent to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dialogue metrics
        """
        self._reset_evaluation_state()
        
        # Run evaluation episodes
        for episode in range(self.num_eval_episodes):
            episode_metrics = self._evaluate_episode(agent, deterministic)
            self._update_evaluation_state(episode_metrics)
        
        # Compute final metrics
        return self._compute_metrics()
    
    def _reset_evaluation_state(self) -> None:
        """Reset evaluation state."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_flags = []
        self.completion_rates = []
        self.action_sequences = []
        self.conversation_histories = []
    
    def _evaluate_episode(
        self, 
        agent: Any, 
        deterministic: bool
    ) -> Dict[str, Any]:
        """Evaluate a single episode."""
        obs, info = self.env.reset(seed=self.seed)
        episode_reward = 0.0
        episode_length = 0
        action_sequence = []
        conversation_history = []
        
        done = False
        while not done:
            # Get action from agent
            if hasattr(agent, 'select_action'):
                action = agent.select_action(obs, deterministic=deterministic)
            elif hasattr(agent, 'predict'):
                action, _ = agent.predict(obs, deterministic=deterministic)
            else:
                raise ValueError("Agent must have select_action or predict method")
            
            # Take action
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store data
            episode_reward += reward
            episode_length += 1
            action_sequence.append(action)
            
            if hasattr(self.env.unwrapped, 'context') and self.env.unwrapped.context:
                conversation_history.append(
                    self.env.unwrapped.context.conversation_history[-1]
                    if self.env.unwrapped.context.conversation_history
                    else ("", "")
                )
            
            obs = next_obs
        
        # Determine success and completion rate
        success = self._determine_success(info)
        completion_rate = info.get("completion_rate", 0.0)
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "success": success,
            "completion_rate": completion_rate,
            "action_sequence": action_sequence,
            "conversation_history": conversation_history,
        }
    
    def _determine_success(self, info: Dict[str, Any]) -> bool:
        """Determine if episode was successful."""
        completion_rate = info.get("completion_rate", 0.0)
        return completion_rate >= 0.8  # 80% completion threshold
    
    def _update_evaluation_state(self, episode_metrics: Dict[str, Any]) -> None:
        """Update evaluation state with episode metrics."""
        self.episode_rewards.append(episode_metrics["episode_reward"])
        self.episode_lengths.append(episode_metrics["episode_length"])
        self.success_flags.append(episode_metrics["success"])
        self.completion_rates.append(episode_metrics["completion_rate"])
        self.action_sequences.append(episode_metrics["action_sequence"])
        self.conversation_histories.append(episode_metrics["conversation_history"])
    
    def _compute_metrics(self) -> DialogueMetrics:
        """Compute final evaluation metrics."""
        # Task completion metrics
        success_rate = np.mean(self.success_flags)
        completion_rate = np.mean(self.completion_rates)
        average_episode_length = np.mean(self.episode_lengths)
        
        # Reward metrics
        average_return = np.mean(self.episode_rewards)
        return_std = np.std(self.episode_rewards)
        return_ci_95 = self._compute_confidence_interval(self.episode_rewards)
        
        # Dialogue quality metrics
        coherence_score = self._compute_coherence_score()
        relevance_score = self._compute_relevance_score()
        diversity_score = self._compute_diversity_score()
        
        # Action distribution metrics
        action_entropy = self._compute_action_entropy()
        action_diversity = self._compute_action_diversity()
        
        return DialogueMetrics(
            success_rate=success_rate,
            completion_rate=completion_rate,
            average_episode_length=average_episode_length,
            average_return=average_return,
            return_std=return_std,
            return_ci_95=return_ci_95,
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            diversity_score=diversity_score,
            action_entropy=action_entropy,
            action_diversity=action_diversity,
        )
    
    def _compute_confidence_interval(
        self, 
        values: List[float], 
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Compute confidence interval for values."""
        n = len(values)
        mean = np.mean(values)
        std = np.std(values)
        
        # Use t-distribution for small samples
        if n < 30:
            from scipy import stats
            t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_val * std / np.sqrt(n)
        else:
            # Use normal distribution for large samples
            z_val = 1.96  # 95% confidence
            margin = z_val * std / np.sqrt(n)
        
        return (mean - margin, mean + margin)
    
    def _compute_coherence_score(self) -> float:
        """Compute dialogue coherence score."""
        # Simple coherence metric based on action transitions
        coherence_scores = []
        
        for action_seq in self.action_sequences:
            if len(action_seq) < 2:
                continue
            
            # Count valid transitions (simplified)
            valid_transitions = 0
            total_transitions = len(action_seq) - 1
            
            for i in range(total_transitions):
                current_action = action_seq[i]
                next_action = action_seq[i + 1]
                
                # Define valid transitions (simplified rules)
                valid_transitions += self._is_valid_transition(current_action, next_action)
            
            coherence_scores.append(valid_transitions / total_transitions if total_transitions > 0 else 0.0)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _is_valid_transition(self, current_action: int, next_action: int) -> bool:
        """Check if action transition is valid."""
        # Simplified transition rules
        valid_transitions = {
            0: [1, 2, 5],  # GREET -> ASK_QUESTION, PROVIDE_INFO, END_CONVERSATION
            1: [1, 2, 3, 4],  # ASK_QUESTION -> ASK_QUESTION, PROVIDE_INFO, CLARIFY, CONFIRM
            2: [1, 3, 4, 5],  # PROVIDE_INFO -> ASK_QUESTION, CLARIFY, CONFIRM, END_CONVERSATION
            3: [1, 2, 4],  # CLARIFY -> ASK_QUESTION, PROVIDE_INFO, CONFIRM
            4: [1, 2, 5],  # CONFIRM -> ASK_QUESTION, PROVIDE_INFO, END_CONVERSATION
            5: [],  # END_CONVERSATION -> (terminal)
        }
        
        return next_action in valid_transitions.get(current_action, [])
    
    def _compute_relevance_score(self) -> float:
        """Compute dialogue relevance score."""
        # Simple relevance metric based on information gathering
        relevance_scores = []
        
        for completion_rate in self.completion_rates:
            # Higher completion rate indicates better relevance
            relevance_scores.append(completion_rate)
        
        return np.mean(relevance_scores)
    
    def _compute_diversity_score(self) -> float:
        """Compute dialogue diversity score."""
        # Compute diversity based on action usage
        all_actions = []
        for action_seq in self.action_sequences:
            all_actions.extend(action_seq)
        
        if not all_actions:
            return 0.0
        
        # Compute entropy of action distribution
        action_counts = np.bincount(all_actions)
        action_probs = action_counts / np.sum(action_counts)
        action_probs = action_probs[action_probs > 0]  # Remove zero probabilities
        
        entropy = -np.sum(action_probs * np.log(action_probs))
        max_entropy = np.log(len(action_probs))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _compute_action_entropy(self) -> float:
        """Compute action entropy."""
        all_actions = []
        for action_seq in self.action_sequences:
            all_actions.extend(action_seq)
        
        if not all_actions:
            return 0.0
        
        action_counts = np.bincount(all_actions)
        action_probs = action_counts / np.sum(action_counts)
        action_probs = action_probs[action_probs > 0]
        
        return -np.sum(action_probs * np.log(action_probs))
    
    def _compute_action_diversity(self) -> float:
        """Compute action diversity (unique actions per episode)."""
        diversity_scores = []
        
        for action_seq in self.action_sequences:
            unique_actions = len(set(action_seq))
            total_actions = len(action_seq)
            diversity_scores.append(unique_actions / total_actions if total_actions > 0 else 0.0)
        
        return np.mean(diversity_scores)


class LearningCurveAnalyzer:
    """Analyzer for learning curves and convergence."""
    
    def __init__(self, window_size: int = 100):
        """Initialize the analyzer.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
    
    def add_episode(self, reward: float, length: int, success: bool) -> None:
        """Add episode data."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rates.append(success)
    
    def get_moving_averages(self) -> Dict[str, List[float]]:
        """Get moving averages of metrics."""
        if len(self.episode_rewards) < self.window_size:
            return {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "success_rates": self.success_rates,
            }
        
        rewards_ma = np.convolve(self.episode_rewards, np.ones(self.window_size)/self.window_size, mode='valid')
        lengths_ma = np.convolve(self.episode_lengths, np.ones(self.window_size)/self.window_size, mode='valid')
        success_ma = np.convolve(self.success_rates, np.ones(self.window_size)/self.window_size, mode='valid')
        
        return {
            "rewards": rewards_ma.tolist(),
            "lengths": lengths_ma.tolist(),
            "success_rates": success_ma.tolist(),
        }
    
    def find_convergence_point(self, threshold: float = 0.95) -> Optional[int]:
        """Find convergence point based on success rate."""
        if len(self.success_rates) < self.window_size:
            return None
        
        moving_averages = self.get_moving_averages()
        success_ma = moving_averages["success_rates"]
        
        for i, success_rate in enumerate(success_ma):
            if success_rate >= threshold:
                return i + self.window_size - 1
        
        return None
    
    def compute_sample_efficiency(self, target_success_rate: float = 0.8) -> float:
        """Compute sample efficiency to reach target success rate."""
        convergence_point = self.find_convergence_point(target_success_rate)
        
        if convergence_point is None:
            return float('inf')
        
        total_samples = sum(self.episode_lengths[:convergence_point + 1])
        return total_samples


def compare_agents(
    agents: Dict[str, Any],
    evaluator: DialogueEvaluator,
    deterministic: bool = True,
) -> Dict[str, DialogueMetrics]:
    """Compare multiple agents.
    
    Args:
        agents: Dictionary of agent names to agent objects
        evaluator: Dialogue evaluator
        deterministic: Whether to use deterministic policies
        
    Returns:
        Dictionary of agent names to metrics
    """
    results = {}
    
    for name, agent in agents.items():
        print(f"Evaluating {name}...")
        metrics = evaluator.evaluate_agent(agent, deterministic)
        results[name] = metrics
    
    return results


def print_comparison_table(results: Dict[str, DialogueMetrics]) -> None:
    """Print comparison table of agent results."""
    print("\n" + "="*80)
    print("AGENT COMPARISON RESULTS")
    print("="*80)
    
    # Header
    print(f"{'Agent':<20} {'Success Rate':<12} {'Avg Return':<12} {'Ep Length':<10} {'Coherence':<12}")
    print("-"*80)
    
    # Results
    for name, metrics in results.items():
        print(f"{name:<20} {metrics.success_rate:<12.3f} {metrics.average_return:<12.2f} "
              f"{metrics.average_episode_length:<10.1f} {metrics.coherence_score:<12.3f}")
    
    print("="*80)
    
    # Detailed metrics
    print("\nDETAILED METRICS:")
    print("-"*50)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Success Rate: {metrics.success_rate:.3f}")
        print(f"  Completion Rate: {metrics.completion_rate:.3f}")
        print(f"  Average Return: {metrics.average_return:.2f} ± {metrics.return_std:.2f}")
        print(f"  Return 95% CI: [{metrics.return_ci_95[0]:.2f}, {metrics.return_ci_95[1]:.2f}]")
        print(f"  Coherence Score: {metrics.coherence_score:.3f}")
        print(f"  Relevance Score: {metrics.relevance_score:.3f}")
        print(f"  Diversity Score: {metrics.diversity_score:.3f}")
        print(f"  Action Entropy: {metrics.action_entropy:.3f}")
        print(f"  Action Diversity: {metrics.action_diversity:.3f}")
