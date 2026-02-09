"""Training utilities for dialogue RL."""

import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from ..algorithms import PPOTrainer, SACTrainer, PolicyGradientTrainer, TrainingConfig
from ..envs import make_dialogue_env
from ..eval.metrics import DialogueEvaluator, DialogueMetrics, compare_agents, print_comparison_table
from .logging import TensorBoardLogger, ConsoleLogger, create_logger
from .seeding import set_seed


class DialogueTrainer:
    """Main trainer class for dialogue policy learning."""
    
    def __init__(self, config: DictConfig):
        """Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Set random seeds
        set_seed(config.seed)
        
        # Create environment
        self.env = self._create_environment()
        
        # Create agent
        self.agent = self._create_agent()
        
        # Create logger
        self.logger = self._create_logger()
        
        # Create evaluator
        self.evaluator = DialogueEvaluator(
            env=self.env,
            num_eval_episodes=100,
            seed=config.seed,
        )
        
        # Training state
        self.total_timesteps = 0
        self.episode_count = 0
        self.start_time = time.time()
    
    def _create_environment(self) -> gym.Env:
        """Create training environment."""
        env_config = self.config.env
        
        # Create environment with Hydra instantiation
        if hasattr(env_config, '_target_'):
            env = OmegaConf.instantiate(env_config, seed=self.config.seed)
        else:
            # Fallback to manual creation
            env = make_dialogue_env(
                max_turns=env_config.max_turns,
                vocab_size=env_config.vocab_size,
                embedding_dim=env_config.embedding_dim,
                seed=self.config.seed,
                normalize_obs=env_config.normalize_obs,
                reward_shaping=env_config.reward_shaping,
            )
        
        return env
    
    def _create_agent(self) -> Union[PPOTrainer, SACTrainer, PolicyGradientTrainer]:
        """Create training agent."""
        algorithm_config = self.config.algorithm
        
        # Convert config to TrainingConfig
        training_config = TrainingConfig(
            learning_rate=algorithm_config.learning_rate,
            gamma=algorithm_config.gamma,
            gae_lambda=getattr(algorithm_config, 'gae_lambda', 0.95),
            clip_ratio=getattr(algorithm_config, 'clip_ratio', 0.2),
            value_loss_coef=getattr(algorithm_config, 'value_loss_coef', 0.5),
            entropy_coef=getattr(algorithm_config, 'entropy_coef', 0.01),
            max_grad_norm=getattr(algorithm_config, 'max_grad_norm', 0.5),
            num_epochs=getattr(algorithm_config, 'num_epochs', 4),
            batch_size=algorithm_config.batch_size,
            buffer_size=getattr(algorithm_config, 'buffer_size', 10000),
            target_entropy=getattr(algorithm_config, 'target_entropy', None),
            tau=getattr(algorithm_config, 'tau', 0.005),
            alpha=getattr(algorithm_config, 'alpha', 0.2),
            auto_entropy=getattr(algorithm_config, 'auto_entropy', True),
        )
        
        # Create agent with Hydra instantiation
        if hasattr(algorithm_config, '_target_'):
            agent = OmegaConf.instantiate(
                algorithm_config,
                env=self.env,
                config=training_config,
                device=self.config.device,
            )
        else:
            # Fallback to manual creation based on algorithm type
            algorithm_name = self.config.algorithm.get('name', 'ppo')
            
            if algorithm_name.lower() == 'ppo':
                agent = PPOTrainer(
                    env=self.env,
                    config=training_config,
                    device=self.config.device,
                )
            elif algorithm_name.lower() == 'sac':
                agent = SACTrainer(
                    env=self.env,
                    config=training_config,
                    device=self.config.device,
                )
            elif algorithm_name.lower() == 'policy_gradient':
                agent = PolicyGradientTrainer(
                    env=self.env,
                    config=training_config,
                    device=self.config.device,
                )
            else:
                raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        return agent
    
    def _create_logger(self) -> Union[TensorBoardLogger, ConsoleLogger]:
        """Create logger."""
        logging_config = self.config.logging
        
        logger_kwargs = {
            'log_frequency': logging_config.log_frequency,
            'save_model': logging_config.save_model,
            'save_frequency': logging_config.save_frequency,
            'log_metrics': logging_config.log_metrics,
            'plot_frequency': logging_config.plot_frequency,
            'save_plots': logging_config.save_plots,
        }
        
        if hasattr(logging_config, '_target_'):
            logger = OmegaConf.instantiate(logging_config, **logger_kwargs)
        else:
            logger_type = logging_config.get('type', 'tensorboard')
            if logger_type == 'tensorboard':
                logger_kwargs['log_dir'] = logging_config.log_dir
            logger = create_logger(logger_type, **logger_kwargs)
        
        return logger
    
    def train(self) -> Dict[str, Any]:
        """Train the agent."""
        print("Starting training...")
        print(f"Algorithm: {self.config.algorithm.get('name', 'unknown')}")
        print(f"Environment: {self.config.env.get('name', 'dialogue_env')}")
        print(f"Total timesteps: {self.config.max_total_timesteps}")
        print(f"Device: {self.config.device}")
        print("-" * 50)
        
        # Log hyperparameters
        hyperparams = OmegaConf.to_container(self.config, resolve=True)
        if hasattr(self.logger, 'log_hyperparameters'):
            self.logger.log_hyperparameters(hyperparams)
        
        # Training loop
        while self.total_timesteps < self.config.max_total_timesteps:
            # Train one episode/batch
            episode_metrics = self._train_step()
            
            # Log metrics
            self.logger.log(
                step=self.total_timesteps,
                episode=self.episode_count,
                metrics=episode_metrics,
                model=getattr(self.agent, 'policy_net', None),
            )
            
            # Evaluation
            if self.total_timesteps % self.config.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self.logger.log(
                    step=self.total_timesteps,
                    episode=self.episode_count,
                    metrics=eval_metrics,
                )
            
            self.episode_count += 1
        
        # Final evaluation
        print("\nTraining completed!")
        final_metrics = self._evaluate()
        print("Final evaluation results:")
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}")
        
        # Close logger
        self.logger.close()
        
        return final_metrics
    
    def _train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        if hasattr(self.agent, 'train'):
            # For algorithms with batch training (PPO, SAC)
            self.agent.train(self.config.max_total_timesteps)
            return {"training_completed": 1.0}
        elif hasattr(self.agent, 'train_episode'):
            # For algorithms with episode-based training (Policy Gradient)
            return self.agent.train_episode()
        else:
            raise ValueError("Agent must have train or train_episode method")
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate the agent."""
        print(f"\nEvaluating at step {self.total_timesteps}...")
        
        metrics = self.evaluator.evaluate_agent(
            agent=self.agent,
            deterministic=self.config.deterministic,
        )
        
        # Convert metrics to dictionary
        eval_metrics = metrics.to_dict()
        
        # Add evaluation prefix
        eval_metrics = {f"eval_{k}": v for k, v in eval_metrics.items()}
        
        return eval_metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'config': OmegaConf.to_container(self.config, resolve=True),
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'agent_state': getattr(self.agent, 'state_dict', lambda: {})(),
            'timestamp': time.time(),
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location='cpu')
        
        self.total_timesteps = checkpoint['total_timesteps']
        self.episode_count = checkpoint['episode_count']
        
        if hasattr(self.agent, 'load_state_dict'):
            self.agent.load_state_dict(checkpoint['agent_state'])
        
        print(f"Checkpoint loaded from {path}")


def train_agent(config_path: str, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """Train an agent with the given configuration.
    
    Args:
        config_path: Path to configuration file
        overrides: List of configuration overrides
        
    Returns:
        Final evaluation metrics
    """
    from hydra import initialize, compose
    
    with initialize(config_path=config_path):
        config = compose(config_name="config", overrides=overrides or [])
    
    trainer = DialogueTrainer(config)
    return trainer.train()


def compare_algorithms(
    config_path: str,
    algorithms: List[str],
    overrides: Optional[List[str]] = None,
) -> Dict[str, DialogueMetrics]:
    """Compare multiple algorithms.
    
    Args:
        config_path: Path to configuration file
        algorithms: List of algorithm names to compare
        overrides: List of configuration overrides
        
    Returns:
        Dictionary of algorithm names to metrics
    """
    from hydra import initialize, compose
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\nTraining {algorithm}...")
        
        with initialize(config_path=config_path):
            config = compose(
                config_name="config", 
                overrides=(overrides or []) + [f"algorithm={algorithm}"]
            )
        
        trainer = DialogueTrainer(config)
        trainer.train()
        
        # Evaluate final performance
        metrics = trainer.evaluator.evaluate_agent(
            agent=trainer.agent,
            deterministic=True,
        )
        
        results[algorithm] = metrics
    
    # Print comparison table
    print_comparison_table(results)
    
    return results
