"""Logging utilities for dialogue RL training."""

import os
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class LogEntry:
    """Container for log entry."""
    step: int
    episode: int
    metrics: Dict[str, float]
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class TensorBoardLogger:
    """TensorBoard logger for RL training."""
    
    def __init__(
        self,
        log_dir: str,
        log_frequency: int = 100,
        save_model: bool = True,
        save_frequency: int = 10000,
        log_metrics: Optional[List[str]] = None,
        plot_frequency: int = 1000,
        save_plots: bool = True,
    ):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save logs
            log_frequency: Frequency of logging (in steps)
            save_model: Whether to save model checkpoints
            save_frequency: Frequency of model saving (in steps)
            log_metrics: List of metrics to log
            plot_frequency: Frequency of plot generation (in steps)
            save_plots: Whether to save plots
        """
        self.log_dir = Path(log_dir)
        self.log_frequency = log_frequency
        self.save_model = save_model
        self.save_frequency = save_frequency
        self.log_metrics = log_metrics or []
        self.plot_frequency = plot_frequency
        self.save_plots = save_plots
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Log storage
        self.log_entries = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        
        # Plotting setup
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def log(
        self,
        step: int,
        episode: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Log metrics and optionally save model.
        
        Args:
            step: Current training step
            episode: Current episode
            metrics: Dictionary of metrics to log
            model: Model to save (if save_model is True)
        """
        # Create log entry
        entry = LogEntry(step=step, episode=episode, metrics=metrics)
        self.log_entries.append(entry)
        
        # Log to TensorBoard
        for key, value in metrics.items():
            if key in self.log_metrics or not self.log_metrics:
                self.writer.add_scalar(key, value, step)
        
        # Store episode-level metrics
        if "episode_reward" in metrics:
            self.episode_rewards.append(metrics["episode_reward"])
        if "episode_length" in metrics:
            self.episode_lengths.append(metrics["episode_length"])
        if "success_rate" in metrics:
            self.success_rates.append(metrics["success_rate"])
        
        # Save model if requested
        if self.save_model and model is not None and step % self.save_frequency == 0:
            self.save_model_checkpoint(model, step)
        
        # Generate plots if requested
        if self.save_plots and step % self.plot_frequency == 0:
            self.generate_plots(step)
        
        # Print progress
        if step % self.log_frequency == 0:
            self.print_progress(step, episode, metrics)
    
    def save_model_checkpoint(self, model: torch.nn.Module, step: int) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.log_dir / f"model_step_{step}.pt"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'timestamp': time.time(),
        }, checkpoint_path)
    
    def generate_plots(self, step: int) -> None:
        """Generate and save training plots."""
        if not self.episode_rewards:
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Step {step}', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True)
        
        # Success rates (if available)
        if self.success_rates:
            axes[1, 0].plot(self.success_rates)
            axes[1, 0].set_title('Success Rates')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Success Rate')
            axes[1, 0].grid(True)
        
        # Moving averages
        if len(self.episode_rewards) > 100:
            window_size = min(100, len(self.episode_rewards) // 10)
            moving_avg = np.convolve(
                self.episode_rewards, 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            axes[1, 1].plot(moving_avg)
            axes[1, 1].set_title(f'Moving Average Rewards (window={window_size})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True)
        
        # Save plot
        plot_path = self.log_dir / f"training_plots_step_{step}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_progress(self, step: int, episode: int, metrics: Dict[str, float]) -> None:
        """Print training progress."""
        print(f"\nStep {step}, Episode {episode}")
        print("-" * 50)
        
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        print("-" * 50)
    
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        self.writer.add_hparams(config, {})
    
    def close(self) -> None:
        """Close the logger."""
        self.writer.close()
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best model checkpoint."""
        checkpoint_files = list(self.log_dir.glob("model_step_*.pt"))
        if not checkpoint_files:
            return None
        
        # Find checkpoint with highest average reward
        best_checkpoint = None
        best_reward = float('-inf')
        
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                step = checkpoint['step']
                
                # Find corresponding reward
                for entry in self.log_entries:
                    if entry.step == step and "episode_reward" in entry.metrics:
                        if entry.metrics["episode_reward"] > best_reward:
                            best_reward = entry.metrics["episode_reward"]
                            best_checkpoint = str(checkpoint_file)
                        break
            except Exception:
                continue
        
        return best_checkpoint


class ConsoleLogger:
    """Simple console logger for basic training output."""
    
    def __init__(self, log_frequency: int = 100):
        """Initialize console logger.
        
        Args:
            log_frequency: Frequency of logging (in steps)
        """
        self.log_frequency = log_frequency
        self.start_time = time.time()
    
    def log(
        self,
        step: int,
        episode: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Log metrics to console."""
        if step % self.log_frequency == 0:
            elapsed_time = time.time() - self.start_time
            print(f"\nStep {step}, Episode {episode}, Time: {elapsed_time:.1f}s")
            print("-" * 50)
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
            print("-" * 50)
    
    def close(self) -> None:
        """Close the logger."""
        pass


def create_logger(logger_type: str, **kwargs) -> Union[TensorBoardLogger, ConsoleLogger]:
    """Create logger based on type.
    
    Args:
        logger_type: Type of logger ('tensorboard' or 'console')
        **kwargs: Additional arguments for logger
        
    Returns:
        Logger instance
    """
    if logger_type.lower() == 'tensorboard':
        return TensorBoardLogger(**kwargs)
    elif logger_type.lower() == 'console':
        return ConsoleLogger(**kwargs)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")
