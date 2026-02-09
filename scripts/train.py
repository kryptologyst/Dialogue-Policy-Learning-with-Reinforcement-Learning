"""Main training script for dialogue policy learning."""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.train.trainer import DialogueTrainer, train_agent, compare_algorithms


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Main training function."""
    
    # Print safety warning
    print("=" * 80)
    print(config.safety_warning)
    print("=" * 80)
    
    # Print configuration
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(config))
    print("-" * 50)
    
    # Create trainer
    trainer = DialogueTrainer(config)
    
    # Train agent
    final_metrics = trainer.train()
    
    # Print final results
    print("\nFinal Results:")
    print("-" * 30)
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def compare(config: DictConfig) -> None:
    """Compare multiple algorithms."""
    
    # Print safety warning
    print("=" * 80)
    print(config.safety_warning)
    print("=" * 80)
    
    # Algorithms to compare
    algorithms = ["ppo", "sac", "policy_gradient"]
    
    # Compare algorithms
    results = compare_algorithms(
        config_path="../configs",
        algorithms=algorithms,
    )
    
    print("\nComparison completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        # Remove "compare" from argv and run comparison
        sys.argv = sys.argv[1:]
        compare()
    else:
        # Run normal training
        main()
