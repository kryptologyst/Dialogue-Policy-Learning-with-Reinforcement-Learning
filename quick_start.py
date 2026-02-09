"""Quick start script for dialogue RL project."""

#!/usr/bin/env python3
"""
Quick start script for Dialogue Policy Learning with RL.

This script provides a simple way to get started with the project,
including environment setup, basic training, and demo launch.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible!")
    return True


def install_dependencies() -> bool:
    """Install project dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def install_package() -> bool:
    """Install package in development mode."""
    return run_command("pip install -e .", "Installing package in development mode")


def run_tests() -> bool:
    """Run basic tests."""
    return run_command("python -m pytest tests/ -v", "Running tests")


def train_basic() -> bool:
    """Run basic training."""
    return run_command(
        "python scripts/train.py algorithm=policy_gradient total_timesteps=1000",
        "Running basic training"
    )


def launch_demo() -> bool:
    """Launch Streamlit demo."""
    print("\n🚀 Launching Streamlit demo...")
    print("The demo will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the demo")
    
    try:
        subprocess.run("streamlit run demo/app.py", shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo launch failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick start for Dialogue RL project")
    parser.add_argument(
        "--skip-install", 
        action="store_true", 
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-tests", 
        action="store_true", 
        help="Skip running tests"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Skip basic training"
    )
    parser.add_argument(
        "--demo-only", 
        action="store_true", 
        help="Only launch the demo"
    )
    
    args = parser.parse_args()
    
    print("🎯 Dialogue Policy Learning - Quick Start")
    print("=" * 50)
    
    # Safety warning
    print("""
⚠️  SAFETY WARNING ⚠️

This is a research/educational project for dialogue policy learning.
NOT FOR PRODUCTION CONTROL OF REAL SYSTEMS.

This system is designed for:
- Research and educational purposes
- Demonstrating RL concepts in dialogue systems
- Academic study of conversation policies

DO NOT USE for:
- Production dialogue systems
- Real-world customer service
- Critical decision-making systems
- Systems requiring guaranteed safety or reliability
""")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Demo only mode
    if args.demo_only:
        return launch_demo()
    
    # Installation
    if not args.skip_install:
        if not install_dependencies():
            sys.exit(1)
        
        if not install_package():
            sys.exit(1)
    
    # Tests
    if not args.skip_tests:
        if not run_tests():
            print("⚠️  Tests failed, but continuing...")
    
    # Basic training
    if not args.skip_training:
        if not train_basic():
            print("⚠️  Training failed, but continuing...")
    
    # Launch demo
    print("\n🎉 Setup completed! Ready to explore dialogue RL.")
    print("\nNext steps:")
    print("1. Launch the interactive demo: streamlit run demo/app.py")
    print("2. Train with different algorithms: python scripts/train.py algorithm=ppo")
    print("3. Compare algorithms: python scripts/train.py compare")
    print("4. Explore the code in src/ directory")
    
    response = input("\n🚀 Launch demo now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        launch_demo()
    else:
        print("👋 Happy learning! Run 'streamlit run demo/app.py' when ready.")


if __name__ == "__main__":
    main()
