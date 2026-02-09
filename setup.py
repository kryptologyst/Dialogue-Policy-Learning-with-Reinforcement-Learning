"""Setup script for the dialogue RL project."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split("\n")
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

setup(
    name="dialogue-rl",
    version="1.0.0",
    description="Reinforcement Learning for Dialogue Policy Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Research Team",
    author_email="research@example.com",
    url="https://github.com/your-repo/dialogue-rl",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "dialogue-train=scripts.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
