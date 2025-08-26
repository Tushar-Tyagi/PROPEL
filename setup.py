#!/usr/bin/env python3
"""
Setup script for LLM Position Bias Analysis Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="llm-position-bias",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive framework for detecting and correcting position bias in LLM-based recommender systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/debiased_ranking",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/debiased_ranking/issues",
        "Source": "https://github.com/yourusername/debiased_ranking",
        "Documentation": "https://github.com/yourusername/debiased_ranking#readme",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: Jupyter",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-asyncio>=0.21.0",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "full": [
            "torch>=1.12.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-bias-analyzer=LLM_debias:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "llm",
        "position-bias",
        "recommender-systems",
        "machine-learning",
        "artificial-intelligence",
        "bias-detection",
        "debiasing",
        "ranking",
        "evaluation",
        "ndcg",
        "propensity-scoring",
    ],
    platforms=["any"],
    license="MIT",
)
