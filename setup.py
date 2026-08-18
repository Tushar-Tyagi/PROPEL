#!/usr/bin/env python3
"""
Setup script for PROPEL (PROpensity-based-Position-bias-Elimination-for-LLMs)
"""

from setuptools import setup, find_packages

def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

setup(
    name="propel",
    version="1.0.0",
    author="PROPEL Authors",
    description="A model-agnostic framework for detecting and eliminating position bias in LLM recommendation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/Tushar-Tyagi/PROPEL",
    packages=find_packages(),
    py_modules=["LLM_debias"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Information Retrieval",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
        "anthropic": [
            "anthropic>=0.7.0",
        ],
    },
    zip_safe=False,
)
