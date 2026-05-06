from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="thermodynamic-marl",
    version="0.1.0",
    author="Kaushal Singh",
    author_email="your.email@example.com",
    description="Thermodynamic Multi-Agent Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KaushalSingh2007/thermodynamic-marl",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "networkx>=2.6",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
    ],
)