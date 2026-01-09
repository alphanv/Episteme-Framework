from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="episteme-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Active Bayesian Inference for Scientific Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/episteme-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "torch>=1.10.0",
        "scikit-learn>=1.0.0",
        "sbi>=0.19.0",
        "numpyro>=0.9.0",
        "jax>=0.3.0",
        "jaxlib>=0.3.0",
        "pyro-ppl>=1.8.0",
        "torchdiffeq>=0.2.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
)
