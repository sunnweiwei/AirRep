from setuptools import setup, find_packages

setup(
    name="airrep",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentence-transformers",
    ],
    python_requires=">=3.8",
)
