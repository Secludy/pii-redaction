from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pii-redaction",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool for redacting PII information from text using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pii-redaction",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.26.0",
        "tqdm>=4.61.0",
        "faker>=8.0.0",
    ],
    entry_points={
        "console_scripts": [
            "pii-redact=pii_redaction.cli:main",
        ],
    },
)