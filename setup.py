from setuptools import setup, find_packages

setup(
    name="novel-writer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.1.7",
        "pydantic>=2.5.3",
        "pyyaml>=6.0.1",
        "rich>=13.7.0",
        "loguru>=0.7.2",
        "pdfplumber>=0.10.3",
        "pandas>=2.2.0",
        "tqdm>=4.66.1",
    ],
    entry_points={
        "console_scripts": [
            "novel-writer=novel_writer.cli:main",
        ],
    },
    python_requires=">=3.8",
)
