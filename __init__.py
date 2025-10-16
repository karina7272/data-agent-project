"""Top‑level package for the data agent project.

This package exposes the main classes used to build an interactive data
analysis assistant.  The agent is capable of loading a tabular dataset,
inferring its schema, handling missing values, performing exploratory
analysis and basic machine learning, and orchestrating calls to large
language models (LLMs) to interpret natural‑language questions.  See
README.md in the project root for installation and usage instructions.
"""

from .data_handler import DataHandler
from .agent import ChatAgent

__all__ = ["DataHandler", "ChatAgent"]