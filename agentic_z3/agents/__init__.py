"""
Agent module containing the three core agents:
- Architect: Hierarchical planning and curriculum generation
- Worker: Type-aware coding with TTRL soft reset
- Coach: Diagnosis and skill crystallization
"""

from .base_agent import BaseAgent
from .architect import Architect
from .worker import Worker
from .coach import Coach

__all__ = ["BaseAgent", "Architect", "Worker", "Coach"]









