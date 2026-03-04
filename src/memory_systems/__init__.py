"""
memory systems package for memory agent security research.

exports the core memory system implementations including the faiss-backed
vector memory system for realistic retrieval-based evaluation.

all comments are lowercase.
"""

from memory_systems.base import MemorySystem
from memory_systems.wrappers import MockMemorySystem, create_memory_system

__all__ = [
    "MemorySystem",
    "MockMemorySystem",
    "create_memory_system",
]
