"""
base interfaces for memory systems.

this module defines the protocol that all memory systems must implement
to be compatible with attacks and defenses in the research framework.
all comments are lowercase.
"""

from typing import Any, List, Optional, Protocol


class MemorySystem(Protocol):
    """
    protocol defining the interface that memory systems must implement.

    this ensures that attacks and defenses can work with different memory
    system implementations (mem0, amem, memgpt) through a consistent interface.
    """

    def store(self, key: str, value: Any) -> None:
        """store a key-value pair in memory."""
        ...

    def retrieve(self, key: str) -> Optional[Any]:
        """retrieve a value by key from memory."""
        ...

    def search(self, query: str) -> List[dict[str, Any]]:
        """search memory for relevant information."""
        ...

    def get_all_keys(self) -> List[str]:
        """get all keys currently stored in memory."""
        ...