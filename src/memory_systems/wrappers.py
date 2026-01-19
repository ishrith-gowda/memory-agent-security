"""
memory system wrappers for external memory libraries.

this module provides wrapper classes that adapt external memory system
implementations (mem0, amem, memgpt) to the common MemorySystem protocol.
all comments are lowercase.
"""

import sys
from typing import Any, Dict, List, Optional

from memory_systems.base import MemorySystem
from utils.logging import logger


class Mem0Wrapper(MemorySystem):
    """
    wrapper for Mem0 memory system.

    adapts the external Mem0 library to the MemorySystem protocol,
    providing consistent interface for storing, retrieving, and searching
    memory content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize the Mem0 wrapper.

        args:
            config: configuration for Mem0 system
        """
        self.config = config or {}
        self.client = None
        self.user_id = self.config.get("user_id", "default_user")
        self.logger = logger

        try:
            # Import and initialize Mem0 client
            sys.path.insert(0, "external/mem0")
            from mem0 import Memory

            self.client = Memory.from_config(self.config)
            self.logger.logger.info("Mem0 wrapper initialized successfully")

        except ImportError as e:
            self.logger.logger.error(f"failed to import Mem0: {e}")
            raise
        except Exception as e:
            self.logger.logger.error(f"failed to initialize Mem0 client: {e}")
            raise

    def store(self, key: str, value: Any) -> None:
        """
        store a key-value pair in Mem0 memory.

        args:
            key: memory key
            value: memory value
        """
        try:
            content = f"Remember: {key} = {value}"
            messages = [{"role": "user", "content": content}]
            self.client.add(messages, user_id=self.user_id)
        except Exception as e:
            self.logger.log_error("mem0_store", e, {"key": key})
            raise

    def retrieve(self, key: str) -> Optional[Any]:
        """
        retrieve a value by key from Mem0 memory.

        args:
            key: memory key to retrieve

        returns:
            retrieved value or None if not found
        """
        try:
            memories = self.client.get_all(user_id=self.user_id)
            for memory in memories:
                if key in memory.get("memory", ""):
                    return memory.get("memory")
            return None
        except Exception as e:
            self.logger.log_error("mem0_retrieve", e, {"key": key})
            raise

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        search Mem0 memory for relevant information.

        args:
            query: search query

        returns:
            list of relevant memory entries
        """
        try:
            memories = self.client.search(query, user_id=self.user_id)
            return [{"memory": mem, "score": 1.0} for mem in memories]
        except Exception as e:
            self.logger.log_error("mem0_search", e, {"query": query})
            raise

    def get_all_keys(self) -> List[str]:
        """
        get all keys currently stored in Mem0 memory.

        returns:
            list of memory keys
        """
        try:
            memories = self.client.get_all(user_id=self.user_id)
            keys = []
            for memory in memories:
                # Extract keys from memory content (simplified)
                content = memory.get("memory", "")
                if "=" in content:
                    key = content.split("=")[0].strip()
                    keys.append(key)
            return keys
        except Exception as e:
            self.logger.log_error("mem0_get_keys", e)
            raise


class AMEMWrapper(MemorySystem):
    """
    wrapper for A-MEM (Agentic Memory) system.

    adapts the external A-MEM library to the MemorySystem protocol,
    providing consistent interface for agentic memory operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize the A-MEM wrapper.

        args:
            config: configuration for A-MEM system
        """
        self.config = config or {}
        self.client = None
        self.logger = logger

        try:
            # Import and initialize A-MEM client
            sys.path.insert(0, "external/amem")
            from agentic_memory import AgenticMemorySystem

            self.client = AgenticMemorySystem(**self.config)
            self.logger.logger.info("A-MEM wrapper initialized successfully")

        except ImportError as e:
            self.logger.logger.error(f"failed to import A-MEM: {e}")
            raise
        except Exception as e:
            self.logger.logger.error(f"failed to initialize A-MEM client: {e}")
            raise

    def store(self, key: str, value: Any) -> None:
        """
        store a key-value pair in A-MEM memory.

        args:
            key: memory key
            value: memory value
        """
        try:
            self.client.store_memory(key, str(value))
        except Exception as e:
            self.logger.log_error("amem_store", e, {"key": key})
            raise

    def retrieve(self, key: str) -> Optional[Any]:
        """
        retrieve a value by key from A-MEM memory.

        args:
            key: memory key to retrieve

        returns:
            retrieved value or None if not found
        """
        try:
            return self.client.retrieve_memory(key)
        except Exception as e:
            self.logger.log_error("amem_retrieve", e, {"key": key})
            raise

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        search A-MEM memory for relevant information.

        args:
            query: search query

        returns:
            list of relevant memory entries
        """
        try:
            results = self.client.search_memory(query)
            return [{"memory": result, "score": 1.0} for result in results]
        except Exception as e:
            self.logger.log_error("amem_search", e, {"query": query})
            raise

    def get_all_keys(self) -> List[str]:
        """
        get all keys currently stored in A-MEM memory.

        returns:
            list of memory keys
        """
        try:
            return self.client.get_all_memory_keys()
        except Exception as e:
            self.logger.log_error("amem_get_keys", e)
            raise


class MemGPTWrapper(MemorySystem):
    """
    wrapper for MemGPT memory system.

    adapts the external MemGPT/Letta library to the MemorySystem protocol,
    providing consistent interface for LLM-powered memory operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        initialize the MemGPT wrapper.

        args:
            config: configuration for MemGPT system
        """
        self.config = config or {}
        self.client = None
        self.agent_id = self.config.get("agent_id")
        self.logger = logger

        try:
            # Import and initialize MemGPT client
            sys.path.insert(0, "external/memgpt")
            from letta import create_client

            self.client = create_client(**self.config)

            # Create or get agent
            if not self.agent_id:
                agent = self.client.agents.create(
                    name="memory_agent_security_agent",
                    memory_blocks=[],
                    model="gpt-4",
                    embedding="text-embedding-ada-002",
                )
                self.agent_id = agent.id

            self.logger.logger.info("MemGPT wrapper initialized successfully")

        except ImportError as e:
            error_msg = f"failed to import MemGPT: {e}"
            self.logger.logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"failed to initialize MemGPT client: {e}"
            self.logger.logger.error(error_msg)
            raise

    def store(self, key: str, value: Any) -> None:
        """
        store a key-value pair in MemGPT memory.

        args:
            key: memory key
            value: memory value
        """
        try:
            content = f"Please remember that {key} is {value}"
            message = {"role": "user", "content": content}
            self.client.agents.messages.create(
                agent_id=self.agent_id, messages=[message]
            )
        except Exception as e:
            self.logger.log_error("memgpt_store", e, {"key": key})
            raise

    def retrieve(self, key: str) -> Optional[Any]:
        """
        retrieve a value by key from MemGPT memory.

        args:
            key: memory key to retrieve

        returns:
            retrieved value or None if not found
        """
        try:
            # Use agent to recall information
            content = f"What is the value of {key}?"
            query = {"role": "user", "content": content}
            response = self.client.agents.messages.create(
                agent_id=self.agent_id, messages=[query]
            )
            # Extract value from response (simplified)
            return response.messages[-1].content if response.messages else None
        except Exception as e:
            self.logger.log_error("memgpt_retrieve", e, {"key": key})
            raise

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        search MemGPT memory for relevant information.

        args:
            query: search query

        returns:
            list of relevant memory entries
        """
        try:
            message = {"role": "user", "content": query}
            response = self.client.agents.messages.create(
                agent_id=self.agent_id, messages=[message]
            )
            return [{"memory": response.messages[-1].content, "score": 1.0}]
        except Exception as e:
            self.logger.log_error("memgpt_search", e, {"query": query})
            raise

    def get_all_keys(self) -> List[str]:
        """
        get all keys currently stored in MemGPT memory.

        returns:
            list of memory keys
        """
        try:
            # Simplified implementation - would query agent's memory blocks
            memories = self.client.agents.core_memory.retrieve(agent_id=self.agent_id)
            keys = []
            for memory in memories:
                # Extract keys from memory content
                content = memory.content
                if "=" in content:
                    key = content.split("=")[0].strip()
                    keys.append(key)
            return keys
        except Exception as e:
            self.logger.log_error("memgpt_get_keys", e)
            raise


def create_memory_system(
    system_type: str, config: Optional[Dict[str, Any]] = None
) -> MemorySystem:
    """
    factory function to create memory system wrappers.

    args:
        system_type: type of memory system ("mem0", "amem", "memgpt")
        config: configuration for the memory system

    returns:
        initialized memory system wrapper

    raises:
        ValueError: if system_type is not supported
    """
    system_type = system_type.lower()

    if system_type == "mem0":
        return Mem0Wrapper(config)
    elif system_type == "amem":
        return AMEMWrapper(config)
    elif system_type == "memgpt":
        return MemGPTWrapper(config)
    else:
        raise ValueError(f"unsupported memory system type: {system_type}")
