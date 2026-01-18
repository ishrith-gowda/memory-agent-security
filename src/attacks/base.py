"""
base interfaces for memory agent security attacks and defenses.

this module defines abstract base classes for implementing attacks against
memory systems and defenses that protect against those attacks. all attacks
and defenses must inherit from these base classes to ensure consistent
interfaces and proper integration with the research framework.
all comments are lowercase.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol

from ..defenses.base import Defense
from ..memory_systems.base import MemorySystem
from ..utils.config import config_manager
from ..utils.logging import logger


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
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """search memory for relevant information."""
        ...
    
    def get_all_keys(self) -> List[str]:
        """get all keys currently stored in memory."""
        ...


class Attack(ABC):
    """
    abstract base class for all memory system attacks.
    
    attacks implement specific adversarial strategies against memory systems,
    such as poisoning, injection, or manipulation attacks. all attacks must
    implement the execute method and provide metadata about their capabilities.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        initialize the attack.
        
        args:
            name: unique name identifier for the attack
            config: attack-specific configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.logger = logger
        
        # load attack configuration if available
        try:
            attack_config = config_manager.load_config(f"attacks/{name}.yaml")
            self.config.update(attack_config)
        except FileNotFoundError:
            self.logger.logger.warning(f"no config found for attack {name}")
    
    @property
    @abstractmethod
    def attack_type(self) -> str:
        """
        the type of attack (e.g., 'poisoning', 'injection', 'manipulation').
        """
        pass
    
    @property
    @abstractmethod
    def target_systems(self) -> List[str]:
        """
        list of memory systems this attack can target.
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        human-readable description of what the attack does.
        """
        pass
    
    @abstractmethod
    def execute(self, memory_system: MemorySystem) -> Dict[str, Any]:
        """
        execute the attack against the given memory system.
        
        args:
            memory_system: the memory system to attack
            
        returns:
            dictionary containing attack results and metadata
        """
        pass
    
    def validate_target(self, memory_system: MemorySystem) -> bool:
        """
        validate that the attack can be executed against the target system.
        
        args:
            memory_system: the memory system to validate
            
        returns:
            true if attack can be executed, false otherwise
        """
        # check if the memory system's type is supported
        system_type = type(memory_system).__name__.lower()
        return system_type in [s.lower() for s in self.target_systems]
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        get metadata about the attack.
        
        returns:
            dictionary with attack metadata
        """
        return {
            "name": self.name,
            "type": self.attack_type,
            "target_systems": self.target_systems,
            "description": self.description,
            "config": self.config
        }


class AttackDefensePair:
    """
    represents a pairing of an attack and its corresponding defense.
    
    this class manages the relationship between specific attacks and defenses,
    allowing for controlled testing and evaluation of defense effectiveness.
    """
    
    def __init__(self, attack: Attack, defense: Defense):
        """
        initialize the attack-defense pair.
        
        args:
            attack: the attack instance
            defense: the defense instance
        """
        self.attack = attack
        self.defense = defense
        self.logger = logger
        
        # validate compatibility
        if not defense.validate_compatibility(attack):
            self.logger.logger.warning(
                f"defense {defense.name} may not be compatible with "
                f"attack {attack.name}"
            )
    
    def execute_test(self, memory_system: MemorySystem) -> Dict[str, Any]:
        """
        execute a test of the attack against the defended memory system.
        
        args:
            memory_system: the memory system to test against
            
        returns:
            dictionary with test results
        """
        results = {
            "attack_name": self.attack.name,
            "defense_name": self.defense.name,
            "attack_success": False,
            "defense_effective": False,
            "details": {}
        }
        
        try:
            # activate defense
            defense_activated = self.defense.activate(memory_system)
            results["defense_activated"] = defense_activated
            
            if defense_activated:
                # execute attack
                attack_results = self.attack.execute(memory_system)
                results["attack_results"] = attack_results
                results["attack_success"] = attack_results.get("success", False)
                
                # check if defense detected/prevented the attack
                results["defense_effective"] = attack_results.get("detected", False)
            
        except Exception as e:
            self.logger.log_error(
                "attack_defense_test",
                e,
                {"attack": self.attack.name, "defense": self.defense.name}
            )
            results["error"] = str(e)
        
        return results