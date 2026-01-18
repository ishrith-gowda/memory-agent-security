"""
base interfaces for memory agent security defenses.

this module defines abstract base classes for implementing defenses against
memory system attacks. defenses protect memory systems from adversarial
attacks through various mechanisms like watermarking, validation, or filtering.
all comments are lowercase.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..memory_systems.base import MemorySystem
from ..utils.config import config_manager
from ..utils.logging import logger

if TYPE_CHECKING:
    from ..attacks.base import Attack


class Defense(ABC):
    """
    abstract base class for all memory system defenses.

    defenses implement protective measures against attacks, such as watermarking,
    validation, or filtering mechanisms. all defenses must implement the
    activate method and provide metadata about their protection capabilities.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        initialize the defense.

        args:
            name: unique name identifier for the defense
            config: defense-specific configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.logger = logger
        self.is_active = False

        # load defense configuration if available
        try:
            defense_config = config_manager.load_config(f"defenses/{name}.yaml")
            self.config.update(defense_config)
        except FileNotFoundError:
            self.logger.logger.warning(f"no config found for defense {name}")

    @property
    @abstractmethod
    def defense_type(self) -> str:
        """
        the type of defense (e.g., 'watermarking', 'validation', 'filtering').
        """
        pass

    @property
    @abstractmethod
    def protected_attacks(self) -> List[str]:
        """
        list of attack types this defense can protect against.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        human-readable description of what the defense does.
        """
        pass

    @abstractmethod
    def activate(self, memory_system: MemorySystem) -> bool:
        """
        activate the defense on the given memory system.

        args:
            memory_system: the memory system to protect

        returns:
            true if activation successful, false otherwise
        """
        pass

    @abstractmethod
    def deactivate(self) -> bool:
        """
        deactivate the defense.

        returns:
            true if deactivation successful, false otherwise
        """
        pass

    @abstractmethod
    def detect_attack(self, operation: str, **kwargs) -> Dict[str, Any]:
        """
        detect if an attack is occurring during a memory operation.

        args:
            operation: the memory operation being performed
            **kwargs: operation-specific parameters

        returns:
            dictionary with detection results
        """
        pass

    def validate_compatibility(self, attack: "Attack") -> bool:
        """
        validate that the defense can protect against the given attack.

        args:
            attack: the attack to validate against

        returns:
            true if defense can protect against attack, false otherwise
        """
        return attack.attack_type in self.protected_attacks

    def get_metadata(self) -> Dict[str, Any]:
        """
        get metadata about the defense.

        returns:
            dictionary with defense metadata
        """
        return {
            "name": self.name,
            "type": self.defense_type,
            "protected_attacks": self.protected_attacks,
            "description": self.description,
            "is_active": self.is_active,
            "config": self.config
        }