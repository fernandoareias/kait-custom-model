"""Base LLM strategy implementation."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from autogen.agentchat import Agent


class LLMStrategy(ABC):
    """Abstract base class for LLM strategies."""

    @abstractmethod
    def create_agent(self, **kwargs) -> Agent:
        """Create an agent using the specific LLM strategy.

        Args:
            **kwargs: Arguments to pass to the agent constructor.

        Returns:
            Agent: The created agent instance.
        """
        pass

    @abstractmethod
    def get_config_list(self) -> List[Dict[str, Any]]:
        """Get the configuration list for the LLM provider.

        Returns:
            List[Dict[str, Any]]: The configuration list for the LLM provider.
        """
        pass 