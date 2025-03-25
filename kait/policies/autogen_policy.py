"""AutoGen LLM strategy implementation."""
from typing import Any, Dict, List

import autogen
from autogen.agentchat import Agent

from kait.policies.base import LLMStrategy


class AutoGenStrategy(LLMStrategy):
    """Strategy for using AutoGen as the LLM provider."""

    def create_agent(self, **kwargs) -> Agent:
        """Create an AutoGen agent.

        Args:
            **kwargs: Arguments to pass to the agent constructor.

        Returns:
            Agent: The created AutoGen agent instance.
        """
        return autogen.AssistantAgent(**kwargs)

    def get_config_list(self) -> List[Dict[str, Any]]:
        """Get the configuration list for AutoGen.

        Returns:
            List[Dict[str, Any]]: The configuration list for AutoGen.
        """
        return autogen.config_list_from_json(env_or_file="KAIT_OPENAI_KEY") 