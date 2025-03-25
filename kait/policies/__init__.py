"""LLM strategy implementations."""
from typing import Any, Dict, List, Optional

from kait.policies.autogen_policy import AutoGenStrategy
from kait.policies.base import LLMStrategy
from kait.policies.openai_policy import OpenAIStrategy


def get_llm_strategy(policy: Optional[str] = None, **kwargs) -> LLMStrategy:
    """Get the appropriate LLM strategy based on the policy.
    
    Args:
        policy (Optional[str], optional): The policy to use. Defaults to None.
        **kwargs: Additional arguments to pass to the strategy constructor.
        
    Returns:
        LLMStrategy: The appropriate LLM strategy instance.
        
    Raises:
        ValueError: If an invalid policy is provided.
    """
    print(f"[+] - Using policy: {policy}")
    if policy is None or policy.lower() == "autogen":
        return AutoGenStrategy()
    elif policy.lower() == "openai":
        if "api_key" not in kwargs:
            raise ValueError("api_key is required for OpenAI strategy")
        return OpenAIStrategy(**kwargs)
    else:
        raise ValueError(f"Invalid policy: {policy}")


__all__ = ["LLMStrategy", "AutoGenStrategy", "OpenAIStrategy", "get_llm_strategy"] 