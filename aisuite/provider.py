from abc import ABC, abstractmethod
from pathlib import Path
import importlib
import os
import functools


class LLMError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message):
        super().__init__(message)


class Provider(ABC):
    @abstractmethod
    def chat_completions_create(
        self, model, messages, tools=None, tool_choice=None, **kwargs
    ):
        """Abstract method for chat completion calls, to be implemented by each provider."""
        pass

    def execute_tool(self, tool_name: str, arguments: dict, tools: list) -> any:
        """Execute a tool call - common implementation for all providers"""
        matching_tool = next(
            (tool for tool in tools if tool.__name__ == tool_name), None
        )

        if matching_tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in provided tools")

        try:
            import json

            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            return matching_tool(**arguments)
        except Exception as e:
            raise Exception(f"Error executing tool: {str(e)}")

    def process_tool_calls(self, response, messages, tools, **kwargs):
        """Process tool calls - to be implemented by providers that support function calling"""
        raise NotImplementedError("This provider does not support function calling")

    def build_tool_result_message(self, tool_response, tool_id, tool_name):
        """Build a tool result message - can be overridden by providers"""
        return {
            "role": "tool",
            "content": str(tool_response),
            "tool_call_id": tool_id,
            "name": tool_name,
        }


class ProviderFactory:
    """Factory to dynamically load provider instances based on naming conventions."""

    PROVIDERS_DIR = Path(__file__).parent / "providers"
    DEFAULT_PROVIDER = "openai"

    @classmethod
    def create_provider(cls, provider_key=None, config=None):
        """Dynamically load and create an instance of a provider based on the naming convention."""
        config = config or {}

        # If no provider specified, use default
        if not provider_key:
            provider_key = cls.DEFAULT_PROVIDER
            if "api_key" not in config and not os.getenv("OPENAI_API_KEY"):
                raise ValueError(
                    "OpenAI API key is required. Please provide it in config or set OPENAI_API_KEY environment variable."
                )

        # Convert provider_key to the expected module and class names
        provider_class_name = f"{provider_key.capitalize()}Provider"
        provider_module_name = f"{provider_key}_provider"

        module_path = f"aisuite.providers.{provider_module_name}"

        # Lazily load the module
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module {module_path}: {str(e)}. Please ensure the provider is supported by doing ProviderFactory.get_supported_providers()"
            )

        # Instantiate the provider class
        provider_class = getattr(module, provider_class_name)
        return provider_class(**config)

    @classmethod
    @functools.cache
    def get_supported_providers(cls):
        """List all supported provider names based on files present in the providers directory."""
        provider_files = Path(cls.PROVIDERS_DIR).glob("*_provider.py")
        return {file.stem.replace("_provider", "") for file in provider_files}
