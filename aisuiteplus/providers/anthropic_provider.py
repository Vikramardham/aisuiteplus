"""Anthropic provider implementation."""

import anthropic
from aisuite.provider import Provider
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.function_call import generate_function_calling_schema

from loguru import logger

DEFAULT_MAX_TOKENS = 4096


class AnthropicProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the Anthropic provider with the given configuration.
        Pass the entire configuration dictionary to the Anthropic client constructor.
        """
        self.client = anthropic.Anthropic(**config)

    def chat_completions_create(self, model, messages, tools=None, **kwargs):
        """Create a chat completion with function calling support."""
        # Extract system message if present
        system_message = None
        if messages and messages[0]["role"] == "system":
            system_message = messages[0]["content"]
            messages = messages[1:]

        if system_message is None:
            system_message = "You are a helpful assistant. You answer user questions to the best of your ability. You can use the provided tools to answer the question. If the tools are not relevant, you can ignore them."

        # Set default max tokens if not provided
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS
        if "tool_choice" in kwargs:
            kwargs["tool_choice"] = {"type": kwargs["tool_choice"]}

        if tools:
            logger.info("Generating function calling schema for tools")
            tools_with_schema = [
                generate_function_calling_schema(tool) for tool in tools
            ]
            return self._process_tool_calls(
                messages, tools, tools_with_schema, model, system_message, **kwargs
            )

        logger.info("Calling Anthropic API without tools")
        response = self.client.messages.create(
            model=model,
            messages=messages,
            system=system_message,
            **kwargs,
        )

        return self.normalize_response(response)

    def _process_tool_calls(
        self, messages, tools, tools_with_schema, model, system_message, **kwargs
    ):
        logger.info("Processing tool calls")
        response = self.client.messages.create(
            model=model,
            system=system_message,
            messages=messages,
            tools=tools_with_schema,
            **kwargs,
        )
        logger.info("Received response from Anthropic API")
        messages.append({"role": "assistant", "content": response.content})
        tool_calls = []
        tool_used = False

        for content in response.content:
            if content.type == "tool_use":
                tool_name = content.name
                arguments = content.input
                tool_id = content.id
                tool_calls.append(tool_name)

                try:
                    logger.info("Executing tool")
                    tool_response = self.execute_tool(tool_name, arguments, tools)
                    messages.append(
                        self.build_tool_result_message(
                            str(tool_response), tool_id, tool_name
                        )
                    )
                    tool_used = True
                except Exception as e:
                    logger.error(f"Error executing tool: {str(e)}")
                    messages.append(
                        self.build_tool_result_message(
                            f"Error: {str(e)}", tool_id, tool_name
                        )
                    )

        if not tool_used:
            return self.normalize_response(response)

        logger.info(f"Messages: {messages[-1]}")
        logger.info("Getting final response after tool execution")
        final_response = self.client.messages.create(
            model=model,
            system=system_message,
            messages=messages,
            tools=tools_with_schema,
            **{k: v for k, v in kwargs.items() if k != "tool_choice"},
        )

        return self.normalize_response(final_response, tool_calls)

    def build_tool_result_message(self, tool_response, tool_id, tool_name):
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": str(tool_response),
                }
            ],
        }

    def normalize_response(self, response, tool_calls=None):
        """Normalize the response from the Anthropic API to match OpenAI's response format."""
        normalized_response = ChatCompletionResponse()
        logger.info("Normalizing response")
        normalized_response.choices[0].message.content = response.content[0].text
        normalized_response.tool_calls = tool_calls
        return normalized_response
