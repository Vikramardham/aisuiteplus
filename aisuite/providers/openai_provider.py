import openai
import os
from aisuite.provider import Provider, LLMError
from aisuite.framework.function_call import generate_function_calling_schema_for_openai
from loguru import logger


class OpenaiProvider(Provider):
    def __init__(self, **config):
        """
        Initialize the OpenAI provider with the given configuration.
        Pass the entire configuration dictionary to the OpenAI client constructor.
        """
        # Ensure API key is provided either in config or via environment variable
        config.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        if not config["api_key"]:
            raise ValueError(
                "OpenAI API key is missing. Please provide it in the config or set the OPENAI_API_KEY environment variable."
            )

        self.client = openai.OpenAI(**config)

    def chat_completions_create(
        self, model, messages, tools=None, tool_choice=None, **kwargs
    ):
        if tools:
            logger.info("Generating function calling schema for tools")
            tools_with_schema = [
                generate_function_calling_schema_for_openai(tool) for tool in tools
            ]
            return self._process_tool_calls(
                model, messages, tools_with_schema, tool_choice, tools, **kwargs
            )
        logger.info("No tools provided, calling OpenAI API directly")
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=None,
            **kwargs,
        )

    def _process_tool_calls(
        self, model, messages, tools_with_schema, tool_choice, tools, **kwargs
    ):
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools_with_schema,
            tool_choice=tool_choice,
            **kwargs,
        )
        logger.info("OpenAI API response received. Processing tool calls.")
        messages.append(response.choices[0].message)

        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments
            tool_id = tool_call.id

            tool_response = self.execute_tool(tool_name, arguments, tools)
            logger.info(f"Tool {tool_name} executed with response: {tool_response}")
            messages.append(
                self.build_tool_result_message(tool_response, tool_id, tool_name)
            )

        logger.info("All tool calls processed. Calling OpenAI API again.")
        final_response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            tools=None,
            **kwargs,
        )
        return final_response

    def build_tool_result_message(self, tool_response, tool_id, tool_name):
        return {
            "role": "tool",
            "content": str(tool_response),
            "tool_call_id": tool_id,
            "name": tool_name,
        }

    def execute_tool(self, tool_name, arguments, tools):
        """Execute a tool call"""
        matching_tool = None
        for tool in tools:
            if tool.__name__ == tool_name:
                matching_tool = tool
                break

        if matching_tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in provided tools")

        try:
            import json

            print(json.loads(arguments))
            return matching_tool(**json.loads(arguments))
        except Exception as e:
            raise Exception(f"Error executing tool: {str(e)}")
