import os
import httpx
import json
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.function_call import generate_function_calling_schema
from loguru import logger


class OllamaProvider(Provider):
    """
    Ollama Provider that makes HTTP calls instead of using SDK.
    It uses the /api/chat endpoint.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _CONNECT_ERROR_MESSAGE = "Ollama is likely not running. Start Ollama by running `ollama serve` on your host."

    def __init__(self, **config):
        """
        Initialize the Ollama provider with the given configuration.
        """
        self.url = config.get("api_url") or os.getenv(
            "OLLAMA_API_URL", "http://localhost:11434"
        )
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(
        self, model, messages, tools=None, tool_choice=None, **kwargs
    ):
        """
        Makes a request to the chat completions endpoint using httpx.
        Supports function calling through tools parameter.
        """
        kwargs["stream"] = False

        if tools:
            logger.info("Generating function calling schema for tools")
            tools_with_schema = [
                generate_function_calling_schema(tool) for tool in tools
            ]

            # Add tool descriptions to system message
            system_message = {
                "role": "system",
                "content": "You are a helpful assistant that can use tools. Available tools:\n",
            }

            for tool_schema in tools_with_schema:
                system_message[
                    "content"
                ] += f"\n{tool_schema['name']}: {tool_schema['description']}\n"
                system_message[
                    "content"
                ] += f"Parameters: {json.dumps(tool_schema['input_schema']['properties'], indent=2)}\n"

            messages = [system_message] + messages

            response = self._make_request(model, messages, **kwargs)
            return self._process_tool_calls(response, messages, tools, model, **kwargs)

        return self._make_request(model, messages, **kwargs)

    def _make_request(self, model, messages, **kwargs):
        """Make HTTP request to Ollama API"""
        data = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        try:
            response = httpx.post(
                self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except httpx.ConnectError:
            raise LLMError(f"Connection failed: {self._CONNECT_ERROR_MESSAGE}")
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Ollama request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    def _process_tool_calls(self, response, messages, tools, model, **kwargs):
        """Process potential function calls in the response"""
        content = response["message"]["content"]

        # Look for function call syntax in the response
        if "I want to use the tool" in content.lower():
            try:
                # Extract tool name and arguments using simple parsing
                # This is a basic implementation - could be improved with better parsing
                lines = content.split("\n")
                for line in lines:
                    if "tool:" in line.lower():
                        tool_name = line.split("tool:")[1].strip()
                    if "arguments:" in line.lower():
                        arguments = line.split("arguments:")[1].strip()
                        try:
                            arguments = json.loads(arguments)
                        except:
                            continue

                        tool_id = "tool_call_1"  # Simple ID for tracking

                        # Execute the tool
                        tool_response = self.execute_tool(tool_name, arguments, tools)

                        # Add results to messages
                        messages.append({"role": "assistant", "content": content})
                        messages.append(
                            self.build_tool_result_message(
                                tool_response, tool_id, tool_name
                            )
                        )

                        # Get final response
                        final_response = self._make_request(model, messages, **kwargs)
                        return self._normalize_response(final_response)

            except Exception as e:
                logger.error(f"Error processing tool call: {e}")

        return self._normalize_response(response)

    def _normalize_response(self, response_data):
        """
        Normalize the API response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["message"][
            "content"
        ]
        return normalized_response
