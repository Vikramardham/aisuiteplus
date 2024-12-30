import os
import httpx
import json
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.function_call import generate_function_calling_schema_for_openai
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
                generate_function_calling_schema_for_openai(tool) for tool in tools
            ]
            kwargs["tool_choice"] = "auto"
            kwargs["stream"] = False

            logger.info("making initial request")
            return self._process_tool_calls(
                messages, tools_with_schema, tools, model, **kwargs
            )

        return self._normalize_response(self._make_request(model, messages, **kwargs))

    def _make_request(self, model, messages, tools=None, **kwargs):
        """Make HTTP request to Ollama API"""
        data = {
            "model": model,
            "messages": messages,
            "tools": tools,
            **kwargs,
        }

        logger.info(f"{json.dumps(data, indent=2)}")

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

    def _process_tool_calls(self, messages, tools_with_schema, tools, model, **kwargs):
        """Process potential function calls in the response"""
        response = self._make_request(
            model, messages, tools=tools_with_schema, **kwargs
        )
        _content = response["message"]["content"]
        logger.info("Making tool calls")

        logger.info(f"{json.dumps(response, indent=2)}")
        # Look for function call syntax in the response
        if "tool_calls" in response["message"]:
            logger.info("Tool calls found")
            try:
                # Extract tool name and arguments using simple parsing
                # This is a basic implementation - could be improved with better parsing
                for tool_call in response["message"]["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    arguments = tool_call["function"]["arguments"]
                    tool_id = None  # TODO: Add tool id

                    tool_response = self.execute_tool(tool_name, arguments, tools)

                    # Add results to messages
                    # messages.append({"role": "assistant", "content": content})
                    messages.append(
                        self.build_tool_result_message(
                            tool_response, tool_id, tool_name
                        )
                    )

                    logger.info(f"{json.dumps(messages, indent=2)}")
                    logger.info(f"tool_response: {tool_response}")
                    logger.info("Making final request")
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
