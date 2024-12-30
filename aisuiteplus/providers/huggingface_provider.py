import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.function_call import generate_function_calling_schema
from loguru import logger


class HuggingfaceProvider(Provider):
    """
    HuggingFace Provider using httpx for direct API calls.
    Currently, this provider support calls to HF serverless Inference Endpoints
    which uses Text Generation Inference (TGI) as the backend.
    TGI is OpenAI protocol compliant.
    https://huggingface.co/inference-endpoints/
    """

    def __init__(self, **config):
        """
        Initialize the provider with the given configuration.
        The token is fetched from the config or environment variables.
        """
        # Ensure API key is provided either in config or via environment variable
        self.token = config.get("token") or os.getenv("HF_TOKEN")
        if not self.token:
            raise ValueError(
                "Hugging Face token is missing. Please provide it in the config or set the HF_TOKEN environment variable."
            )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 30)

    def chat_completions_create(
        self, model, messages, tools=None, tool_choice=None, **kwargs
    ):
        """
        Makes a request to the chat completions endpoint using httpx.
        Supports function calling through tools parameter.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        if tools:
            # Convert tools to HuggingFace's expected format
            tools_with_schema = [
                {"type": "function", "function": generate_function_calling_schema(tool)}
                for tool in tools
            ]
            data = {
                "model": model,
                "messages": messages,
                "tools": tools_with_schema,
                "tool_choice": tool_choice or "auto",
                **kwargs,
            }
        else:
            data = {
                "model": model,
                "messages": messages,
                **kwargs,
            }

        url = f"https://api-inference.huggingface.co/models/{model}/v1/chat/completions"
        try:
            logger.info(f"Request data: {data}")
            logger.info(f"Request url: {url}")
            logger.info(f"Request headers: {headers}")
            response = httpx.post(url, json=data, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            response_data = response.json()

            if tools and "tool_calls" in response_data["choices"][0]["message"]:
                return self._process_tool_calls(
                    model, messages, response_data, tools, **kwargs
                )

            return self._normalize_response(response_data)
        except httpx.HTTPStatusError as http_err:
            logger.error(
                "Function calling may not be supported for this model. Please drop tools/tool_choice kwargs and try again"
            )
            raise LLMError(f"Hugging Face request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

    def _process_tool_calls(self, model, messages, response_data, tools, **kwargs):
        """Process tool calls and return final response"""
        message = response_data["choices"][0]["message"]
        messages.append(message)

        for tool_call in message.get("tool_calls", []):
            tool_name = tool_call["function"]["name"]
            arguments = tool_call["function"]["arguments"]
            tool_id = tool_call.get("id", "call_1")

            try:
                tool_response = self.execute_tool(tool_name, arguments, tools)
                messages.append(
                    self.build_tool_result_message(tool_response, tool_id, tool_name)
                )
            except Exception as e:
                raise LLMError(f"Tool execution failed: {e}")

        # Get final response after tool execution
        final_data = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        try:
            url = f"https://api-inference.huggingface.co/models/{model}"
            final_response = httpx.post(
                url,
                json=final_data,
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout,
            )
            final_response.raise_for_status()
            return self._normalize_response(final_response.json())
        except Exception as e:
            raise LLMError(f"Final response failed: {e}")

    def _normalize_response(self, response_data):
        """
        Normalize the response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["choices"][0][
            "message"
        ]["content"]
        return normalized_response
