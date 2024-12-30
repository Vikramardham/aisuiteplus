# %%
import time

import aisuite as ai
from dotenv import load_dotenv

load_dotenv(".env")

client = ai.Client()

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that can calculate the sum of two numbers.",
    },
    {"role": "user", "content": "What is the sum of 23 and 13332?"},
]


def add_two_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


def get_weather(location: str) -> str:
    return f"Weather in {location} is amazing"


# %%

# model = os.getenv("MODEL")
# model = "openai:gpt-4o-mini"
# model = "ollama:llama3.2"
model = "anthropic:claude-3-5-sonnet-20241022"
# model = "huggingface:Qwen/QwQ-32B-Preview"
start_time = time.time()
response = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=[add_two_numbers],
    tool_choice="auto",
)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
print(response.choices[0].message.content)

# %%
