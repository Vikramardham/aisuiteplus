"""Tests for Anthropic provider's function calling capabilities using DeepEval framework."""

import pytest
import os
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
import aisuite as ai
from dotenv import load_dotenv

from .functions import (
    calculate_age,
    convert_temperature,
    analyze_text,
    manage_shopping_cart,
    process_financial_data,
    search_web,
    execute_api_call,
    execute_sql_query,
    search_course_catalog,
    get_course_enrollment_status,
    query_student_progress,
)

MODEL = "anthropic:claude-3-sonnet-20240229"
client = ai.Client()

load_dotenv()


@pytest.fixture(autouse=True)
def set_api_key_env_var(monkeypatch):
    """Fixture to set environment variables for tests."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY"))
    monkeypatch.setenv(
        "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")
    )  # Required for DeepEval metrics


def test_simple_direct_response():
    """Test when model should respond directly without function call."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math assistant. Provide clear and concise answers.",
        },
        {"role": "user", "content": "What is 2 plus 2?"},
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[calculate_age],  # Function available but shouldn't be used
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input="What is 2 plus 2?",
        actual_output=response.choices[0].message.content,
        expected_output="4",
        retrieval_context=["Basic arithmetic: 2 + 2 = 4"],
    )

    assert response.tool_calls is None
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)


def test_simple_function_call():
    """Test simple function calling with calculate_age."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can calculate ages. Use the calculate_age function when appropriate.",
        },
        {"role": "user", "content": "How old is someone born in 1990?"},
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[calculate_age],
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input="How old is someone born in 1990?",
        actual_output=response.choices[0].message.content,
        expected_output="A person born in 1990 is 34 years old.",
    )

    print(response.choices[0].message.content)
    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)


def test_medium_function_call():
    """Test medium complexity function calling with convert_temperature."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can convert temperatures. Use the convert_temperature function when needed.",
        },
        {"role": "user", "content": "Convert 25 degrees Celsius to Fahrenheit"},
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[convert_temperature],
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input="Convert 25 degrees Celsius to Fahrenheit",
        actual_output=response.choices[0].message.content,
        expected_output="25 degrees Celsius is equal to 77 degrees Fahrenheit",
        retrieval_context=[
            "Temperature conversion formula: °F = (°C × 9/5) + 32",
            "25°C × (9/5) + 32 = 77°F",
        ],
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)


def test_complex_function_call():
    """Test complex function calling with analyze_text."""
    sample_text = "The quick brown fox jumps over the lazy dog"
    messages = [
        {
            "role": "system",
            "content": "You are a text analysis assistant. Use the analyze_text function to provide detailed text statistics.",
        },
        {
            "role": "user",
            "content": f"Analyze this text: {sample_text}",
        },
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[analyze_text],
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input=f"Analyze this text: {sample_text}",
        actual_output=response.choices[0].message.content,
        expected_output="The text contains 9 words, all unique, with an average word length of 3.89 characters.",
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)


def test_advanced_function_call():
    """Test advanced function calling with shopping cart management."""
    messages = [
        {
            "role": "system",
            "content": "You are a shopping assistant that can manage a cart. Use the manage_shopping_cart function to handle cart operations.",
        },
        {
            "role": "user",
            "content": "Add a laptop ($999.99) and 2 books ($29.99 each) to my cart and calculate the total",
        },
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[manage_shopping_cart],
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input="Add a laptop ($999.99) and 2 books ($29.99 each) to my cart and calculate the total",
        actual_output=response.choices[0].message.content,
        expected_output="Your cart total is $1059.97 (1 laptop at $999.99 and 2 books at $29.99 each)",
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)


def test_complex_multi_function_scenario():
    """Test complex scenario requiring multiple function calls with financial data."""
    messages = [
        {
            "role": "system",
            "content": "You are a financial analysis assistant. Use the process_financial_data function to analyze transactions.",
        },
        {
            "role": "user",
            "content": """Analyze these transactions:
        - 2024-01-15: Grocery shopping $150.50
        - 2024-01-20: Restaurant $75.25
        - 2024-02-01: Grocery shopping $200.75
        First show me the total spent, then break it down by category.""",
        },
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[process_financial_data],
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input=messages[1]["content"],
        actual_output=response.choices[0].message.content,
        expected_output="""The total spent is $426.50. Here's the breakdown by category:
- Groceries: $351.25 (two purchases)
- Dining: $75.25 (one purchase)""",
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)


def test_multiple_tool_selection():
    """Test when model has multiple tools available and needs to select the right one."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can perform various tasks.",
        },
        {
            "role": "user",
            "content": "What's the temperature in Fahrenheit if it's 30 degrees Celsius?",
        },
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[
            calculate_age,
            convert_temperature,
            analyze_text,
        ],  # Multiple tools available
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input="What's the temperature in Fahrenheit if it's 30 degrees Celsius?",
        actual_output=response.choices[0].message.content,
        expected_output="30 degrees Celsius is equal to 86 degrees Fahrenheit",
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)
    assert (
        "convert_temperature" in response.tool_calls
    )  # Verify correct tool was chosen


def test_multi_function_calculation():
    """Test scenario requiring multiple function calls to compute final result."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can perform calculations.",
        },
        {
            "role": "user",
            "content": "Someone born in 1990 went shopping. They bought 2 items at $25.50 each and 1 item at $15.75. Calculate their age and total shopping cost.",
        },
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[calculate_age, manage_shopping_cart],
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input=messages[1]["content"],
        actual_output=response.choices[0].message.content,
        expected_output="The person is 34 years old and their shopping total is $66.75",
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        FaithfulnessMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)
    assert len(response.tool_calls) >= 2  # Verify multiple tools were used


def test_realistic_functions():
    """Test realistic scenarios with course catalog search, enrollment status, and progress tracking."""
    messages = [
        {
            "role": "system",
            "content": "You are an educational platform assistant that can help students find courses, check enrollment status, and track progress.",
        },
        {
            "role": "user",
            "content": "I'm student-123. Find Python programming courses, check my enrollment status in python-101, and show my progress in all courses since January 2024.",
        },
    ]

    response = client.chat.completions.create(
        messages=messages,
        model=MODEL,
        tools=[
            search_course_catalog,
            get_course_enrollment_status,
            query_student_progress,
        ],
        tool_choice="auto",
    )

    test_case = LLMTestCase(
        input=messages[1]["content"],
        actual_output=response.choices[0].message.content,
        expected_output="""I found a Python programming course that might interest you. You're currently enrolled in Python Programming Fundamentals (python-101) and have completed 65% of the course. You're also taking Full Stack Web Development with 35% progress. Your Python course grades are strong with 88.5 in assignments, 92.0 in quizzes, and 85.0 in the project.""",
        retrieval_context=[
            "Course catalog search results",
            "Enrollment status data",
            "Student progress records",
        ],
    )

    metrics = [
        AnswerRelevancyMetric(threshold=0.7),
        ContextualRelevancyMetric(threshold=0.7),
    ]
    assert_test(test_case, metrics)
    assert len(response.tool_calls) >= 3  # Verify all three tools were used
