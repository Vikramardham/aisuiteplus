"""Test functions with varying complexity for Anthropic provider tests."""

from typing import List, Dict, Union, Optional, Any
from datetime import datetime


def calculate_age(birth_year: int) -> int:
    """Calculate age from birth year.
    :param birth_year: Year of birth
    """
    current_year = datetime.now().year
    return current_year - birth_year


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between Celsius and Fahrenheit.
    :param value: Temperature value to convert
    :param from_unit: Original unit (celsius/fahrenheit)
    :param to_unit: Target unit (celsius/fahrenheit)
    """
    if from_unit.lower() == to_unit.lower():
        return value

    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        return (value * 9 / 5) + 32
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        return (value - 32) * 5 / 9
    else:
        raise ValueError("Invalid temperature units. Use 'celsius' or 'fahrenheit'")


def analyze_text(text: str) -> Dict[str, Union[int, List[str], float]]:
    """Analyze text and return various statistics.
    :param text: Text to analyze
    """
    words = text.split()
    unique_words = set(words)
    word_count = len(words)
    avg_word_length = (
        sum(len(word) for word in words) / word_count if word_count > 0 else 0
    )

    return {
        "word_count": word_count,
        "unique_words": len(unique_words),
        "avg_word_length": round(avg_word_length, 2),
        "long_words": [word for word in words if len(word) > 6],
    }


def manage_shopping_cart(
    action: str,
    items: Optional[List[Dict[str, Union[str, float, int]]]] = None,
    cart: Optional[List[Dict[str, Union[str, float, int]]]] = None,
) -> Dict[str, Union[List[Dict[str, Union[str, float, int]]], float]]:
    """Manage a shopping cart with various operations.
    :param action: Action to perform (add/remove/calculate)
    :param items: Items to add/remove [{"name": str, "price": float, "quantity": int}]
    :param cart: Current cart state
    """
    if cart is None:
        cart = []

    if action == "add" and items:
        cart.extend(items)
    elif action == "remove" and items:
        for item in items:
            if item in cart:
                cart.remove(item)
    elif action == "calculate":
        total = sum(item["price"] * item["quantity"] for item in cart)
        return {"cart": cart, "total": round(total, 2)}

    return {"cart": cart, "total": 0.0}


def process_financial_data(
    transactions: List[Dict[str, Union[str, float]]],
    analysis_type: str,
    category_filter: Optional[str] = None,
    date_range: Optional[Dict[str, str]] = None,
) -> Dict[str, Union[float, Dict[str, float], List[Dict[str, Union[str, float]]]]]:
    """Process financial transactions and perform various analyses.
    :param transactions: List of transactions [{"date": str, "amount": float, "category": str}]
    :param analysis_type: Type of analysis (total/by_category/filter)
    :param category_filter: Optional category to filter by
    :param date_range: Optional date range {"start": str, "end": str}
    """

    def is_in_date_range(date_str: str) -> bool:
        if not date_range:
            return True
        date = datetime.strptime(date_str, "%Y-%m-%d")
        start = datetime.strptime(date_range["start"], "%Y-%m-%d")
        end = datetime.strptime(date_range["end"], "%Y-%m-%d")
        return start <= date <= end

    filtered_transactions = [
        t
        for t in transactions
        if (not category_filter or t["category"] == category_filter)
        and is_in_date_range(t["date"])
    ]

    if analysis_type == "total":
        return {"total": sum(t["amount"] for t in filtered_transactions)}
    elif analysis_type == "by_category":
        categories = {}
        for t in filtered_transactions:
            cat = t["category"]
            categories[cat] = categories.get(cat, 0) + t["amount"]
        return {"categories": categories}
    elif analysis_type == "filter":
        return {"transactions": filtered_transactions}
    else:
        raise ValueError("Invalid analysis type")


def search_web(query: str) -> Dict[str, str]:
    """Simulates a web search and returns the first result.
    :param query: Search query string
    """
    # Simulated response
    return {
        "title": f"Search results for: {query}",
        "content": f"This is simulated content for the search query: {query}",
        "url": f"https://example.com/search?q={query}",
    }


def execute_api_call(
    endpoint: str, method: str, params: Dict[str, str]
) -> Dict[str, Any]:
    """Simulates an API call and returns a response.
    :param endpoint: API endpoint
    :param method: HTTP method (GET, POST, etc.)
    :param params: Request parameters
    """
    # Simulated response
    return {
        "status": "success",
        "data": {
            "request": {"endpoint": endpoint, "method": method, "params": params},
            "response": "Simulated API response",
        },
    }


def execute_sql_query(query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Simulates executing a SQL query with parameters.
    :param query: SQL query string
    :param params: Query parameters
    """
    # Simulated response
    return [
        {
            "query": query,
            "params": params,
            "results": [
                {"id": 1, "value": "Sample data 1"},
                {"id": 2, "value": "Sample data 2"},
            ],
        }
    ]


def search_course_catalog(
    query: str, filters: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Search the course catalog for courses matching the query and filters.
    :param query: Search query for course title or description
    :param filters: Optional filters like {"level": "beginner/intermediate/advanced", "category": "programming/data-science/web-dev"}
    """
    # Simulated course catalog response
    courses = {
        "python-programming": {
            "title": "Python Programming Fundamentals",
            "description": "Learn Python basics, data structures, and algorithms",
            "level": "beginner",
            "category": "programming",
            "rating": 4.8,
            "enrolled": 1500,
        },
        "web-development": {
            "title": "Full Stack Web Development",
            "description": "Master HTML, CSS, JavaScript and backend development",
            "level": "intermediate",
            "category": "web-dev",
            "rating": 4.6,
            "enrolled": 1200,
        },
    }

    if query.lower() in "python programming":
        return {"results": [courses["python-programming"]]}
    elif query.lower() in "web development":
        return {"results": [courses["web-development"]]}
    return {"results": list(courses.values())}


def get_course_enrollment_status(course_id: str, student_id: str) -> Dict[str, Any]:
    """Get enrollment status and progress for a student in a course.
    :param course_id: Unique identifier for the course
    :param student_id: Unique identifier for the student
    """
    # Simulated enrollment data
    enrollments = {
        "python-101": {
            "student-123": {
                "status": "enrolled",
                "progress": 65,
                "start_date": "2024-01-15",
                "last_accessed": "2024-03-20",
                "completed_modules": ["intro", "basics", "data-types"],
                "remaining_modules": ["functions", "classes"],
            }
        }
    }

    if course_id in enrollments and student_id in enrollments[course_id]:
        return {"status": "success", "data": enrollments[course_id][student_id]}
    return {
        "status": "not_found",
        "message": "No enrollment found for this student and course",
    }


def query_student_progress(
    student_id: str, time_period: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """Query the database for a student's progress across all enrolled courses.
    :param student_id: Unique identifier for the student
    :param time_period: Optional date range {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
    """
    # Simulated student progress data
    progress_data = [
        {
            "course_id": "python-101",
            "course_name": "Python Programming Fundamentals",
            "enrollment_date": "2024-01-15",
            "progress_percentage": 65,
            "grades": {"assignments": 88.5, "quizzes": 92.0, "project": 85.0},
            "completion_status": "in_progress",
            "expected_completion": "2024-05-15",
        },
        {
            "course_id": "web-dev-201",
            "course_name": "Full Stack Web Development",
            "enrollment_date": "2024-02-01",
            "progress_percentage": 35,
            "grades": {"assignments": 90.0, "quizzes": 88.5},
            "completion_status": "in_progress",
            "expected_completion": "2024-06-30",
        },
    ]

    if time_period:
        # Filter by time period if provided
        filtered_data = [
            entry
            for entry in progress_data
            if time_period["start"] <= entry["enrollment_date"] <= time_period["end"]
        ]
        return filtered_data
    return progress_data
