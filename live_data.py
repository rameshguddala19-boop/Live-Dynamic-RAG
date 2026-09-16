import os
import time
import re

import requests
from dotenv import load_dotenv


# Load .env variables
load_dotenv()


# Simple in-memory cache
_CACHE = {}

# Cache validity: 60 seconds
_TTL = 60


def get_current_weather(city: str) -> dict:
    """
    Get current weather for a city
    from OpenWeatherMap.
    """

    key = os.getenv(
        "OPENWEATHER_API_KEY"
    )

    if not key:
        return {
            "error": (
                "OPENWEATHER_API_KEY not set "
                "— no live data available."
            )
        }

    now = time.time()

    # Check cache
    city_key = city.lower()

    if city_key in _CACHE:

        timestamp, data = _CACHE[city_key]

        if now - timestamp < _TTL:
            return data

    try:

        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": key,
                "units": "metric",
            },
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        result = {
            "city": data["name"],
            "temp_c": data["main"]["temp"],
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_ms": data["wind"]["speed"],
        }

    except Exception as e:

        return {
            "error": f"Live lookup failed: {e}"
        }

    # Save in cache
    _CACHE[city_key] = (
        now,
        result
    )

    return result


def extract_city(
    question: str
) -> str | None:
    """
    Fallback city extractor.

    Example:
    'weather in Hyderabad'
    -> Hyderabad
    """

    match = re.search(
        r"\bin\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)",
        question
    )

    return (
        match.group(1)
        if match
        else None
    )