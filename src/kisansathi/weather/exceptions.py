"""Exceptions for the weather tool."""


class WeatherError(Exception):
    """Base exception for weather operations."""


class InvalidCoordinatesError(WeatherError):
    """Raised when latitude/longitude are outside valid ranges."""


class WeatherRequestError(WeatherError):
    """Raised when the HTTP request to the weather API fails."""


class WeatherTimeoutError(WeatherError):
    """Raised when the weather API request times out."""


class WeatherParseError(WeatherError):
    """Raised when the weather API response is malformed."""


class WeatherUnavailableError(WeatherError):
    """Raised when the weather API returns incomplete or unavailable data."""
