import unittest

from kisansathi.weather.client import OpenMeteoClient
from kisansathi.weather.exceptions import (
    WeatherParseError,
    WeatherRequestError,
    WeatherTimeoutError,
    WeatherUnavailableError,
)
from kisansathi.weather.models import WeatherRequest


class FakeTransport:
    def __init__(self, response: object = None, raise_exc: Exception | None = None) -> None:
        self.response = response
        self.raise_exc = raise_exc
        self.calls: list[tuple[str, dict, float]] = []

    def get_json(self, url: str, params: dict, timeout: float) -> object:
        self.calls.append((url, dict(params), timeout))
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


class WeatherClientTests(unittest.TestCase):
    def test_successful_response(self) -> None:
        fake_response = {
            "latitude": 28.6139,
            "longitude": 77.209,
            "timezone": "Asia/Kolkata",
            "current": {
                "temperature_2m": 25.5,
                "precipitation": 0.0,
                "rain": 0.0,
                "wind_speed_10m": 3.2,
                "wind_direction_10m": 120,
                "weather_code": 1,
                "time": "2026-10-05T12:00",
            },
        }
        transport = FakeTransport(response=fake_response)
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=28.6139, longitude=77.2090)
        resp = client.get_forecast(req)
        self.assertEqual(resp.current.temperature_c, 25.5)
        self.assertEqual(resp.current.weather_code, 1)
        self.assertEqual(resp.current.wind_speed_mps, 3.2)
        self.assertEqual(resp.latitude, 28.6139)
        self.assertEqual(len(resp.forecast), 0)
        self.assertTrue(transport.calls)

    def test_timeout_raises_weather_timeout_error(self) -> None:
        from kisansathi.weather.exceptions import WeatherTimeoutError

        transport = FakeTransport(raise_exc=TimeoutError("timeout"))
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=0, longitude=0)
        with self.assertRaises(WeatherTimeoutError):
            client.get_forecast(req)

    def test_http_failure_raises_weather_request_error(self) -> None:
        from kisansathi.weather.exceptions import WeatherRequestError

        transport = FakeTransport(raise_exc=WeatherRequestError("fail"))
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=0, longitude=0)
        with self.assertRaises(WeatherRequestError):
            client.get_forecast(req)

    def test_malformed_json_raises_weather_parse_error(self) -> None:
        # transport raising parse error directly
        from kisansathi.weather.exceptions import WeatherParseError

        transport = FakeTransport(raise_exc=WeatherParseError("malformed"))
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=0, longitude=0)
        with self.assertRaises(WeatherParseError):
            client.get_forecast(req)

    def test_malformed_response_structure_raises_weather_parse_error(self) -> None:
        transport = FakeTransport(response="not_a_dict")  # type: ignore
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=0, longitude=0)
        with self.assertRaises(WeatherParseError):
            client.get_forecast(req)

    def test_missing_current_data_raises_weather_unavailable(self) -> None:
        transport = FakeTransport(response={})
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=0, longitude=0)
        with self.assertRaises(WeatherUnavailableError):
            client.get_forecast(req)

    def test_parsed_values_mapped_correctly(self) -> None:
        fake_response = {
            "current": {
                "temperature_2m": 10.0,
                "precipitation": 2.5,
                "rain": 1.5,
                "wind_speed_10m": 5.0,
                "wind_direction_10m": 180,
                "weather_code": 61,
                "time": "2026-10-05T00:00",
            },
            "timezone": "UTC",
        }
        transport = FakeTransport(response=fake_response)
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=45, longitude=90, timezone="UTC")
        resp = client.get_forecast(req)
        self.assertEqual(resp.current.temperature_c, 10.0)
        self.assertEqual(resp.current.precipitation_mm, 2.5)
        self.assertEqual(resp.current.rain_mm, 1.5)
        self.assertEqual(resp.current.wind_speed_mps, 5.0)
        self.assertEqual(resp.current.wind_direction_deg, 180)
        self.assertEqual(resp.current.weather_code, 61)
        self.assertEqual(resp.current.time_iso, "2026-10-05T00:00")
        self.assertEqual(resp.timezone, "UTC")

    def test_timezone_handling(self) -> None:
        fake_response = {
            "current": {
                "temperature_2m": 20.0,
                "weather_code": 0,
                "time": "2026-10-05T08:00",
            },
            "timezone": "auto",
        }
        transport = FakeTransport(response=fake_response)
        client = OpenMeteoClient(transport=transport)
        req = WeatherRequest(latitude=10, longitude=20)
        resp = client.get_forecast(req)
        self.assertIsNotNone(resp.timezone)
