import unittest

from kisansathi.weather.models import (
    WeatherCurrent,
    WeatherForecastPoint,
    WeatherRequest,
    WeatherResponse,
)


class WeatherModelsTests(unittest.TestCase):
    def test_weather_request_valid(self) -> None:
        req = WeatherRequest(latitude=28.6139, longitude=77.2090, timezone="Asia/Kolkata")
        self.assertEqual(req.latitude, 28.6139)
        self.assertEqual(req.longitude, 77.2090)
        self.assertEqual(req.timezone, "Asia/Kolkata")

    def test_weather_request_no_timezone(self) -> None:
        req = WeatherRequest(latitude=0, longitude=0)
        self.assertIsNone(req.timezone)

    def test_weather_request_invalid_latitude(self) -> None:
        with self.assertRaises(ValueError):
            WeatherRequest(latitude=91, longitude=0)
        with self.assertRaises(ValueError):
            WeatherRequest(latitude=-91, longitude=0)

    def test_weather_request_invalid_longitude(self) -> None:
        with self.assertRaises(ValueError):
            WeatherRequest(latitude=0, longitude=181)
        with self.assertRaises(ValueError):
            WeatherRequest(latitude=0, longitude=-181)

    def test_weather_request_invalid_timezone(self) -> None:
        with self.assertRaises(ValueError):
            WeatherRequest(latitude=0, longitude=0, timezone="")

    def test_weather_current_optional_fields(self) -> None:
        current = WeatherCurrent()
        self.assertIsNone(current.temperature_c)
        self.assertIsNone(current.weather_code)

    def test_weather_current_with_values(self) -> None:
        current = WeatherCurrent(
            temperature_c=25.5,
            precipitation_mm=0.0,
            rain_mm=0.0,
            wind_speed_mps=3.2,
            wind_direction_deg=120,
            weather_code=1,
            time_iso="2026-10-05T12:00",
        )
        self.assertEqual(current.temperature_c, 25.5)
        self.assertEqual(current.weather_code, 1)

    def test_weather_forecast_point(self) -> None:
        point = WeatherForecastPoint(time_iso="2026-10-06T00:00", temperature_c=30.0)
        self.assertEqual(point.time_iso, "2026-10-06T00:00")

    def test_weather_forecast_point_invalid(self) -> None:
        with self.assertRaises(ValueError):
            WeatherForecastPoint(time_iso="")

    def test_weather_response(self) -> None:
        current = WeatherCurrent(temperature_c=25.0)
        resp = WeatherResponse(
            latitude=1.0,
            longitude=2.0,
            timezone="auto",
            current=current,
            forecast=(WeatherForecastPoint(time_iso="2026-10-06T00:00"),),
        )
        self.assertEqual(resp.latitude, 1.0)
        self.assertEqual(len(resp.forecast), 1)
        self.assertIs(resp.current, current)
