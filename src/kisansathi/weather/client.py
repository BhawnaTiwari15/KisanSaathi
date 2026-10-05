"""Open-Meteo client with transport abstraction."""

from collections.abc import Mapping
from json import JSONDecodeError
from typing import Protocol
from urllib import error as urlerror
from urllib import parse, request

from kisansathi.weather.exceptions import (
    InvalidCoordinatesError,
    WeatherParseError,
    WeatherRequestError,
    WeatherTimeoutError,
    WeatherUnavailableError,
)
from kisansathi.weather.models import (
    WeatherCurrent,
    WeatherForecastPoint,
    WeatherRequest,
    WeatherResponse,
)

OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_S = 10.0


class Transport(Protocol):
    def get_json(self, url: str, params: Mapping[str, object], timeout: float) -> object:
        """Return parsed JSON from the given URL with query parameters."""
        ...


class _StdlibTransport:
    def get_json(self, url: str, params: Mapping[str, object], timeout: float) -> object:
        if not isinstance(url, str) or not url:
            raise ValueError("url must be a non-empty string")
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        query = parse.urlencode(params, doseq=True)
        full_url = f"{url}?{query}" if query else url
        try:
            with request.urlopen(full_url, timeout=float(timeout)) as response:
                data = response.read()
        except TimeoutError as exc:
            raise WeatherTimeoutError("Weather request timed out") from exc
        except urlerror.URLError as exc:
            raise WeatherRequestError("Weather request failed") from exc
        except OSError as exc:
            raise WeatherRequestError("Weather request failed") from exc

        try:
            import json

            return json.loads(data.decode("utf-8"))
        except (UnicodeError, JSONDecodeError) as exc:
            raise WeatherParseError("Weather response is malformed") from exc


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str) and value.strip():
        try:
            # try int first
            if value.lstrip("+-").isdigit():
                return int(value)
            num = float(value)
            if num.is_integer():
                return int(num)
            return None
        except ValueError:
            return None
    return None


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value if value else None
    return None


class OpenMeteoClient:
    """Client for the Open-Meteo Forecast API."""

    def __init__(
        self,
        base_url: str = OPEN_METEO_BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: Transport | None = None,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
            raise ValueError("timeout_s must be a positive number")
        self._base_url = base_url.strip()
        self._timeout_s = float(timeout_s)
        self._transport = transport or _StdlibTransport()

    def get_forecast(
        self,
        req: WeatherRequest,
        *,
        include_forecast: bool = False,
        forecast_days: int = 1,
    ) -> WeatherResponse:
        if not isinstance(req, WeatherRequest):
            raise TypeError("req must be a WeatherRequest")
        if isinstance(forecast_days, bool) or not isinstance(forecast_days, int) or forecast_days < 1:
            raise ValueError("forecast_days must be a positive integer")

        params: dict[str, object] = {
            "latitude": req.latitude,
            "longitude": req.longitude,
            "current": [
                "temperature_2m",
                "precipitation",
                "rain",
                "wind_speed_10m",
                "wind_direction_10m",
                "weather_code",
            ],
            "timezone": req.timezone if req.timezone is not None else "auto",
        }
        if include_forecast:
            params["forecast_days"] = forecast_days
            params["daily"] = [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
                "wind_speed_10m_max",
                "wind_direction_10m_dominant",
                "weather_code",
            ]

        try:
            data = self._transport.get_json(self._base_url, params, self._timeout_s)
        except TimeoutError as exc:
            raise WeatherTimeoutError("Weather request timed out") from exc
        except InvalidCoordinatesError:
            raise
        except WeatherTimeoutError:
            raise
        except WeatherParseError:
            raise
        except WeatherRequestError:
            raise
        except WeatherUnavailableError:
            raise

        if not isinstance(data, dict):
            raise WeatherParseError("Weather response is malformed")

        current_data = data.get("current")
        if not isinstance(current_data, dict):
            raise WeatherUnavailableError("Current weather data is unavailable")

        current = WeatherCurrent(
            temperature_c=_coerce_float(current_data.get("temperature_2m")),
            precipitation_mm=_coerce_float(current_data.get("precipitation")),
            rain_mm=_coerce_float(current_data.get("rain")),
            wind_speed_mps=_coerce_float(current_data.get("wind_speed_10m")),
            wind_direction_deg=_coerce_float(current_data.get("wind_direction_10m")),
            weather_code=_coerce_int(current_data.get("weather_code")),
            time_iso=_coerce_str(current_data.get("time")),
        )

        forecast_points: tuple[WeatherForecastPoint, ...] = ()
        if include_forecast:
            daily_data = data.get("daily")
            if isinstance(daily_data, dict):
                times = daily_data.get("time")
                if isinstance(times, list) and times:
                    max_idx = len(times)
                    tmax_list = daily_data.get("temperature_2m_max") if isinstance(daily_data.get("temperature_2m_max"), list) else []
                    tmin_list = daily_data.get("temperature_2m_min") if isinstance(daily_data.get("temperature_2m_min"), list) else []
                    precip_sum = daily_data.get("precipitation_sum") if isinstance(daily_data.get("precipitation_sum"), list) else []
                    rain_sum = daily_data.get("rain_sum") if isinstance(daily_data.get("rain_sum"), list) else []
                    wind_max = daily_data.get("wind_speed_10m_max") if isinstance(daily_data.get("wind_speed_10m_max"), list) else []
                    wind_dir = daily_data.get("wind_direction_10m_dominant") if isinstance(daily_data.get("wind_direction_10m_dominant"), list) else []
                    codes = daily_data.get("weather_code") if isinstance(daily_data.get("weather_code"), list) else []
                    points: list[WeatherForecastPoint] = []
                    for i in range(min(max_idx, forecast_days)):
                        t_iso = _coerce_str(times[i]) or ""
                        point = WeatherForecastPoint(
                            time_iso=t_iso,
                            temperature_c=_coerce_float(tmax_list[i]) if i < len(tmax_list) else _coerce_float(tmin_list[i]) if i < len(tmin_list) else None,
                            precipitation_mm=_coerce_float(precip_sum[i]) if i < len(precip_sum) else None,
                            rain_mm=_coerce_float(rain_sum[i]) if i < len(rain_sum) else None,
                            wind_speed_mps=_coerce_float(wind_max[i]) if i < len(wind_max) else None,
                            wind_direction_deg=_coerce_float(wind_dir[i]) if i < len(wind_dir) else None,
                            weather_code=_coerce_int(codes[i]) if i < len(codes) else None,
                        )
                        points.append(point)
                    forecast_points = tuple(points)

        response_timezone = _coerce_str(data.get("timezone"))
        if response_timezone is None and req.timezone is not None:
            response_timezone = req.timezone

        try:
            return WeatherResponse(
                latitude=float(req.latitude),
                longitude=float(req.longitude),
                timezone=response_timezone,
                current=current,
                forecast=forecast_points,
            )
        except (TypeError, ValueError) as exc:
            raise WeatherParseError("Weather response is malformed") from exc
