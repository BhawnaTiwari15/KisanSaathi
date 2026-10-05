"""Typed domain models for weather data (Open-Meteo)."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeatherRequest:
    """Request parameters for weather forecast."""

    latitude: float
    longitude: float
    timezone: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.latitude, (int, float)) or isinstance(self.latitude, bool):
            raise TypeError("latitude must be a number")
        if not isinstance(self.longitude, (int, float)) or isinstance(self.longitude, bool):
            raise TypeError("longitude must be a number")
        if self.latitude < -90.0 or self.latitude > 90.0:
            raise ValueError("latitude must be within [-90, 90]")
        if self.longitude < -180.0 or self.longitude > 180.0:
            raise ValueError("longitude must be within [-180, 180]")
        if self.timezone is not None:
            if not isinstance(self.timezone, str) or not self.timezone.strip():
                raise ValueError("timezone must be a non-empty string if provided")


@dataclass(frozen=True, slots=True)
class WeatherCurrent:
    """Current weather conditions."""

    temperature_c: float | None = None
    precipitation_mm: float | None = None
    rain_mm: float | None = None
    wind_speed_mps: float | None = None
    wind_direction_deg: float | None = None
    weather_code: int | None = None
    time_iso: str | None = None

    def __post_init__(self) -> None:
        self._validate_optional_float(self.temperature_c, "temperature_c")
        self._validate_optional_float(self.precipitation_mm, "precipitation_mm")
        self._validate_optional_float(self.rain_mm, "rain_mm")
        self._validate_optional_float(self.wind_speed_mps, "wind_speed_mps")
        self._validate_optional_float(self.wind_direction_deg, "wind_direction_deg")
        if self.weather_code is not None:
            if isinstance(self.weather_code, bool) or not isinstance(self.weather_code, int):
                raise TypeError("weather_code must be an integer or None")
        if self.time_iso is not None:
            if not isinstance(self.time_iso, str):
                raise TypeError("time_iso must be a string or None")

    @staticmethod
    def _validate_optional_float(value: float | None, field_name: str) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be a number or None")

    _validate_optional_float_static = _validate_optional_float


@dataclass(frozen=True, slots=True)
class WeatherForecastPoint:
    """A single weather forecast point."""

    time_iso: str
    temperature_c: float | None = None
    precipitation_mm: float | None = None
    rain_mm: float | None = None
    wind_speed_mps: float | None = None
    wind_direction_deg: float | None = None
    weather_code: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.time_iso, str) or not self.time_iso.strip():
            raise ValueError("time_iso must be a non-empty string")
        WeatherCurrent._validate_optional_float_static(self.temperature_c, "temperature_c")
        WeatherCurrent._validate_optional_float_static(self.precipitation_mm, "precipitation_mm")
        WeatherCurrent._validate_optional_float_static(self.rain_mm, "rain_mm")
        WeatherCurrent._validate_optional_float_static(self.wind_speed_mps, "wind_speed_mps")
        WeatherCurrent._validate_optional_float_static(self.wind_direction_deg, "wind_direction_deg")
        if self.weather_code is not None:
            if isinstance(self.weather_code, bool) or not isinstance(self.weather_code, int):
                raise TypeError("weather_code must be an integer or None")


@dataclass(frozen=True, slots=True)
class WeatherResponse:
    """Weather forecast response from Open-Meteo."""

    latitude: float
    longitude: float
    timezone: str | None
    current: WeatherCurrent
    forecast: tuple[WeatherForecastPoint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.latitude, (int, float)) or isinstance(self.latitude, bool):
            raise TypeError("latitude must be a number")
        if not isinstance(self.longitude, (int, float)) or isinstance(self.longitude, bool):
            raise TypeError("longitude must be a number")
        if self.timezone is not None and not isinstance(self.timezone, str):
            raise TypeError("timezone must be a string or None")
        if not isinstance(self.current, WeatherCurrent):
            raise TypeError("current must be a WeatherCurrent")
        if not isinstance(self.forecast, tuple):
            raise TypeError("forecast must be a tuple")
        for point in self.forecast:
            if not isinstance(point, WeatherForecastPoint):
                raise TypeError("forecast must contain only WeatherForecastPoint objects")
