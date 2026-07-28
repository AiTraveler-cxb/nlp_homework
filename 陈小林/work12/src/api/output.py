from pydantic import BaseModel, Field


class WeatherInfo(BaseModel):
    city: str = Field(..., description="City name")
    weather: str = Field(..., description="Weather details")
