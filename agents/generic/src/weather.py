import os
import asyncio
import aiohttp
import urllib.parse
import uuid
from discovery.src.base_agent import BaseAgent
from loguru import logger

class WeatherAgent(BaseAgent):
    def __init__(self, name="HALO-Weather", role="Aggregator"):
        super().__init__(name=name, role=role)
        
        self.register_handlers({
            "get_current_weather": self.get_weather
        })
        
        self.desc = "Fetches live weather data. Requires 'location' parameter (e.g. 'London' or 'New York')."

    async def get_weather(self, msg: dict):
        location = msg.get("params", {}).get("location", "London")
        safe_location = urllib.parse.quote(location)
        
        print(f"[{self.name}] ☁️ Fetching weather for '{location}'...")

        try:
            async with aiohttp.ClientSession() as session:
                # STEP 1: Geocoding (Turn city name into Lat/Long)
                # 100% Free Open-Meteo Geocoding API
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={safe_location}&count=1&format=json"
                async with session.get(geo_url) as geo_res:
                    geo_data = await geo_res.json()
                    
                    if "results" not in geo_data:
                        return {"error": f"Could not find coordinates for {location}"}
                        
                    lat = geo_data["results"][0]["latitude"]
                    lon = geo_data["results"][0]["longitude"]
                    city = geo_data["results"][0]["name"]

                # STEP 2: Fetch the actual weather
                # 100% Free Open-Meteo Weather API
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
                async with session.get(weather_url) as weather_res:
                    weather_data = await weather_res.json()
                    current = weather_data.get("current_weather", {})
                    
                    # WMO Weather interpretation codes
                    weather_codes = {
                        0: "Clear sky ☀️",
                        1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
                        45: "Fog 🌫️", 48: "Depositing rime fog 🌫️",
                        51: "Light drizzle 🌧️", 53: "Moderate drizzle 🌧️", 55: "Dense drizzle 🌧️",
                        61: "Light rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
                        71: "Light snow ❄️", 73: "Moderate snow ❄️", 75: "Heavy snow ❄️",
                        95: "Thunderstorm ⛈️"
                    }
                    
                    condition = weather_codes.get(current.get("weathercode"), "Unknown condition")
                    
                    result = {
                        "location": city,
                        "temperature_celsius": current.get("temperature"),
                        "windspeed_kmh": current.get("windspeed"),
                        "condition": condition,
                        "is_day": current.get("is_day") == 1
                    }
                    
                    logger.debug(f"[{self.name}] ✅ Weather in {city}: {result['temperature_celsius']}°C, {condition}")
                    return result

        except Exception as e:
            print(f"[{self.name}] ❌ Error fetching weather: {e}")
            return {"error": str(e)}

async def main():
    container_id = os.environ.get("HOSTNAME", uuid.uuid4().hex)[:6]
    agent = WeatherAgent(name=f"Weather-{container_id}")
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())