import os
os.chdir("/app")

from generic.src.esp32_agent import Esp32Agent
import asyncio


class SmartLight(Esp32Agent):
    """
    Generic smart light agent. Instantiate once per physical LED.
    Set the agent name and ESP32 host to match the specific room.

    Example usage (three lights):
        SmartLight("LivingRoomLight", "LIVING_ROOM_ESP32_HOST")
        SmartLight("BedroomLight",    "BEDROOM_ESP32_HOST")
        SmartLight("KitchenLight",    "HALLWAY_ESP32_HOST")
    """

    def __init__(self, name: str = "LivingRoomLight", host_env: str = "LIGHT_ESP32_HOST"):
        super().__init__(
            name       = name,
            role       = "Actuator",
            esp32_host = os.environ.get(host_env, f"{name.lower()}.local"),
        )
        self.desc = (
            f"DescriptionStart: ESP32-S3 LED smart light ({name}). "
            "turn_on turns the light on at full brightness. "
            "turn_off turns the light off. "
            "set_brightness_by_percent_0_100 dims the light - pass "
            "params as {'level': 0-100}. 0 = off, 100 = full. "
            "get_status returns whether the light is on and current brightness. "
            "Send these actions directly without fetching state first. DescriptionEnd"
        )

        self.register_handlers({
            "turn_on":                          self.turn_on,
            "turn_off":                         self.turn_off,
            "set_brightness_by_percent_0_100":  self.set_brightness,
            "get_status":                       self.get_status,
            "dim_for_media":                    self.dim_for_media,
            "restore_from_media":               self.restore_from_media,

        })

    async def turn_on(self, msg: dict) -> str:
        result = await self._send("LIGHT_ON")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach {self.name} ({result})."
        return f"{self.name} is on."

    async def turn_off(self, msg: dict) -> str:
        result = await self._send("LIGHT_OFF")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach {self.name} ({result})."
        return f"{self.name} is off."

    async def set_brightness(self, msg: dict) -> str:
        level = msg.get("params", {}).get("level", 100)
        try:
            level = int(level)
        except (ValueError, TypeError):
            level = 100
        level = max(0, min(100, level))
        result = await self._send("LIGHT_BRIGHTNESS", {"level": level})
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach {self.name} ({result})."
        if level == 0:
            return f"{self.name} dimmed to off."
        return f"{self.name} set to {level}% brightness."

    async def get_status(self, msg: dict) -> str:
        data = await self._get_status()
        if not data:
            return f"Could not reach {self.name} to check status."
        on  = data.get("on", False)
        bri = data.get("brightness", 0)
        if not on:
            return f"{self.name} is off."
        return f"{self.name} is on at {bri}% brightness."
    
    async def dim_for_media(self, msg: dict) -> str:
        """Dim ONLY if currently on. Called by MockTV when Netflix opens."""
        data = await self._get_status()
        if data and data.get("on"):
            self._pre_media_brightness = data.get("brightness", 100)
            await self._send("LIGHT_BRIGHTNESS", {"level": 20})
            return f"{self.name} dimmed for media."
        return f"{self.name} was off, left as-is."

    async def restore_from_media(self, msg: dict) -> str:
        """Restore brightness ONLY if currently on."""
        data = await self._get_status()
        if data and data.get("on"):
            level = getattr(self, "_pre_media_brightness", 100)
            await self._send("LIGHT_BRIGHTNESS", {"level": level})
            return f"{self.name} brightness restored."
        return f"{self.name} was off, left as-is."



async def main():
    # Change the name and env var to match your specific light
    agent = SmartLight(
        name     = os.environ.get("LIGHT_NAME", "LivingRoomLight"),
        host_env = "LIGHT_ESP32_HOST",
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())