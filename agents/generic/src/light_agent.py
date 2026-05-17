from esp32_agent import Esp32Agent
import asyncio
import os

class SmartLight(Esp32Agent):
    def __init__(self):
        super().__init__(
            name      = "SmartLight",
            role      = "Actuator",
            esp32_host = os.environ.get("LIGHT_ESP32_HOST", "smartlight.local"),
        )
        self.desc = "Controls an ESP32-S3 LED smart light. Supports on, off, toggle and brightness."

        self.register_handlers({
            "turn_on":                         self.turn_on,
            "turn_off":                        self.turn_off,
            "toggle":                          self.toggle,
            "set_brightness_by_percent_0_100": self.set_brightness,
            "get_status":                      self.get_status,
        })

    async def turn_on(self, msg: dict) -> str:
        return await self._send("LIGHT_ON")

    async def turn_off(self, msg: dict) -> str:
        return await self._send("LIGHT_OFF")

    async def toggle(self, msg: dict) -> str:
        return await self._send("LIGHT_TOGGLE")

    async def set_brightness(self, msg: dict) -> str:
        level = int(msg.get("params", {}).get("level", 100))
        level = max(0, min(100, level))
        return await self._send("LIGHT_BRIGHTNESS", {"level": level})

    async def get_status(self, msg: dict) -> dict:
        return await self._get_status()


async def main():
    agent = SmartLight()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())