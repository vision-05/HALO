from generic.src.esp32_agent import Esp32Agent
import asyncio
import os


class MockTV(Esp32Agent):
    def __init__(self):
        super().__init__(
            name       = "MockTV",
            role       = "Actuator",
            esp32_host = os.environ.get("TV_ESP32_HOST", "mocktv.local"),
        )
        self.desc = (
            "ESP32-S3 TFT mock TV. "
            "power_on shows the TV home screen. "
            "open_netflix plays the Netflix intro then shows the Netflix home screen. "
            "power_off blanks the screen."
        )

        self.register_handlers({
            "power_on":      self.power_on,
            "power_off":     self.power_off,
            "open_netflix":  self.open_netflix,
        })

    async def power_on(self, msg: dict) -> str:
        """Show the TV home screen."""
        return await self._send("TV_HOME")

    async def power_off(self, msg: dict) -> str:
        """Blank the screen."""
        return await self._send("TV_OFF")

    async def open_netflix(self, msg: dict) -> str:
        """Play the Netflix intro animation then show the Netflix home screen."""
        return await self._send("NETFLIX")
    
    async def go_to_home_screen(self, msg: dict) -> str:
        """Go to the TV home screen."""
        return await self._send("TV_HOME")
    
    async def close_netflix(self, msg: dict) -> str:
        """Close Netflix and return to the TV home screen (TV stays on)."""
        return await self._send("TV_HOME")


async def main():
    agent = MockTV()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())