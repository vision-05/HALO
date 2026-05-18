import os
os.chdir("/app")

from generic.src.esp32_agent import Esp32Agent
import asyncio
import json


class Blinds(Esp32Agent):
    def __init__(self):
        super().__init__(
            name       = "Blinds",
            role       = "Actuator",
            esp32_host = os.environ.get("BLINDS_ESP32_HOST", "blinds.local"),
        )
        self.desc = (
            "DescriptionStart: ESP32-S3 window blinds. "
            "open_blinds tilts the slats open (small clockwise turn). "
            "close_blinds tilts the slats closed (small anticlockwise turn). "
            "get_status reports whether the blinds are open, closed, or unknown. "
            "Send these actions directly without fetching state first. DescriptionEnd"
        )

        self.register_handlers({
            "open_blinds":  self.open_blinds,
            "close_blinds": self.close_blinds,
            "get_status":   self.get_status,
        })

    async def open_blinds(self, msg: dict) -> str:
        """Open the blinds, then turn the living room light off."""
        result = await self._send("BLINDS_OPEN")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the blinds ({result})."

        # Automation: blinds open → living room light off
        await self.send_msg("LivingRoomLight", json.dumps({
            "action": "turn_off",
            "source": self.name,
            "target": "LivingRoomLight",
            "params": {}
        }))
        return "Blinds opened, living room light turned off."

    async def close_blinds(self, msg: dict) -> str:
        """Close the blinds."""
        result = await self._send("BLINDS_CLOSE")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the blinds ({result})."
        return "Blinds closed."

    async def get_status(self, msg: dict) -> str:
        """Report current blind position."""
        data = await self._get_status()
        if not data:
            return "I couldn't reach the blinds to check their status."

        state = data.get("state", "unknown")
        if state == "open":
            return "open"
        if state == "closed":
            return "closed"
        return "The blinds position is unknown (not moved since power-on)."
    

async def main():
    agent = Blinds()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())