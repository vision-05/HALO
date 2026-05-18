import os
os.chdir("/app")

from generic.src.esp32_agent import Esp32Agent
import asyncio


class TvLightCombo(Esp32Agent):
    """
    One ESP32 that is BOTH the mock TV (TFT screen) and the living
    room LED light. Exposes every TV and light handler. The combined
    firmware routes TV_* commands to the screen and LIGHT_* to the LED.

    Because screen and light are the SAME device, the Netflix->dim
    automation is handled internally here (no second agent or mesh
    message needed).
    """

    def __init__(self):
        super().__init__(
            name       = "LivingRoom",
            role       = "Actuator",
            esp32_host = os.environ.get("COMBO_ESP32_HOST", "livingroom.local"),
        )
        self.desc = (
            "DescriptionStart: ESP32-S3 living room device - it is BOTH a TV "
            "(TFT screen) and the living room LED light. "
            "power_on shows the TV home screen. power_off blanks the screen. "
            "go_to_home_screen shows the TV home screen. "
            "open_netflix plays the Netflix intro then the Netflix home screen "
            "and automatically dims the living room light if it is on. "
            "close_netflix returns to the TV home screen and restores the "
            "light brightness if it is on. "
            "turn_on / turn_off control the living room LED. "
            "set_brightness_by_percent_0_100 dims the LED - params {'level':0-100}. "
            "get_status reports light on/off and brightness. "
            "Send these actions directly without fetching state first. DescriptionEnd"
        )

        # Remember brightness before Netflix dim so we can restore it
        self._pre_netflix_brightness = 100

        self.register_handlers({
            # ── TV handlers ───────────────────────────────────────────────
            "power_on":           self.power_on,
            "power_off":          self.power_off,
            "go_to_home_screen":  self.go_to_home_screen,
            "open_netflix":       self.open_netflix,
            "close_netflix":      self.close_netflix,
            # ── Light handlers ────────────────────────────────────────────
            "turn_on":                          self.turn_on,
            "turn_off":                         self.turn_off,
            "set_brightness_by_percent_0_100":  self.set_brightness,
            "get_status":                       self.get_status,
        })

    # ── TV handlers ───────────────────────────────────────────────────────────

    async def power_on(self, msg: dict) -> str:
        """Turn the TV on - shows the TV home screen."""
        result = await self._send("TV_HOME")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room device ({result})."
        return "TV on - showing the home screen."

    async def power_off(self, msg: dict) -> str:
        """Turn the TV off - blanks the screen."""
        result = await self._send("TV_OFF")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room device ({result})."
        return "TV off."

    async def go_to_home_screen(self, msg: dict) -> str:
        """Go to the TV home screen."""
        result = await self._send("TV_HOME")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room device ({result})."
        return "Showing the TV home screen."

    async def open_netflix(self, msg: dict) -> str:
        """Open Netflix and dim the living room light if it is on."""
        # Check the light state first (same device, so _get_status works)
        status = await self._get_status()
        if status and status.get("on"):
            self._pre_netflix_brightness = status.get("brightness", 100)
            # Dim before playing the intro
            await self._send("LIGHT_BRIGHTNESS", {"level": 20})

        result = await self._send("NETFLIX")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room device ({result})."

        if status and status.get("on"):
            return "Netflix opened and the living room light dimmed for movie mode."
        return "Netflix opened."

    async def close_netflix(self, msg: dict) -> str:
        """Close Netflix (back to TV home) and restore the light if it is on."""
        result = await self._send("TV_HOME")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room device ({result})."

        status = await self._get_status()
        if status and status.get("on"):
            level = getattr(self, "_pre_netflix_brightness", 100)
            await self._send("LIGHT_BRIGHTNESS", {"level": level})
            return "Netflix closed and the living room light restored."
        return "Netflix closed - back to the TV home screen."

    # ── Light handlers ────────────────────────────────────────────────────────

    async def turn_on(self, msg: dict) -> str:
        result = await self._send("LIGHT_ON")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room light ({result})."
        return "Living room light is on."

    async def turn_off(self, msg: dict) -> str:
        result = await self._send("LIGHT_OFF")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room light ({result})."
        return "Living room light is off."

    async def set_brightness(self, msg: dict) -> str:
        level = msg.get("params", {}).get("level", 100)
        try:
            level = int(level)
        except (ValueError, TypeError):
            level = 100
        level = max(0, min(100, level))
        result = await self._send("LIGHT_BRIGHTNESS", {"level": level})
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the living room light ({result})."
        if level == 0:
            return "Living room light dimmed to off."
        return f"Living room light set to {level}% brightness."

    async def get_status(self, msg: dict) -> str:
        data = await self._get_status()
        if not data:
            return "Could not reach the living room device to check status."
        on  = data.get("on", False)
        bri = data.get("brightness", 0)
        if not on:
            return "The living room light is off."
        return f"The living room light is on at {bri}% brightness."


async def main():
    agent = TvLightCombo()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())