import os
os.chdir("/app")

from generic.src.esp32_agent import Esp32Agent
import asyncio
import json
from loguru import logger


class WashingMachine(Esp32Agent):
    def __init__(self):
        super().__init__(
            name       = "WashingMachine",
            role       = "Actuator",
            esp32_host = os.environ.get("WASH_ESP32_HOST", "washingmachine.local"),
        )
        self.desc = (
            "DescriptionStart: ESP32-S3 washing machine. "
            "start_wash begins a timed cycle - pass the number of hours in "
            "params as {'hours': N}. The drum spins for a (time-scaled) cycle "
            "and the user is messaged automatically when it finishes. "
            "stop_wash aborts the cycle. spin_cycle runs a short manual spin. "
            "get_status returns idle / running (with time left) / done. "
            "Send these actions directly without fetching state first. DescriptionEnd"
        )

        self.register_handlers({
            "start_wash":  self.start_wash,
            "stop_wash":   self.stop_wash,
            "spin_cycle":  self.spin_cycle,
            "get_status":  self.get_status,
        })

        # Tracks whether a cycle is active so we only watch when needed
        self._cycle_active = False

    # ─── Handlers ─────────────────────────────────────────────────────────────

    async def start_wash(self, msg: dict) -> str:
        hours = msg.get("params", {}).get("hours", 1)
        try:
            hours = int(hours)
        except (ValueError, TypeError):
            hours = 1

        result = await self._send("WASH_START", {"hours": hours})
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the washing machine ({result})."

        self._cycle_active = True
        return (f"Washing machine started on a {hours}-hour cycle. "
                f"I'll message you when it's finished.")

    async def stop_wash(self, msg: dict) -> str:
        result = await self._send("WASH_STOP")
        self._cycle_active = False
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the washing machine ({result})."
        return "Washing machine stopped."

    async def spin_cycle(self, msg: dict) -> str:
        result = await self._send("WASH_SPIN")
        if result in ("UNREACHABLE", "TIMEOUT", "AUTH_ERROR"):
            return f"Could not reach the washing machine ({result})."
        self._cycle_active = True
        return "Running a quick spin cycle."

    async def get_status(self, msg: dict) -> str:
        data = await self._get_status()
        if not data:
            return "I couldn't reach the washing machine to check its status."

        state = data.get("state", "unknown")
        hours = data.get("requested_hours", 0)
        secs  = data.get("seconds_left", 0)

        if state == "idle":
            return "The washing machine is idle - not running."
        if state == "running":
            mins, s = divmod(int(secs), 60)
            remaining = f"{mins} min {s} sec" if mins else f"{s} sec"
            return (f"The washing machine is running a {hours}-hour cycle. "
                    f"About {remaining} of (scaled) time remaining.")
        if state == "done":
            return "The washing machine has finished its cycle."
        return f"Washing machine state: {state}."

    # ─── Background completion watcher ────────────────────────────────────────

    async def _notify_user(self, message: str) -> None:
        """Ask the LanguageAgent to send a Telegram message to the user."""
        payload = {
            "action": "send_chat_message",
            "source": self.name,
            "target": "LanguageAgent",
            "params": {"message": message},
        }
        await self.send_msg("LanguageAgent", json.dumps(payload))

    async def _watch_cycle(self) -> None:
        """
        Poll the ESP32 every 3s. When it reports just_finished, push a
        Telegram message to the user via the LanguageAgent.
        """
        while True:
            try:
                if self._cycle_active:
                    data = await self._get_status()
                    if data and data.get("just_finished"):
                        logger.success("[WashingMachine] Cycle finished - notifying user")
                        await self._notify_user(
                            "🧺 Your washing machine has finished its cycle!"
                        )
                        self._cycle_active = False
            except Exception as e:
                logger.warning(f"[WashingMachine] watch error: {e}")
            await asyncio.sleep(3.0)

    async def run(self) -> None:
        asyncio.create_task(self._watch_cycle())
        await super().run()


async def main():
    agent = WashingMachine()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())