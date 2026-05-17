"""
esp32_agent.py
==============
Base class for any HALO agent that talks to an ESP32 over HTTP.
Inherit from this instead of BaseAgent directly.

Handles:
  - HMAC-SHA256 request signing
  - HTTP POST to /command
  - Automatic retries on connection error
  - State syncing via GET /status

Usage:
    class SmartLight(Esp32Agent):
        def __init__(self):
            super().__init__("SmartLight", "Actuator", "smartlight.local")
            self.register_handlers({
                "turn_on":  self.turn_on,
                "turn_off": self.turn_off,
            })

        async def turn_on(self, msg: dict) -> str:
            return await self._send("LIGHT_ON")

        async def turn_off(self, msg: dict) -> str:
            return await self._send("LIGHT_OFF")
"""

from discovery.src.base_agent import BaseAgent
import asyncio
import hashlib
import hmac
import json
import os
import time
import requests
from loguru import logger


def _load_hmac_key() -> bytes:
    hex_key = os.environ.get("HALO_HTTP_KEY", "")
    if hex_key:
        return bytes.fromhex(hex_key)
    key_path = ".keys/halo_http_key.hex"
    if os.path.exists(key_path):
        with open(key_path) as f:
            return bytes.fromhex(f.read().strip())
    raise RuntimeError("No HALO_HTTP_KEY found. Run generate_halo_http_key.py first.")


class Esp32Agent(BaseAgent):
    """
    Base class for HALO agents that proxy commands to an ESP32 over HTTP.
    Subclasses only need to define their handlers and call self._send().
    """

    def __init__(self, name: str, role: str, esp32_host: str) -> None:
        super().__init__(name, role)
        self.esp32_host = esp32_host
        self.base_url   = f"http://{esp32_host}"
        self._hmac_key  = _load_hmac_key()
        self._online    = False

    # =========================================================================
    # Signing
    # =========================================================================

    def _sign(self, body: str) -> tuple:
        ts      = str(int(time.time()))
        message = (ts + "." + body).encode()
        sig     = hmac.new(self._hmac_key, message, hashlib.sha256).hexdigest()
        return ts, sig

    # =========================================================================
    # HTTP
    # =========================================================================

    async def _send(self, cmd: str, params: dict = None) -> str:
        """
        POST a signed command to the ESP32.
        Returns "OK", "AUTH_ERROR", "UNREACHABLE", or "TIMEOUT".
        """
        payload = {"cmd": cmd}
        if params:
            payload["params"] = params

        body    = json.dumps(payload)
        ts, sig = self._sign(body)
        loop    = asyncio.get_running_loop()

        try:
            resp = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.base_url}/command",
                    data=body,
                    headers={
                        "Content-Type": "application/json",
                        "X-HALO-Ts":   ts,
                        "X-HALO-Sig":  sig,
                    },
                    timeout=10,
                )
            )
            self._online = resp.status_code in (200, 202)
            if resp.status_code == 401:
                logger.error(f"[{self.name}] Auth rejected — check HMAC key matches firmware")
                return "AUTH_ERROR"
            logger.debug(f"[{self.name}] → {cmd} ({resp.status_code})")
            return resp.json().get("status", "OK")

        except requests.exceptions.ConnectionError:
            self._online = False
            logger.warning(f"[{self.name}] Cannot reach {self.base_url}")
            return "UNREACHABLE"
        except requests.exceptions.Timeout:
            self._online = True   # device is there, just busy
            logger.debug(f"[{self.name}] Timeout on {cmd} — device busy")
            return "TIMEOUT"

    async def _get_status(self) -> dict:
        """Fetch current device state from GET /status."""
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{self.base_url}/status", timeout=5)
            )
            if resp.status_code == 200:
                data = resp.json()
                self.state.update(data)
                return data
        except Exception as e:
            logger.warning(f"[{self.name}] Status fetch failed: {e}")
        return {}

    async def _ping(self) -> bool:
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: requests.get(f"{self.base_url}/ping", timeout=3)
            )
            self._online = resp.status_code == 200
        except Exception:
            self._online = False
        return self._online

    async def _health_loop(self) -> None:
        """Ping the ESP32 every 10 s and update self._online."""
        while True:
            await self._ping()
            await asyncio.sleep(10.0)

    async def run(self) -> None:
        asyncio.create_task(self._health_loop())
        await super().run()