from discovery.src.base_agent import BaseAgent
import asyncio
import aiohttp
import json
import os
from loguru import logger
from datetime import datetime, timedelta

MIELE_API_BASE = "https://api.mcs3.miele.com/v1"
MIELE_AUTH_BASE = "https://auth.domestic.miele-iot.com/partner/realms/mcs/protocol/openid-connect"

# Program IDs for WT1 washer-dryer
PROGRAM_IDS = {
    "cottons": 1,
    "minimum_iron": 2,
    "delicates": 3,
    "woollens": 4,
    "silks": 5,
    "drum_clean": 6,
    "rinse": 7,
    "spin_drain": 8,
    "automatic_plus": 10,
    "express_20": 15,
    "denim": 16,
    "proofing": 17,
    "sportswear": 18,
    "outerwear": 19,
    "pillows": 20,
    "quick_power_wash": 21,
    # Drying programs (WT1 is a washer-dryer combo)
    "dry_cottons": 30,
    "dry_minimum_iron": 31,
    "dry_woollens": 32,
    "dry_gentle": 33,
    "dry_timed_warm": 34,
    "dry_timed_cold": 35,
}

# Status codes returned by the API
STATUS_CODES = {
    1: "Off",
    2: "On",
    3: "Programmed",
    4: "Programmed waiting to start",
    5: "Running",
    6: "Pause",
    7: "End",
    8: "Failure",
    9: "Abort",
    10: "Idle",
    11: "Rinse hold",
    12: "Service",
    13: "Super freezing",
    14: "Super cooling",
    15: "Super heating",
    144: "Default",
}


class MieleWasherDryer(BaseAgent):
    def __init__(self) -> None:
        super().__init__("MieleWasherDryer", "Actuator")

        self.client_id = os.environ.get("MIELE_CLIENT_ID")
        self.client_secret = os.environ.get("MIELE_CLIENT_SECRET")
        self.token_file = "generic/miele_tokens.json"
        self.device_id = None
        self._session: aiohttp.ClientSession = None

        self.state = {
            "status": "unknown",
            "status_raw": None,
            "program": None,
            "program_id": None,
            "remaining_time": None,
            "elapsed_time": None,
            "spin_speed": None,
            "temperature": None,
            "door": "unknown",
            "mobile_start": False,
            "remote_control": False,
            "drying_step": None,
            "error": None,
        }

        self.desc = (
            "DescriptionStart: Miele WT1 washer-dryer controller. "
            "Can start, stop, pause programs and query status. "
            "Machine must have Mobile Start enabled on its panel before remote start is possible. "
            "Use get_status to check current state before sending commands. "
            "DescriptionEnd"
        )

        self.register_handlers({
            "start_program":           self.start_program,
            "start":                   self.start,
            "stop":                    self.stop,
            "pause":                   self.pause,
            "get_status":              self.get_status_handler,
            "get_programs":            self.get_programs,
            "enable_mobile_start":     self.enable_mobile_start,
        })

    # ------------------------------------------------------------------
    # Auth / token management
    # ------------------------------------------------------------------

    async def _load_tokens(self) -> dict:
        if os.path.exists(self.token_file):
            with open(self.token_file) as f:
                return json.load(f)
        raise RuntimeError("No Miele tokens found. Run miele_auth.py first.")

    async def _save_tokens(self, tokens: dict) -> None:
        os.makedirs(os.path.dirname(self.token_file), exist_ok=True)
        with open(self.token_file, "w") as f:
            json.dump(tokens, f)

    async def _refresh_access_token(self, tokens: dict) -> dict:
        logger.debug("Refreshing Miele access token")
        async with self._session.post(
                f"{MIELE_AUTH_BASE}/token",
                data={
                    "grant_type":    "refresh_token",
                    "refresh_token": tokens["refresh_token"],
                    "client_id":     self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Token refresh failed ({resp.status}): {text}")
            new_tokens = await resp.json()
            new_tokens["refreshed_at"] = datetime.now().isoformat()
            if "refresh_token" not in new_tokens:
                new_tokens["refresh_token"] = tokens["refresh_token"]
                await self._save_tokens(new_tokens)
                logger.success("Miele access token refreshed")
                return new_tokens

    async def _get_valid_token(self) -> str:
        tokens = await self._load_tokens()
        refreshed_at = tokens.get("refreshed_at")
        expires_in = tokens.get("expires_in", 3600)

        if refreshed_at:
            age = (datetime.now() - datetime.fromisoformat(refreshed_at)).total_seconds()
            # Refresh at 90% of lifetime
            if age >= expires_in * 0.9:
                tokens = await self._refresh_access_token(tokens)

        return tokens["access_token"]

    def _auth_headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Accept":        "application/json",
            "Content-Type":  "application/json",
        }

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    async def _discover_device(self) -> None:
        token = await self._get_valid_token()
        async with self._session.get(
            f"{MIELE_API_BASE}/devices",
            headers=self._auth_headers(token),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to list devices: {resp.status}")
            devices = await resp.json()

        for device_id, device in devices.items():
            type_raw = device.get("ident", {}).get("type", {}).get("value_raw")
            # Type 12 = Washer-dryer combo
            # Type 1  = Washing machine (fallback)
            if type_raw in (12, 1):
                self.device_id = device_id
                tech_type = device.get("ident", {}).get("deviceIdentLabel", {}).get("techType", "unknown")
                logger.success(f"Found Miele device: {tech_type} (ID: {device_id})")
                self.state["device_id"] = device_id
                self.state["tech_type"] = tech_type
                return

        raise RuntimeError("No Miele washer or washer-dryer found on account")

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    async def _get_device_state(self) -> dict:
        token = await self._get_valid_token()
        async with self._session.get(
            f"{MIELE_API_BASE}/devices/{self.device_id}/state",
            headers=self._auth_headers(token),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"Failed to get device state ({resp.status}): {text}")
            return await resp.json()

    async def _send_action(self, payload: dict) -> bool:
        token = await self._get_valid_token()
        async with self._session.put(
            f"{MIELE_API_BASE}/devices/{self.device_id}/actions",
            headers=self._auth_headers(token),
            json=payload,
        ) as resp:
            if resp.status in (200, 204):
                return True
            text = await resp.text()
            logger.error(f"Action failed ({resp.status}): {text}")
            return False

    async def _check_action_available(self, action: str) -> bool:
        token = await self._get_valid_token()
        async with self._session.get(
            f"{MIELE_API_BASE}/devices/{self.device_id}/actions",
            headers=self._auth_headers(token),
        ) as resp:
            if resp.status != 200:
                return False
            data = await resp.json()
            available = data.get("processAction", [])
            # processAction values: 1=Start, 2=Stop, 3=Pause, 4=StartSuperFreezing,
            # 5=StopSuperFreezing, 6=StartSuperCooling, 7=StopSuperCooling
            action_map = {"start": 1, "stop": 2, "pause": 3}
            return action_map.get(action) in available

    # ------------------------------------------------------------------
    # State polling
    # ------------------------------------------------------------------

    async def _poll_state(self) -> None:
        while True:
            try:
                raw = await self._get_device_state()
                self._parse_state(raw)
            except Exception as e:
                logger.warning(f"State poll failed: {e}")
            await asyncio.sleep(30)

    def _parse_state(self, raw: dict) -> None:
        status_raw = raw.get("status", {}).get("value_raw")
        self.state.update({
            "status":         STATUS_CODES.get(status_raw, "unknown"),
            "status_raw":     status_raw,
            "program":        raw.get("ProgramID", {}).get("value_localized"),
            "program_id":     raw.get("ProgramID", {}).get("value_raw"),
            "remaining_time": raw.get("remainingTime", [None, None]),
            "elapsed_time":   raw.get("elapsedTime", [None, None]),
            "spin_speed":     raw.get("spinningSpeed", {}).get("value_localized"),
            "temperature":    raw.get("targetTemperature", [{}])[0].get("value_localized"),
            "door":           "open" if raw.get("signalDoor") else "closed",
            "mobile_start":   raw.get("remoteEnable", {}).get("mobileStart", False),
            "remote_control": raw.get("remoteEnable", {}).get("fullRemoteControl", False),
            "drying_step":    raw.get("dryingStep", {}).get("value_localized"),
            "error":          raw.get("signalFailure"),
        })
        logger.debug(f"State updated: {self.state['status']} | program: {self.state['program']}")

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def start_program(self, msg: dict) -> None:
        """Start a specific wash/dry program by name or ID.
        
        params:
            program (str): program name from PROGRAM_IDS, OR
            program_id (int): raw program ID
            temperature (int, optional): target temp in Celsius
            spin_speed (int, optional): spin speed in RPM
        """
        params = msg.get("params", {})
        program_name = params.get("program")
        program_id = params.get("program_id")

        if program_name and not program_id:
            program_id = PROGRAM_IDS.get(program_name.lower().replace(" ", "_"))
            if program_id is None:
                logger.error(f"Unknown program: {program_name}")
                return

        if not self.state.get("mobile_start"):
            logger.error("Mobile Start is not enabled on the machine panel. Cannot start remotely.")
            return

        payload = {"programId": program_id}
        if params.get("temperature"):
            payload["temperature"] = params["temperature"]
        if params.get("spin_speed"):
            payload["spinningSpeed"] = params["spin_speed"]

        success = await self._send_action(payload)
        if success:
            self.state.update({"program": program_name or str(program_id), "status": "Programmed"})
            logger.success(f"Program {program_name or program_id} started")

    async def start(self, msg: dict) -> None:
        """Start the currently loaded program (machine must be in Programmed state)."""
        if not await self._check_action_available("start"):
            logger.error("Start action not available in current state")
            return
        success = await self._send_action({"processAction": 1})
        if success:
            self.state["status"] = "Running"
            logger.success("Washer-dryer started")

    async def stop(self, msg: dict) -> None:
        """Stop the current program."""
        if not await self._check_action_available("stop"):
            logger.error("Stop action not available in current state")
            return
        success = await self._send_action({"processAction": 2})
        if success:
            self.state["status"] = "Off"
            logger.success("Washer-dryer stopped")

    async def pause(self, msg: dict) -> None:
        """Pause the current program."""
        if not await self._check_action_available("pause"):
            logger.error("Pause action not available in current state")
            return
        success = await self._send_action({"processAction": 3})
        if success:
            self.state["status"] = "Pause"
            logger.success("Washer-dryer paused")

    async def get_status_handler(self, msg: dict) -> dict:
        """Fetch and return current machine state."""
        try:
            raw = await self._get_device_state()
            self._parse_state(raw)
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
        return self.state

    async def get_programs(self, msg: dict) -> dict:
        """Return available programs from the API."""
        token = await self._get_valid_token()
        async with self._session.get(
            f"{MIELE_API_BASE}/devices/{self.device_id}/programs",
            headers=self._auth_headers(token),
        ) as resp:
            if resp.status != 200:
                logger.error(f"Failed to get programs: {resp.status}")
                return {}
            return await resp.json()

    async def enable_mobile_start(self, msg: dict) -> None:
        """Remind the user to enable Mobile Start on the machine panel.
        Cannot be done via API — must be done physically on the appliance.
        """
        logger.warning(
            "Mobile Start cannot be enabled via API. "
            "Please press the Mobile Start button on the machine panel until it lights up yellow."
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._session = aiohttp.ClientSession()
        try:
            await self._discover_device()
            # Initial state fetch
            raw = await self._get_device_state()
            self._parse_state(raw)
            logger.success(f"MieleWasherDryer ready — status: {self.state['status']}")
            # Run state polling alongside base agent tasks
            await asyncio.gather(
                super().run(),
                self._poll_state(),
            )
        except Exception as e:
            logger.error(f"MieleWasherDryer startup failed: {e}")
            raise
        finally:
            await self._session.close()


async def main() -> None:
    agent = MieleWasherDryer()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
