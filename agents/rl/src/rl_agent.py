from __future__ import annotations

import asyncio
import datetime as dt
from importlib import import_module
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))


try:
	BaseAgent = import_module("discovery.src.base_agent").BaseAgent
except Exception:
	# Lightweight fallback for test environments where the full discovery agent
	# (and its heavy dependencies like pyzmq) are not installed. Tests create
	# RoutineRLAgent instances via object.__new__ and patch methods, so this
	# stub only needs to provide a minimal surface area.
	class _NoopScheduler:
		def add_job(self, *args, **kwargs):
			return None
		def start(self):
			return None

	class BaseAgent:  # pragma: no cover - test fallback
		def __init__(self, name: str, role: str) -> None:
			self.name = name
			self.role = role
			self.state = {}
			self.handlers = {}
			self.outbound_socks = {}
			self.scheduler = _NoopScheduler()

		def register_handlers(self, handlers):
			self.handlers.update(handlers)

		async def broadcast_and_discover(self):
			return None

		async def heartbeat(self):
			return None

		async def prune_network(self):
			return None

		async def expose_handlers(self):
			return None

		async def backup(self):
			return None

		async def recv_msg(self):
			return None

		async def send_msg(self, dest: str, payload: str) -> None:
			return None

		def get_handlers(self):
			return list(self.handlers.keys())
from RL.PPO import DEFAULT_MODEL_PATH, DEFAULT_PROFILE_PATH, load_or_train_model, predict_next_action
from RL.environment import ACTIONS, SmartHomeEnv
from RL.online_update import DEFAULT_ACTIVITY_LOG_PATH, run_monthly_update


class RoutineRLAgent(BaseAgent):
	"""PPO-backed routing agent that predicts a smart-home action every 15 minutes."""

	def __init__(
		self,
		name: str = "RLAgent",
		role: str = "Planner",
		profile_path: Path | str = DEFAULT_PROFILE_PATH,
		model_path: Path | str = DEFAULT_MODEL_PATH,
		activity_log_path: Path | str = DEFAULT_ACTIVITY_LOG_PATH,
		prediction_interval_minutes: int = 15,
		monthly_window_days: int = 30,
		monthly_timesteps: int = 2_000,
		monthly_learning_rate: float | None = 0.00005,
		bootstrap_timesteps: int = 25_000,
		seed: int = 42,
	) -> None:
		super().__init__(name, role)

		self.profile_path = Path(profile_path)
		self.model_path = Path(model_path)
		self.activity_log_path = Path(activity_log_path)
		self.prediction_interval_minutes = max(1, int(prediction_interval_minutes))
		self.monthly_window_days = max(1, int(monthly_window_days))
		self.monthly_timesteps = max(1, int(monthly_timesteps))
		self.monthly_learning_rate = monthly_learning_rate
		self.bootstrap_timesteps = max(1, int(bootstrap_timesteps))
		self.seed = int(seed)
		self._startup_prediction_delay = 10
		self._last_prediction_payload: Optional[dict[str, Any]] = None

		self.activity_log_path.parent.mkdir(parents=True, exist_ok=True)
		self.model_path.parent.mkdir(parents=True, exist_ok=True)
		self.profile_path.parent.mkdir(parents=True, exist_ok=True)
		if not self.profile_path.exists():
			fallback_profile = {}
			if DEFAULT_PROFILE_PATH.exists() and DEFAULT_PROFILE_PATH != self.profile_path:
				try:
					fallback_profile = json.loads(DEFAULT_PROFILE_PATH.read_text(encoding="utf-8"))
				except Exception:
					fallback_profile = {}
			self.profile_path.write_text(json.dumps(fallback_profile, indent=2), encoding="utf-8")

		self.env = SmartHomeEnv(self.profile_path)
		self.model = load_or_train_model(
			profile_path=self.profile_path,
			model_path=self.model_path,
			total_timesteps=self.bootstrap_timesteps,
			seed=self.seed,
			run_check=False,
		)

		self.desc = (
			"DescriptionStart: PPO ROUTING PLANNER. Uses a trained routine profile to predict a smart-home action "
			"every 15 minutes, then forwards that prediction to LanguageAgent as a schema-compliant routing prompt. "
			"DescriptionEnd"
		)

		self.register_handlers(
			{
				"predict_next_action": self.predict_next_action_handler,
				"route_prediction_now": self.route_prediction_handler,
				"refresh_monthly_profile": self.refresh_monthly_profile_handler,
				"get_last_routing_payload": self.get_last_routing_payload,
			}
		)

		self.scheduler.add_job(
			self.route_prediction,
			trigger="interval",
			minutes=self.prediction_interval_minutes,
			next_run_time=dt.datetime.now() + dt.timedelta(seconds=self._startup_prediction_delay),
			id=f"{self.name}:route_prediction",
			coalesce=True,
			max_instances=1,
		)
		self.scheduler.add_job(
			self.refresh_monthly_profile,
			trigger="interval",
			days=30,
			id=f"{self.name}:monthly_refresh",
			coalesce=True,
			max_instances=1,
		)

	def _json_safe(self, value: Any) -> Any:
		if isinstance(value, dict):
			return {str(key): self._json_safe(val) for key, val in value.items()}
		if isinstance(value, list):
			return [self._json_safe(item) for item in value]
		if isinstance(value, tuple):
			return [self._json_safe(item) for item in value]
		if isinstance(value, (str, int, float, bool)) or value is None:
			return value
		return str(value)

	def _safe_float(self, value: Any, default: float) -> float:
		try:
			if value is None:
				return default
			if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "off"}:
				return default
			return float(value)
		except (TypeError, ValueError):
			return default

	def _safe_int(self, value: Any, default: int, low: int, high: int) -> int:
		try:
			numeric = int(float(value))
		except (TypeError, ValueError):
			return default
		return min(high, max(low, numeric))

	def _resolve_context(self) -> dict[str, Any]:
		state_context = self.state.get("routine_context", {}) if isinstance(self.state.get("routine_context"), dict) else {}
		merged: dict[str, Any] = {**state_context}
		for key in ("current_time_hours", "day_name", "season", "rain_intensity", "low_stock_count", "leave_hour", "return_hour", "movie_hour", "walk_hour", "shopping_hour"):
			if key in self.state:
				merged[key] = self.state[key]

		current_time_hours = self._safe_float(merged.get("current_time_hours"), 17.5)
		day_name = str(merged.get("day_name", "Monday"))
		season = str(merged.get("season", "Winter"))
		rain_intensity = min(1.0, max(0.0, self._safe_float(merged.get("rain_intensity"), 0.0)))
		low_stock_count = self._safe_int(merged.get("low_stock_count"), 0, 0, 4)

		return {
			"current_time_hours": current_time_hours,
			"day_name": day_name,
			"season": season,
			"rain_intensity": rain_intensity,
			"low_stock_count": low_stock_count,
			"leave_hour": self._safe_float(merged.get("leave_hour"), self.env.leave_hour),
			"return_hour": self._safe_float(merged.get("return_hour"), self.env.return_hour),
			"movie_hour": self._safe_float(merged.get("movie_hour"), self.env.movie_hour),
			"walk_hour": self._safe_float(merged.get("walk_hour"), self.env.walk_hour),
			"shopping_hour": self._safe_float(merged.get("shopping_hour"), self.env.shopping_hour),
		}

	def _predict_action(self) -> tuple[int, str, dict[str, Any], list[float]]:
		context = self._resolve_context()
		observation = self.env.build_observation_from_state(
			current_time_hours=context["current_time_hours"],
			day_name=context["day_name"],
			season=context["season"],
			rain_intensity=context["rain_intensity"],
			low_stock_count=context["low_stock_count"],
		)
		action_index, action_name = predict_next_action(
			self.model,
			self.env,
			state=(context["current_time_hours"], context["day_name"], context["season"], context["rain_intensity"], context["low_stock_count"]),
		)
		return action_index, action_name, context, observation.tolist()

	def _build_llm_request(self, action_index: int, action_name: str, context: dict[str, Any]) -> dict[str, Any]:
		instruction = (
			"You are the HALO routing layer. A PPO planner predicted a household action and now you must convert it "
			"into a valid HALO network_payload that routes the request to the correct person or device agent. "
			"Return ONLY valid JSON in the LanguageAgent schema. Use on_success if you need to chain additional actions. "
			"Do not invent unsupported fields.\n\n"
			f"Predicted action index: {action_index}\n"
			f"Predicted action: {action_name}\n"
			f"Source agent: {self.name}\n"
			f"Current context: {json.dumps(context, indent=2, default=str)}"
		)
		return {
			"action": "self_prompt",
			"source": self.name,
			"target": "LanguageAgent",
			"params": {
				"instruction": instruction,
				"predicted_action_index": action_index,
				"predicted_action": action_name,
				"current_context": context,
				"model_path": str(self.model_path),
				"profile_path": str(self.profile_path),
				"schedule": "every_15_minutes",
				"available_actions": ACTIONS,
			},
		}

	def _append_activity_log(self, payload: dict[str, Any]) -> None:
		self.activity_log_path.parent.mkdir(parents=True, exist_ok=True)
		with self.activity_log_path.open("a", encoding="utf-8") as file:
			file.write(json.dumps(payload, default=str) + "\n")

	async def route_prediction(self) -> dict[str, Any]:
		action_index, action_name, context, observation = self._predict_action()
		payload = self._build_llm_request(action_index, action_name, context)
		record = {
			"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
			"agent_name": self.name,
			"action_index": action_index,
			"action_name": action_name,
			"context": self._json_safe(context),
			"observation": observation,
			"message": payload,
		}
		self._last_prediction_payload = record
		self.state["last_rl_prediction"] = record
		self._append_activity_log(record)

		if "LanguageAgent" not in self.outbound_socks:
			print(f"[{self.name}] LanguageAgent is not connected yet; prediction will be logged locally.")
		else:
			await self.send_msg("LanguageAgent", json.dumps(payload))
		return record

	async def refresh_monthly_profile(self) -> dict[str, Any]:
		result = await asyncio.to_thread(
			run_monthly_update,
			log_path=self.activity_log_path,
			profile_path=self.profile_path,
			model_path=self.model_path,
			window_days=self.monthly_window_days,
			online_timesteps=self.monthly_timesteps,
			learning_rate=self.monthly_learning_rate,
			seed=self.seed,
		)
		self.env = SmartHomeEnv(self.profile_path)
		self.model = result["model"]
		self.state["last_monthly_refresh"] = {
			"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
			"profile_path": str(self.profile_path),
			"model_path": str(self.model_path),
			"sample_window_days": self.monthly_window_days,
		}
		return result

	async def predict_next_action_handler(self, msg: dict) -> dict[str, Any]:
		action_index, action_name, context, observation = self._predict_action()
		return {"action_index": action_index, "action_name": action_name, "context": context, "observation": observation, "available_actions": ACTIONS}

	async def route_prediction_handler(self, msg: dict) -> dict[str, Any]:
		return await self.route_prediction()

	async def refresh_monthly_profile_handler(self, msg: dict) -> dict[str, Any]:
		return await self.refresh_monthly_profile()

	def get_last_routing_payload(self, msg: dict) -> dict[str, Any]:
		return self._last_prediction_payload or {}

	async def run(self) -> None:
		self.scheduler.start()
		tasks = [self.broadcast_and_discover(), self.heartbeat(), self.prune_network(), self.expose_handlers(), self.backup(), self.recv_msg()]
		await asyncio.gather(*tasks)

