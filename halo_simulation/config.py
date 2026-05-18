"""Global simulation configuration — all tunable constants live here."""

from __future__ import annotations

import os

# Time
MINUTES_PER_DAY = 1440
DEFAULT_RUN_DAYS = 30
# SimPy ordering: deliver bus messages slightly after env.now so minute-boundary timeouts
# resolve before inbox.put at the same nominal timestamp.
MESSAGE_BUS_SEND_DELAY = 0.001  # simulated minutes (~60 ms wall-clock equivalent)
# HTTP /stream: SimPy env.run is chunked so (a) a client disconnect can stop the worker thread and
# (b) queued SSE events are flushed to the browser frequently. Keep small (2 sim min) so the browser
# sees events within ~2 sim-min of them happening; the sim thread yields the GIL between chunks.
STREAM_STOP_CHECK_CHUNK_MINUTES = 2.0  # simulated minutes per env.run slice
# Optional /stream?demo_wall_seconds=N: sleep after each chunk so the full run lasts ~N wall seconds (human demos).
DEMO_WALL_SECONDS_MAX = 7200.0

# Negotiation
MAX_ITERATIONS = 10
CONVERGENCE_THRESHOLD = 0.5
NEGOTIATION_TIMEOUT = 2  # simulated minutes (auto rounds only; HITL uses indefinite wait when applicable)
FALLBACK_TO_UNWEIGHTED_AVERAGE = True
# Human-in-the-loop: if any of these agent ids are in a negotiation round *and* still marked at home
# in the device's preference snapshot, the device waits for inbox replies only (no sim-minute deadline).
NEGOTIATION_INDEFINITE_WAIT_AGENT_IDS: tuple[str, ...] = ("person_cli",)

# Agent weights
DEFAULT_COMFORT_WEIGHT = 0.8
AWAY_COMFORT_WEIGHT = 0.2
# Applied as ``comfort_weight * (1.0 if home else PRESENCE_MULTIPLIER_AWAY)`` in ``effective_person_weight``.
# Use 0 so away occupants do not pull the weighted comfort target toward their last declared °C (device term still applies).
PRESENCE_MULTIPLIER_AWAY = 0.0
DEFAULT_DEVICE_WEIGHT = 0.4
CARBON_HIGH_THRESHOLD = 250  # gCO2/kWh
CARBON_WEIGHT_BOOST = 0.3

# Learning
EMA_ALPHA = 0.15
ROUTINE_WINDOW_DAYS = 14
ROUTINE_STABLE_STD_MINUTES = 20
ANOMALY_THRESHOLD_MULTIPLIER = 2.0
BAYESIAN_PRIOR_MU = 21.0  # °C
BAYESIAN_PRIOR_SIGMA = 3.0

# Devices
THERMOSTAT_MIN = 14.0
THERMOSTAT_MAX = 28.0
# Shower duration negotiation (simulated minutes); weighted_proposal uses these clip bounds.
SHOWER_DURATION_MIN_MINUTES = 5.0
SHOWER_DURATION_MAX_MINUTES = 25.0
SHOWER_DEVICE_OPTIMAL_MINUTES = 10.0
SHOWER_MINUTES_TOLERANCE = 2.0
TEMPERATURE_TOLERANCE = 1.5  # °C
DEFAULT_FAILURE_PROBABILITY = 0.001
FAILURE_RECOVERY_TIMEOUT = 30  # simulated minutes
MAX_RECOVERY_ATTEMPTS = 3

# Carbon profile (UK baseline, gCO2/kWh by hour)
CARBON_HOURLY_BASELINE = [
    340,
    330,
    320,
    310,
    300,
    290,
    270,
    240,
    210,
    180,
    160,
    140,
    130,
    130,
    140,
    160,
    200,
    260,
    300,
    320,
    330,
    340,
    345,
    342,
]

# Weather
WEATHER_BASELINE_TEMP = 12.0
WEATHER_SUMMER_OFFSET = 8.0
WEATHER_WINTER_OFFSET = -5.0
# When outdoor temperature is at or above this (°C), thermostat applies minimum setpoint (heating off).
OUTDOOR_HEATING_OFF_CELSIUS = 23.0

# Specialist broadcast intervals (simulated minutes)
CARBON_BROADCAST_INTERVAL = 30
WEATHER_BROADCAST_INTERVAL = 60

# Carbon spike scenario: force high carbon in evening (minutes from midnight)
CARBON_SPIKE_START_MINUTE = 17 * 60
CARBON_SPIKE_END_MINUTE = 20 * 60
CARBON_SPIKE_INTENSITY = 280  # gCO2/kWh (high band)

# Dishwasher: prefer scheduling after this minute on high-carbon days
DISHWASHER_LOW_CARBON_AFTER_MINUTE = 22 * 60
# When True, schedule gate calls the LLM (HTTP, can block the sim thread several seconds).
# Default False: use carbon/time heuristic only — same batch still runs once for all pending requesters.
# Enable with env DISHWASHER_USE_LLM_SCHEDULE=true if you want JSON approve/defer from the model.
DISHWASHER_USE_LLM_SCHEDULE = os.getenv("DISHWASHER_USE_LLM_SCHEDULE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# After schedule gate approves, optionally negotiate ``dishwasher_delay`` with requesters (extra rounds).
# Default off for speed and so CLI human-in-the-loop is never stuck (CliPersonAgent does not auto-reply).
# Set DISHWASHER_USE_DELAY_NEGOTIATION=true to enable; dishwasher rounds never use indefinite sim-wait.
DISHWASHER_USE_DELAY_NEGOTIATION = os.getenv("DISHWASHER_USE_DELAY_NEGOTIATION", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)
# Person auto-accept band when proposed delay is within this many minutes of “as soon as possible” (0).
DISHWASHER_DELAY_NEGOTIATION_TOLERANCE_MIN = 25.0
# Clamp LLM / heuristic / negotiation deferral (simulated minutes from approval until physical start).
DISHWASHER_DEFER_MINUTES_MAX = 180.0
# Heuristic high-carbon path: minutes-to-22:00 is capped so the sim does not sit hundreds of sim minutes in Wait.
DISHWASHER_DEFER_DIRTY_GRID_CAP_MINUTES = 90.0
# After LLM returns approve=false, re-run scheduling this often (sim minutes), in addition to carbon broadcasts.
DISHWASHER_DECLINED_RETRY_SIM_MINUTES = 15.0
# If the oldest pending request is at least this many sim minutes old, skip further LLM declines and use the
# heuristic schedule (always approves with a deferral) so runs do not stall for days of simulated time.
DISHWASHER_APPROVE_FALSE_OVERRIDE_AFTER_SIM_MIN = 90.0

# Hot water tank (shower scenarios): normalized 0–1 fraction in agent state UI
HOT_WATER_DRAIN_PER_SHOWER = 0.35
HOT_WATER_RECHARGE_PER_MINUTE_BASE = 0.009
HOT_WATER_RECHARGE_GRID_CLEAN_MULTIPLIER = 2.0

# Fused: auto preheat when tank is low and grid carbon is below CARBON_HIGH_THRESHOLD.
# The shower agent also tries this immediately after a shower ends (tank often drops below trigger),
# in addition to the periodic poll below.
FUSED_AUTO_PREHEAT_POLL_MINUTES = 22.0
FUSED_AUTO_PREHEAT_TANK_TRIGGER = 0.48
FUSED_AUTO_PREHEAT_TANK_TARGET = 0.88
FUSED_AUTO_PREHEAT_FILL_PER_MINUTE = 0.045
FUSED_AUTO_PREHEAT_BLOCK_NOTICE_COOLDOWN_MIN = 55.0

# Negotiation device longevity pull (small bias toward device optimal operating point)
DEVICE_LONGEVITY_PULL = 0.05

# Preference range for satisfaction score (denominator)
TEMPERATURE_PREFERENCE_RANGE = 14.0  # e.g. span across min-max comfort window


def anthropic_api_key() -> str:
    """API key for Claude / Anthropic Messages API (LLM specialist). First non-empty wins."""
    for name in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "CLAUDE_KEY"):
        v = os.getenv(name, "").strip()
        if v:
            return v
    return ""


# LLM gateway (Anthropic direct vs LiteLLM / corporate proxy — same Messages JSON shape)
DEFAULT_LLM_MODEL = "claude-haiku-4-5-20251001"


def llm_protocol() -> str:
    """
    ``anthropic`` — POST ``/v1/messages`` (Anthropic request/response shape).

    ``openai`` — POST ``/v1/chat/completions`` (OpenAI shape; typical **LiteLLM** surface — use with
    ``LITELLM_BASE_URL`` + Bearer virtual key).
    """
    p = os.getenv("LLM_PROTOCOL", os.getenv("LLM_API", "anthropic")).strip().lower()
    if p in ("openai", "openai_chat", "chat_completions", "litellm", "litellm_openai"):
        return "openai"
    return "anthropic"


def llm_model() -> str:
    """Model id sent to POST …/v1/messages (override if your gateway maps names)."""
    m = os.getenv("LLM_MODEL", os.getenv("ANTHROPIC_MODEL", DEFAULT_LLM_MODEL)).strip()
    return m or DEFAULT_LLM_MODEL


def llm_anthropic_messages_url() -> str:
    """
    Full URL for Anthropic-compatible POST /v1/messages.

    - Default: ``https://api.anthropic.com/v1/messages``
    - LiteLLM: set ``LLM_ANTHROPIC_BASE_URL=https://your-proxy`` (origin only) → ``…/v1/messages``
    - Or set ``LLM_MESSAGES_URL`` to the exact endpoint if your gateway uses a non-standard path.
    """
    full = os.getenv("LLM_MESSAGES_URL", "").strip()
    if full:
        return full.rstrip("/")
    base = os.getenv(
        "LLM_ANTHROPIC_BASE_URL",
        os.getenv("LITELLM_BASE_URL", "https://api.anthropic.com"),
    ).strip().rstrip("/")
    return f"{base}/v1/messages"


def llm_openai_chat_url() -> str:
    """
    Full URL for OpenAI-compatible POST ``/v1/chat/completions`` (LiteLLM default).

    Set ``LITELLM_BASE_URL`` or ``LLM_ANTHROPIC_BASE_URL`` to the proxy origin, or
    ``LLM_OPENAI_CHAT_URL`` for a non-standard path.
    """
    full = os.getenv("LLM_OPENAI_CHAT_URL", "").strip()
    if full:
        return full.rstrip("/")
    base = os.getenv("LITELLM_BASE_URL", os.getenv("LLM_ANTHROPIC_BASE_URL", "")).strip().rstrip("/")
    if not base:
        return ""
    return f"{base}/v1/chat/completions"


def llm_messages_request_headers(api_key: str) -> dict[str, str]:
    """
    Headers for Anthropic Messages API shape.

    - ``LLM_AUTH_STYLE`` unset or ``x-api-key``: direct Anthropic (``x-api-key`` + ``anthropic-version``).
    - ``LLM_AUTH_STYLE=bearer``: ``Authorization: Bearer <key>`` (common for LiteLLM virtual keys).
    """
    key = (api_key or "").strip()
    style = os.getenv("LLM_AUTH_STYLE", "x-api-key").strip().lower()
    headers: dict[str, str] = {"content-type": "application/json"}
    if style in ("bearer", "authorization", "litellm"):
        headers["authorization"] = f"Bearer {key}"
    else:
        headers["x-api-key"] = key
        ver = os.getenv("ANTHROPIC_VERSION", "2023-06-01").strip() or "2023-06-01"
        headers["anthropic-version"] = ver
    return headers


def llm_request_headers(api_key: str, *, protocol: str) -> dict[str, str]:
    """HTTP headers for the active LLM protocol (Anthropic Messages vs OpenAI chat)."""
    if protocol == "openai":
        key = (api_key or "").strip()
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }
    return llm_messages_request_headers(api_key)
