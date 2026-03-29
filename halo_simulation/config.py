"""Global simulation configuration — all tunable constants live here."""

# Time
MINUTES_PER_DAY = 1440
DEFAULT_RUN_DAYS = 30
# SimPy ordering: deliver bus messages slightly after env.now so minute-boundary timeouts
# resolve before inbox.put at the same nominal timestamp.
MESSAGE_BUS_SEND_DELAY = 0.001  # simulated minutes (~60 ms wall-clock equivalent)

# Negotiation
MAX_ITERATIONS = 10
CONVERGENCE_THRESHOLD = 0.5
NEGOTIATION_TIMEOUT = 5  # simulated minutes
FALLBACK_TO_UNWEIGHTED_AVERAGE = True

# Agent weights
DEFAULT_COMFORT_WEIGHT = 0.8
AWAY_COMFORT_WEIGHT = 0.2
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

# Specialist broadcast intervals (simulated minutes)
CARBON_BROADCAST_INTERVAL = 30
WEATHER_BROADCAST_INTERVAL = 60

# Carbon spike scenario: force high carbon in evening (minutes from midnight)
CARBON_SPIKE_START_MINUTE = 17 * 60
CARBON_SPIKE_END_MINUTE = 20 * 60
CARBON_SPIKE_INTENSITY = 280  # gCO2/kWh (high band)

# Dishwasher: prefer scheduling after this minute on high-carbon days
DISHWASHER_LOW_CARBON_AFTER_MINUTE = 22 * 60

# Negotiation device longevity pull (small bias toward device optimal operating point)
DEVICE_LONGEVITY_PULL = 0.05

# Preference range for satisfaction score (denominator)
TEMPERATURE_PREFERENCE_RANGE = 14.0  # e.g. span across min-max comfort window
