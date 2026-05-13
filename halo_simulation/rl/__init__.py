"""RL integration hooks — headless stepping of HALO scenarios for Gymnasium / SB3."""

from halo_simulation.rl.driver import ACTION_DELTAS, TemperatureRlDriver

__all__ = ["ACTION_DELTAS", "TemperatureRlDriver", "HaloTemperatureRlEnv"]


def __getattr__(name: str):
    if name == "HaloTemperatureRlEnv":
        from halo_simulation.rl.gym_env import HaloTemperatureRlEnv

        return HaloTemperatureRlEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
