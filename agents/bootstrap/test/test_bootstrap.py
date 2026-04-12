from bootstrap.src.bootstrap import BootstrapAgent
from discovery.src.base_agent import BaseAgent
import asyncio
import pytest

@pytest.mark.asyncio
async def test_bootstrap():
    boot = BootstrapAgent("B")
    p1 = BaseAgent("T", "Occupant")
    p2 = BaseAgent("V", "Occupant")

    boot.start_network()
    boot.broadcast_and_discover()