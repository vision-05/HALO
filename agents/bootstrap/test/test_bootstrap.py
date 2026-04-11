from bootstrap.src.bootstrap import BootstrapAgent
import asyncio
import pytest

@pytest.mark.asyncio
async def test_bootstrap_discovery():
    a1 = BootstrapAgent()
    a2 = BootstrapAgent()

    await a1.broadcast_and_discover()
    await a2.broadcast_and_discover()