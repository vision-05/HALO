from bootstrap.src.bootstrap import BootstrapAgent
import asyncio
import pytest

@pytest.mark.asyncio
async def test_bootstrap_discovery():
    a1 = BootstrapAgent("B1")
    a2 = BootstrapAgent("B2")

    await asyncio.gather(a1.broadcast_and_discover(), a2.broadcast_and_discover())

