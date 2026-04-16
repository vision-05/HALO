from bootstrap import BootstrapAgent
import asyncio

async def main():
    b = BootstrapAgent("Boot")
    b.start_network()

    asyncio.create_task(b.broadcast_and_discover())
    asyncio.create_task(b.heartbeat())
    asyncio.create_task(b.prune_network())

    await b.recv_msg()

asyncio.run(main())