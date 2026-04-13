from bootstrap import BootstrapAgent
import asyncio

async def main():
    b = BootstrapAgent("Boot")

    await b.broadcast_and_discover()

    
    await b.recv_msg()

asyncio.run(main())