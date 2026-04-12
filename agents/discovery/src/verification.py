from base_agent import BaseAgent
import asyncio

async def test():
    a1 = BaseAgent("A1", "Admin")
    a2 = BaseAgent("A2", "Occupant")
    a3 = BaseAgent("A3", "Occupant")

    await asyncio.gather(a1.broadcast_and_discover(), a2.broadcast_and_discover(), a3.broadcast_and_discover())

    await asyncio.sleep(3.0)

    asyncio.create_task(a1.recv_msg())
    asyncio.create_task(a2.recv_msg())
    asyncio.create_task(a3.recv_msg())


    await a1.send_msg("A2", b"Hello from A1")
    await a3.send_msg("A2", b"HEllo bad")

    await asyncio.sleep(1.0)

    await asyncio.gather(a1.stop_broadcasting(), a2.stop_broadcasting(), a3.stop_broadcasting())

asyncio.run(test())