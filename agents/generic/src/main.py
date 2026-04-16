from discovery.src.base_agent import BaseAgent
import asyncio
import uuid
import os

AGENT_NAME = os.environ.get("AGENT_NAME", None)
ROLE = os.environ.get("AGENT_ROLE", None)

container_id = str(uuid.uuid1())

async def main():
    a = BaseAgent(AGENT_NAME+container_id, ROLE)

    asyncio.create_task(a.broadcast_and_discover())

    asyncio.create_task(a.heartbeat())
    asyncio.create_task(a.prune_network())
    
    await a.recv_msg()


asyncio.run(main())