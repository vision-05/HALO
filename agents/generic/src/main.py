from discovery.src.base_agent import BaseAgent
import asyncio
import uuid
import os

AGENT_NAME = os.environ.get("AGENT_NAME", None)
ROLE = os.environ.get("AGENT_ROLE", None)

container_id = str(uuid.uuid1())

async def main() -> None:
    a = BaseAgent(AGENT_NAME+container_id, ROLE)
    await a.run()


asyncio.run(main())