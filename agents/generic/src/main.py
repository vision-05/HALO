from discovery.src.base_agent import BaseAgent
import asyncio
import uuid

container_id = str(uuid.uuid1())

async def main():
    livingroomlight = BaseAgent("LivingRoomLight"+container_id, "Actuator")

    asyncio.create_task(livingroomlight.broadcast_and_discover())

    asyncio.create_task(livingroomlight.heartbeat())
    asyncio.create_task(livingroomlight.prune_network())
    
    await livingroomlight.recv_msg()


asyncio.run(main())