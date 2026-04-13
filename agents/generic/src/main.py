from discovery.src.base_agent import BaseAgent
import asyncio

async def main():
    livingroomlight = BaseAgent("LivingRoomLight", "Actuator")
    carbonagg = BaseAgent("CarbonAggregator", "Aggregator")
    thermostat = BaseAgent("Thermostat", "Actuator")
    fridgechecker = BaseAgent("FridgeBot", "Actuator")
    groceryshopper = BaseAgent("GroceryAggregator", "Aggregator")

    await asyncio.gather(livingroomlight.broadcast_and_discover(), carbonagg.broadcast_and_discover(), thermostat.broadcast_and_discover(),
                         fridgechecker.broadcast_and_discover(), groceryshopper.broadcast_and_discover())
    
    await livingroomlight.recv_msg()
    await carbonagg.recv_msg()
    await thermostat.recv_msg()
    await fridgechecker.recv_msg()
    await groceryshopper.recv_msg()

asyncio.run(main())