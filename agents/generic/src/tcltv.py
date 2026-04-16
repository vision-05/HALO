from discovery.src.base_agent import BaseAgent
import subprocess
import asyncio

class TclTv(BaseAgent):
    def __init__(self):
        super().__init__("TV", "Actuator")

        self.tv_ip = "192.168.1.161"
        self.local_state = {}
        self.handlers = {"on": self.turn_onoff, "off": self.turn_onoff}

        subprocess.run(["adb", "connect", self.tv_ip], capture_output=True)

    async def turn_onoff(self):
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "26"])

async def main():
    tv = TclTv()
    asyncio.create_task(tv.broadcast_and_discover())
    asyncio.create_task(tv.heartbeat())
    asyncio.create_task(tv.prune_network())
    await tv.recv_msg()

asyncio.run(main())