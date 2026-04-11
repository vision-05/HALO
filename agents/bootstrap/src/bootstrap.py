import time
import zmq
import zmq.asyncio
import asyncio

from discovery.src.discovery import Discovery

class BootstrapAgent:
    def __init__(self):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.port = self.sock.bind_to_random_port("tcp://*")
        self.service = Discovery("Bootstrap", "Bootstrap", self.port, new_peer_callback=self.start_network)

        self.public_key = None
        self.private_key = None
        self.network_UUID = None

        self.peer_found_event = asyncio.Event()

    async def broadcast_and_discover(self):
        await self.service.start()

        try:
            await asyncio.wait_for(self.peer_found_event.wait(), 5.0)
        except TimeoutError:
            print("No agents detected in 5s")
        finally:
            self.service.stop()

    def start_network(self, pname, pdata):
        print(f"Discovered {pname} with {pdata}")
        self.peer_found_event.set()

    def gen_uuid(self):
        pass

    def gen_key(self):
        pass