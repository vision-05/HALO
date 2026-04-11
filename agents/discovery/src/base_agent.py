import zmq.asyncio
import zmq
from discovery.src.discovery import Discovery
import asyncio

class BaseAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.ctx = zmq.asyncio.Context()
        self.sock = self.ctx.socket(zmq.ROUTER)
        self.sock.setsockopt_string(zmq.IDENTITY, self.name)
        self.port = self.sock.bind_to_random_port("tcp://*")
        self.peers = {}

        def verify_peer(peername, peerdata):
            clean_name = peername.split('.')[0]
            print(f"{self.name} discovered {peername} at {peerdata['ip']}:{peerdata['port']}")
            self.peers[clean_name] = peerdata
            self.peer_found_event.set()

        self.service = Discovery(name, role, self.port, new_peer_callback=verify_peer)

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
            await self.service.stop()

    async def send_msg(self, dest, payload):
        peer = self.peers.get(dest)
        print(self.peers)
        if not peer:
            return
        
        addr = f"tcp://{peer['ip']}:{peer['port']}"

        self.sock.connect(addr)
        await asyncio.sleep(0.1)
        self.sock.send_multipart([dest.encode('utf-8'), payload])
        print("Sent message")

    async def recv_msg(self):
        while True:
            frames = await self.sock.recv_multipart()
            sender_id = frames[0].decode('utf-8')

            message_data = frames[1]

            print(f"{self.name} received {message_data} from {sender_id}")