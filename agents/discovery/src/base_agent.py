import zmq.asyncio
import zmq
from discovery.src.discovery import Discovery
import asyncio

class BaseAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.ctx = zmq.asyncio.Context()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.curve_server = True
        self.router.curve_secretkey = None #our private key
        self.router.setsockopt_string(zmq.IDENTITY, self.name)
        self.port = self.router.bind_to_random_port("tcp://*")
        self.peers = {}
        self.outbound_socks = {}

        def verify_peer(peername, peerdata):
            clean_name = peername.split('.')[0]
            print(f"{self.name} discovered {peername} at {peerdata['ip']}:{peerdata['port']}")
            self.peers[clean_name] = peerdata
            dealer = self.context.socket(zmq.DEALER)
            self.dealer.curve_server = False
            self.dealer.curve_publickey = None #our public key
            self.dealer.curve_secretkey = None #our private key
            self.dealer.curve_serverkey = None #peer public key
            dealer.connect(f"tcp://{peerdata['ip']}:{peerdata['port']}")
            self.outbound_socks[clean_name] = dealer
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