import zmq.asyncio
import zmq
from discovery.src.discovery import Discovery
import asyncio
import os

class BaseAgent:
    def __init__(self, name, role):
        self.name = name
        self.role = role

        self.gen_key()

        self.ctx = zmq.asyncio.Context()
        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.curve_server = True
        self.router.curve_secretkey = self.secret_key
        self.router.setsockopt_string(zmq.IDENTITY, self.name)
        self.port = self.router.bind_to_random_port("tcp://*")
        self.peers = {}
        self.outbound_socks = {}

        def verify_peer(peername, peerdata):
            clean_name = peername.split('.')[0]
            print(f"{self.name} discovered {peername} at {peerdata['ip']}:{peerdata['port']}")
            self.peers[clean_name] = peerdata
            dealer = self.ctx.socket(zmq.DEALER)
            dealer.setsockopt_string(zmq.IDENTITY, self.name)
            dealer.curve_server = False
            dealer.curve_publickey = self.public_key
            dealer.curve_secretkey = self.secret_key
            dealer.curve_serverkey = peerdata['pubkey']
            dealer.connect(f"tcp://{peerdata['ip']}:{peerdata['port']}")
            self.outbound_socks[clean_name] = dealer
            self.peer_found_event.set()

        self.service = Discovery(name, role, self.port, self.public_key, new_peer_callback=verify_peer)
        self.network_UUID = None

        self.peer_found_event = asyncio.Event()

    async def broadcast_and_discover(self):
        await self.service.start()

        try:
            await asyncio.wait_for(self.peer_found_event.wait(), 5.0)
        except asyncio.TimeoutError:
            print("No agents detected in 5s")
        finally:
            await self.service.stop()

    async def send_msg(self, dest, payload):
        dealer = self.outbound_socks.get(dest)
        if not dealer:
            return
        
        await dealer.send(payload)
        print("Sent message")

    async def recv_msg(self):
        while True:
            frames = await self.router.recv_multipart()
            sender_id = frames[0].decode('utf-8')

            message_data = frames[1]

            print(f"{self.name} received {message_data} from {sender_id}")

    def gen_key(self, key_dir='./.keys'):
        os.makedirs(key_dir, exist_ok=True)

        public_path = os.path.join(key_dir, f"{self.name}_public_key")
        secret_path = os.path.join(key_dir, f"{self.name}_secret_key")

        if os.path.exists(public_path) and os.path.exists(secret_path):
            with open(public_path, "rb") as f:
                self.public_key = f.read()
            with open(secret_path, "rb") as f:
                self.secret_key = f.read()
            print(f"{self.name} loaded identiy from disk")
            return
        
        print(f"Generating")

        self.public_key, self.secret_key = zmq.curve_keypair()

        with open(public_path, "wb") as f:
            f.write(self.public_key)

        with open(secret_path, "wb") as f:
            f.write(self.secret_key)

        print(f"Generated keypair")

        try:
            os.chmod(secret_path, 0o600) #private key file perms
        except:
            Exception