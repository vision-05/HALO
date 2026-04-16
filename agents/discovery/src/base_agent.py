import zmq.asyncio
import zmq
from discovery.src.discovery import Discovery
import asyncio
import os
import time

class BaseAgent:
    """Base Agent implementation"""
    def __init__(self, name, role):
        """Generates/loads public/private key pair for this agent\n
        Creates router socket (for receiving messages)\n
        Binds to random port\n
        Includes lambda callback for discovering new peers"""
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
        self.pubkey_lookup = {}
        self.state = {}
        self.heartbeats = {}

        self.service = Discovery(name, role, self.port, self.public_key, new_peer_callback=self.verify_peer)
        self.network_UUID = None

    def verify_peer(self, peername, peerdata):
        asyncio.create_task(self.verification_prompt(peername, peerdata))

    async def verification_prompt(self, peername, peerdata):
        clean_name = peername.split('.')[0]
        print(f"{self.name} discovered {peername} at {peerdata['ip']}:{peerdata['port']}")
        #acc_input = input(f"{self.name} Do you accept the connection to {clean_name}? [y/n] ")
        #if acc_input == "n": #comment out in testing
        #    return
        self.connect_peer(clean_name, peerdata)
        
    def connect_peer(self, clean_name, peerdata):
        print("Connecting...")
        self.peers[clean_name] = peerdata
        dealer = self.ctx.socket(zmq.DEALER)
        dealer.setsockopt_string(zmq.IDENTITY, self.name)
        dealer.curve_server = False
        dealer.curve_publickey = self.public_key
        dealer.curve_secretkey = self.secret_key
        self.pubkey_lookup[clean_name] = peerdata['pubkey']
        dealer.curve_serverkey = peerdata['pubkey']
        dealer.connect(f"tcp://{peerdata['ip']}:{peerdata['port']}")
        self.outbound_socks[clean_name] = dealer

    async def broadcast_and_discover(self):
        """Broadcast agent and wait 5 seconds to discover other agents\n
        Stop broadcasting and discovering (change this later)"""
        await self.service.start()

    async def stop_broadcasting(self):
        await self.service.stop()

    async def broadcast_state(self):
        for dealer in self.outbound_socks:
            payload = str(self.state).encode('utf-8')
            await dealer.send(payload)
            await asyncio.sleep(10.0)

    async def heartbeat(self):
        while True:
            for dealer in self.outbound_socks.values():
                payload = "heartbeat".encode('utf-8')
                await dealer.send(payload)
        
            await asyncio.sleep(1.0)

    async def prune_network(self): #check heartbeat count
        while True:
            print(self.heartbeats)
            if len(self.heartbeats) < 1:
                await asyncio.sleep(0.5)
                continue

            min_node = min(self.heartbeats, key=self.heartbeats.get)
            max_node = max(self.heartbeats, key=self.heartbeats.get)

            print(self.heartbeats[max_node] - self.heartbeats[min_node])

            if self.heartbeats[min_node] < time.time() - 3:
                print(f"pruning {min_node}")
                del self.heartbeats[min_node]
                del self.outbound_socks[min_node]
                del self.peers[min_node]

            await asyncio.sleep(0.5)

    async def send_msg(self, dest, payload):
        """Fetch the dealer corresponding to the destination agent and send the message"""
        dealer = self.outbound_socks.get(dest)
        if not dealer:
            return
        
        payload = payload.encode('utf-8')
        await dealer.send(payload)
        print("Sent message")

    async def recv_msg(self):
        """Receive messages from the network, running constantly for agent lifetime. Encodes input string to utf-8 for sending to other agents"""
        while True:
            frames = await self.router.recv_multipart()
            sender_id = frames[0].decode('utf-8')

            if sender_id not in self.pubkey_lookup.keys():
                print("Dropped unauthorised packet")
                continue

            message_data = frames[1]

            print(frames[1])

            if frames[1] == b"heartbeat":
                self.heartbeats[sender_id] = time.time()

            print(f"{self.name} received {message_data} from {sender_id}")

    def gen_key(self, key_dir='./.keys'):
        """Check whether public/private key pair already exists for this agent on disk\n
        Loads keys if on disk, otherwise generates new pair and writes to file"""
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