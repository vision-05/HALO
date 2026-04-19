import zmq.asyncio
import zmq
from discovery.src.discovery import Discovery
import asyncio
import os
import time
import json
import inspect
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import sys

class BaseAgent:
    """Base Agent implementation"""
    def __init__(self, name: str, role: str) -> None:
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
        self.handlers = {}
        self.desc = ""

        self.service = Discovery(name, role, self.port, self.public_key, new_peer_callback=self.verify_peer)
        self.network_UUID = None

        logger.add(sys.stderr, format="{time} {level} {message}", filter=self.name, level="SUCCESS")

    def verify_peer(self, peername: str, peerdata: Dict[str, Any]) -> None:
        asyncio.create_task(self.verification_prompt(peername, peerdata))

    async def run(self) -> None:
        asyncio.create_task(self.broadcast_and_discover())
        asyncio.create_task(self.heartbeat())
        asyncio.create_task(self.prune_network())
        asyncio.create_task(self.expose_handlers())
        await self.recv_msg()

    async def verification_prompt(self, peername: str, peerdata: Dict[str, Any]) -> None:
        clean_name = peername.split('.')[0]
        logger.debug(f"{self.name} discovered {peername} at {peerdata['ip']}:{peerdata['port']}")
        #acc_input = input(f"{self.name} Do you accept the connection to {clean_name}? [y/n] ")
        #if acc_input == "n": #comment out in testing
        #    return
        self.connect_peer(clean_name, peerdata)
        
    def connect_peer(self, clean_name: str, peerdata: Dict[str, Any]) -> None:
        logger.debug("Connecting...")
        if clean_name in self.outbound_socks:
            old_socket = self.outbound_socks.pop(clean_name)
            old_socket.close(linger = 0)

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
    
    async def expose_handlers(self) -> None:
        while True:
            await self.send_msg("Claude", json.dumps({"action": "schema", self.name: self.get_handlers()}))
            await asyncio.sleep(5.0)

    async def broadcast_and_discover(self) -> None:
        """Broadcast agent and wait 5 seconds to discover other agents\n
        Stop broadcasting and discovering (change this later)"""
        await self.service.start()

    async def stop_broadcasting(self) -> None:
        await self.service.stop()

    async def broadcast_state(self) -> None:
        for dealer in self.outbound_socks:
            payload = str(self.state).encode('utf-8')
            await dealer.send(payload)
            await asyncio.sleep(10.0)

    async def heartbeat(self) -> None:
        while True:
            for dealer in self.outbound_socks.values():
                payload = "heartbeat".encode('utf-8')
                await dealer.send(payload)
        
            await asyncio.sleep(1.0)

    async def prune_network(self) -> None: #check heartbeat count
        while True:
            if len(self.heartbeats) < 1:
                await asyncio.sleep(0.5)
                continue

            min_node = min(self.heartbeats, key=self.heartbeats.get)
            max_node = max(self.heartbeats, key=self.heartbeats.get)

            if self.heartbeats[min_node] < time.time() - 5:
                logger.warning(f"Pruned {min_node}")
                del self.heartbeats[min_node]
                del self.outbound_socks[min_node]
                del self.peers[min_node]

            await asyncio.sleep(0.5)

    async def send_msg(self, dest: str, payload: str) -> None:
        """Fetch the dealer corresponding to the destination agent and send the message"""
        dealer = self.outbound_socks.get(dest)
        if not dealer:
            return
        
        payload = payload.encode('utf-8')
        await dealer.send(payload)

    def inject_wildcards(self, res: Any, new_msg: Union[Dict, List, str, Any]) -> Union[Dict, List, str, Any]:
        if not isinstance(res, (list, tuple)):
            res = [res]

        if isinstance(new_msg, dict):
            return {key: self.inject_wildcards(res, value) for key, value in new_msg.items()}

        elif isinstance(new_msg, list):
            return [self.inject_wildcards(res, item) for item in new_msg]

        elif isinstance(new_msg, str):
            idx = new_msg.find("$*")
            if idx != -1:
              return new_msg.replace("$*", str(res))
            else:
                return new_msg
        else:
            return new_msg

    async def recv_msg(self) -> None:
        """Receive messages from the network, running constantly for agent lifetime. Encodes input string to utf-8 for sending to other agents"""
        while True:
            frames = await self.router.recv_multipart()
            sender_id = frames[0].decode('utf-8')

            if sender_id not in self.pubkey_lookup.keys():
                continue

            message_data = frames[1]

            if frames[1] == b"heartbeat":
                self.heartbeats[sender_id] = time.time()
                continue

            try:
                data = json.loads(frames[1].decode('utf-8'))
                if data.get("action",None) != "schema":
                    logger.debug(data)
                if hasattr(self, "handlers"):
                    action = self.handlers.get(data["action"], None)
                    if action is not None:
                        if inspect.iscoroutinefunction(action):
                            await action(data)
                        else:
                            res = action(data)
                            next_act = data.get("on_success", None)
                            if next_act is not None:
                                injected = self.inject_wildcards(res, next_act)
                                logger.debug(injected)
                                logger.debug("Sending new message")
                                await self.send_msg(injected["target"], json.dumps(injected))
            except json.JSONDecodeError:
                logger.error("Failed decode")

            logger.debug(f"{self.name} received {message_data} from {sender_id}")

    def get_handlers(self) -> List[str]:
        return [f"DescriptionStart: {self.desc} DescriptionEnd "] + list(self.handlers.keys())

    def gen_key(self, key_dir: str = './.keys') -> None:
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
            logger.success(f"{self.name} loaded identiy from disk")
            return
        
        logger.debug(f"Generating")

        self.public_key, self.secret_key = zmq.curve_keypair()

        with open(public_path, "wb") as f:
            f.write(self.public_key)

        with open(secret_path, "wb") as f:
            f.write(self.secret_key)

        logger.success(f"Generated keypair")

        try:
            os.chmod(secret_path, 0o600) #private key file perms
        except Exception:
            logger.error("Failed ot set permissions")