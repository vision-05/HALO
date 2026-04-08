import zmq
import zmq.asyncio
from src.discovery import HaloServiceListener, Discovery, get_local_ip

ctx = zmq.asyncio.Context()
sock = ctx.socket(zmq.ROUTER)
zport = sock.bind_to_random_port("tcp://*")

ctx1 = zmq.asyncio.Context()
sock1 = ctx1.socket(zmq.ROUTER)
zport1 = sock1.bind_to_random_port("tcp://*")

def peer_found(peer_name, peer_data):
    print(f"Peer {peer_name} with data {peer_data}")

Discovery('test', 'Admin', get_local_ip(), zport, new_peer_callback=peer_found)
Discovery('test1', 'Admin', get_local_ip(), zport1, new_peer_callback=peer_found)