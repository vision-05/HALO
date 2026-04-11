import zmq
import zmq.asyncio
from src.discovery import HaloServiceListener, Discovery
import time
import pytest
import asyncio

ctx = zmq.asyncio.Context()
sock = ctx.socket(zmq.ROUTER)
zport = sock.bind_to_random_port("tcp://*")

ctx1 = zmq.asyncio.Context()
sock1 = ctx1.socket(zmq.ROUTER)
zport1 = sock1.bind_to_random_port("tcp://*")

ctx2 = zmq.asyncio.Context()
sock2 = ctx2.socket(zmq.ROUTER)
zport2 = sock2.bind_to_random_port("tcp://*")

def peer_found(peer_name, peer_data):
    print(f"Peer {peer_name} with data {peer_data}")

@pytest.mark.asyncio
async def test_discovery():
    peer_found_event = asyncio.Event()
    peers = []

    def verify_peer(peername, peerdata):
        print(f"{peername} with {peerdata}")
        peers.append(peername)

        if len(peers) == 6:
            peer_found_event.set()

    s1 = Discovery('test', 'Admin', zport, new_peer_callback=verify_peer)
    s2 = Discovery('test1', 'Admin', zport1, new_peer_callback=verify_peer)
    s3 = Discovery('test2', 'Admin', zport2, new_peer_callback=verify_peer)

    await s1.start()
    await s2.start()
    await s3.start()

    try:
        await asyncio.wait_for(peer_found_event.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        pytest.fail("The test timed out")
    finally:
        await s1.stop()
        await s2.stop()
        await s3.stop()
    
    assert len(peers) == 6