import zmq
import zmq.asyncio
from discovery.src.discovery import HaloServiceListener, Discovery
from discovery.src.base_agent import BaseAgent
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

@pytest.mark.asyncio
async def test_base_discovery():
    a1 = BaseAgent("B1", "Boot")
    a2 = BaseAgent("B2", "Client")

    await asyncio.gather(a1.broadcast_and_discover(), a2.broadcast_and_discover())

    assert len(a1.peers) == len(a2.peers)

@pytest.mark.asyncio
async def test_base_messaging():
    a1 = BaseAgent("A1", "Admin")
    a2 = BaseAgent("A2", "Admin")

    await asyncio.gather(a1.broadcast_and_discover(), a2.broadcast_and_discover())

    asyncio.create_task(a1.recv_msg())
    asyncio.create_task(a2.recv_msg())

    await a1.send_msg("A2", b"Hello from A1")

    await asyncio.sleep(1.0)
