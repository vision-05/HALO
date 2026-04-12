import asyncio
import socket
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf, AsyncServiceBrowser
import zmq.utils.z85

def get_local_ip():
    """get_local_ip -> None
    Function to get the local IP address on the network (not localhost)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


class HaloServiceListener:
    """HaloServiceListener
       Standard listener that follows ZeroConf implementation"""
    def __init__(self, loop, on_discovered_callback, on_lost_callback):
        self.loop = loop
        self.on_discovered = on_discovered_callback
        self.on_lost = on_lost_callback

    def add_service(self, zc, type_, name):
        asyncio.run_coroutine_threadsafe(self.on_discovered(zc, type_, name), self.loop)

    def remove_service(self, zc, type_, name):
        asyncio.run_coroutine_threadsafe(self.on_lost(zc, type_, name), self.loop)

    def update_service(self, zc, type_, name):
        pass

class Discovery:
    """Discovery"""
    def __init__(self, agent_name, agent_role,  zmq_port, public_key, new_peer_callback):
        self.aiozc = AsyncZeroconf()
        self.browser = None
        self.service_type = "_halo._tcp.local."
        self.local_ip = get_local_ip()

        encoded_key = zmq.utils.z85.encode(public_key)

        self.my_info = AsyncServiceInfo(
            self.service_type,
            f"{agent_name}.{self.service_type}",
            addresses=[socket.inet_aton(self.local_ip)],
            port=zmq_port,
            properties={"role": agent_role,
                        "pubkey": encoded_key},
            server=f"{agent_name}.local.",
        )

        self.new_peer_callback = new_peer_callback
        self.active_peers = {}

    async def start(self):
        await self.aiozc.async_register_service(self.my_info)

        loop = asyncio.get_running_loop()

        listener = HaloServiceListener(loop, self._handle_new_peer, self._handle_lost_peer)

        self.browser = AsyncServiceBrowser(self.aiozc.zeroconf, self.service_type, listener=listener)

    async def _handle_new_peer(self, zc, type_, name):
        if name == self.my_info.name:
            return
        
        info = await self.aiozc.async_get_service_info(type_, name)
        if info:
            ip = socket.inet_ntoa(info.addresses[0])
            port = info.port
            role = info.properties.get(b"role", b"").decode("utf-8")
            pubkey = info.properties.get(b"pubkey", b"")
            decoded_pubkey = zmq.utils.z85.decode(pubkey) if pubkey else None

            peer_data = {"ip": ip, "port": port, "role": role, "pubkey": decoded_pubkey}

            self.active_peers[name] = peer_data        
            self.new_peer_callback(name, peer_data)
        print(f"Added {name}")

    async def _handle_lost_peer(self, name):
        if name in self.active_peers:
            del self.active_peers[name]
            print(f"Lost {name}")

    async def stop(self):
        if self.browser:
            await self.browser.async_cancel()
        await self.aiozc.async_unregister_service(self.my_info)
        await self.aiozc.async_close()

