import asyncio
from zeroconf import IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf, AsyncServiceBrowser

class HaloServiceListener:
    def __init__(self, on_discovered_callback, on_lost_callback):
        self.on_discovered = on_discovered_callback
        self.on_lost = on_lost_callback

    def add_service(self, zc, type_, name):
        self.on_discovered(zc, type_, name)

    def remove_service(self, zc, type_, name):
        self.on_lost(zc, type_, name)

    def update_service(self, zc, type_, name):
        pass

class Discovery:
    def __init__(self, agent_name, agent_role, local_ip, zmq_port, new_peer_callback):
        self.aiozc = AsyncZeroconf()
        self.browser = None
        self.service_type = "_halo._tcp.local."

        self.my_info = AsyncServiceInfo(
            self.service_type,
            f"{agent_name}.{self.service_type}",
            addresses=[socket.inet_aton(local_ip)],
            port=zmq_port,
            properties={"role": agent_role},
            server=f"{agent_name}.local.",
        )

        self.new_peer_callback = new_peer_callback
        self.active_peers = {}

    async def start(self):
        pass

    def _handle_new_peer(self, zc, type_, name):
        pass

    def _handle_lost_peer(self, name):
        pass

    async def stop(self):
        pass