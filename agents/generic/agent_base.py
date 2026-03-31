#Base agent class exposes:


class BaseAgent:
    def __init__(self):
        self.negotiation_protocol = None
        self.HALO_protocol = None

    def heartbeat(self):
        pass

    def recv(self):
        pass

    def send(self):
        pass
