import time
import zmq
import zmq.asyncio
import asyncio
import uuid

from discovery.src.base_agent import BaseAgent

class BootstrapAgent(BaseAgent):
    def __init__(self, name):
       super().__init__(name, "Bootstrap")

    def start_network(self):
        self.gen_uuid()

    def gen_uuid(self):
        self.network_UUID = uuid.uuid1()
