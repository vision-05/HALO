import time
import zmq
import zmq.asyncio
import asyncio

from discovery.src.base_agent import BaseAgent

class BootstrapAgent(BaseAgent):
    def __init__(self, name):
       super().__init__(name, "Bootstrap")

    def start_network(self):
        print("starting")

    def gen_uuid(self):
        pass

    def gen_key(self):
        pass