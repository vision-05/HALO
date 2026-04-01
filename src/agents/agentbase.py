"""
An agent is a standalone synchronous entity, that has an asynchronous inbound message queue and works
in a network of agents to complete a series of tasks asynchronously
Agent class implements the following:
Constructor: creates process and state for agent tasks
get_features: returns agent features
dispatch: add a task to the agent queue (non-blocking)
negotiate: request a negotiation with other agents (asynchronous lockstep)"""

import threading
import queue
import asyncio

class Agent:
    def __init__(self):
        """Spawn a new process where all agent tasks run on"""
        self.process = threading.Thread(target=self.process, args=(), daemon=True)
        self.queue = queue.PriorityQueue()
        self.process.start()
        self.dummy_state = None

    def process(self):
        while True:
            task, args = self.queue.get()
            task(args)

    def get_features(self):
        print(self.features)

    def dispatch(self, feature, args):
        if feature in self.features:
            self.queue.put((feature, args))

    def negotiate(self, agent, callback):
        pass

    def dummy_print(self, s):
        self.dummy_state = s