from discovery.src.base_agent import BaseAgent
import os
import asyncio
from loguru import logger

a_name = os.environ.get("AGENT_NAME")

class Advocator(BaseAgent):
    def __init__(self):
        super().__init__(a_name, "Advocator")
        self.register_handlers({"set_user_telegram_id": self.set_user_telegram,
                                "update_user_with_new_state": self.update_user_state})
        
        self.desc = f"Agent that advocates for user {a_name}" + ", user update_user_state for this user to store preferences of any kind, each with a very specific key, using nested maps starting with the key {state_update: {movie_preference: {genre: horror, amount: 80%, times: morning}}}, when anything resembling a preference is mentioned. This agent does not have access to telegram do not send messages through it. Only save state. "

    def set_user_telegram(self, msg):
        self.state["tel_user_id"] = msg["params"]["tel_user_id"]

    def update_user_state(self, msg):
        self.state.update(msg["params"]["state_update"])

async def main():
    ag = Advocator()
    await ag.run()

asyncio.run(main())