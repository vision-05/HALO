from agents.language.src.language_agent import CLAUDE_API_KEY
from langchain_openai import ChatOpenAI
import os
import random

"""Class to generate realistic, synthetic data for fine tuning HALO llama prompts.
Creates realistic prompts for direct user requests, indirect user requests (i.e. another agent decided it is laundry time/person is hungry etc).
Creates realistic HALO network configurations, including devices on network, functionality, stored state.
Output to json file to be consumed by the actor -> critic -> improvement system"""
class TaskGenerator:
    def __init__(self, n_examples: int) -> None:
        self.n_examples = n_examples

        self.network_scenarios = []
        self.prompts = {}

        self.llm = ChatOpenAI(
            model='bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0',
            temperature=0.1,
            max_tokens=1024,
            api_key=CLAUDE_API_KEY,
            base_url='https://litellm.prod.outshift.ai/'
        )

    def generate_networks(self, network_count: int) -> None:
        """Generate {network_count} configurations of a HALO network to base the prompts from.
        Provides standard schema for Haiku to conform to when coming up with these configurations"""


        n_devices = random.randint(3,15)

        device_creation_prompt = """"""

        for i in range(n_devices):
            response = self.llm.invoke(device_creation_prompt)
            content = response.content


        for i in range(network_count):
            pass
