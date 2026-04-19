# Generic Agents

Generic agents are important periphery of HALO, with the implementations of different, specialised agents.
So far we have implemented an agent for:
- Fire TV
- Fetching media ID from title (i.e. show from Netflix, game from Luna)

As we increase functionality, we will want to write new specialist agents using the base provided in BaseAgent.

## Example
```python
from discovery.src.base_agent import BaseAgent
import asyncio

class Example(BaseAgent):
    def __init__(self, name: str) -> None:
        super.__init__(name, "role")

        self.desc = "General description for LLM usage"
        self.handlers = {"command_name_1": self.command1,
                         "command_name_2": self.command2}

        self.state = {"power": "on",
                      "activity": "Sleeping"}

    def command1(self, msg: Any) -> Any:
        pass

    def command2(self, msg: Any) -> Any:
        pass

async def main() -> None:
    agent = Example("name")
    await agent.run()

asyncio.run(main())
```

## Framework:
Agents have a registry of handlers that are exposed to the network every x seconds, where functions accept a dict in the form `{"params": {"param1": val, "param2": val}}` etc.

Commands can return values or just do actions. If a command does return a value, it can be passed on to the next action automatically through the LLM's message chaining ability. Message chaining isn't fully implemented however, so there may still be bugs

## Message passing
Make sure your description and function describe exactly the flow of data and parameters of each command. The LLM is smart and will try to create chains of actions for you. The only messages you should send are messages to the LLM to convey results that appear spontaneously, changes in state, or to request a list of suitable agents for tasks. This is because we want to avoid hardcoded recipients with a dynamic message.

## State
Agents store an arbitrary state dictionary. Store any relevant status of the agent or the devices it controls in here, to be requested or broadcast to the network.