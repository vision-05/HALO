# Quickstart Guide

This document is a brief introduction to making your own HALO agents.

A HALO agent is a self contained program that completes tasks on the HALO network. They are all derived from the HALO [`BaseAgent`](api/base_agent.md) class, and they talk to each other using multicast DNS or mDNS contained in their `self.discovery` object. The specific details are not so important.

## Running agents
We have chosen to use docker-compose to run each agent. An agent runs in its own container as specified in the services list. For example:  
- Language Agent: docker compose up language
- Stream browser (for netflix, disney+, spotify, luna fetching): docker compose up stream
- TV: docker compose up livingroomtv
- Advocator: docker compose up advocator
- Watchdog (network self-healing): docker compose up halo-watchdog
- Bootstrap (for network initialisation): docker compose up bootstrap

Some of these require environment variables:  
- Language agent
 - CLAUDE_KEY
 - TELEGRAM_KEY
 - TEL_CHAT_ID
 - LANG_MODE (LOCAL/CLAUDE)
- Advocator
 - AGENT_NAME
 
You can put these in a `.env` file in the HALO folder.


## Writing agents
All agents have to start up, fetch their state, connect to other nodes on the network and begin their heartbeat. This is completed by the `run()` method in the BaseAgent class. Note that this is an asynchronous method so it must be awaited in an async function.  
```python
import asyncio
from discovery.src.base_agent import BaseAgent

class DerivedAgent(BaseAgent):
    def __init__(self):
        super().__init__("name", "role")
        pass #fill in constructor
        
    def function(self): #program logic
        pass
        

async def main(): #async main function
    agent = DerivedAgent() #create agent class
    await agent.run() #run the setup function
    
if __name__ == "__main__":
    asyncio.run(main()) #run the startup sequence
    
```

### State
Agent state is stored in the `self.state` field. By executing the `run()` method the agent automatically reads the existing state file, or creates a new one. It then periodically writes to the state file. The file is a json file, and state is stored as a map such as
```python
{"power_status": "on", "media": {"service": "netflix", "title": "Stranger Things"}}
```

To update the state, call the `update` method, i.e.
```python
self.state.update("key", "value")
```

### Methods and Handlers
Anything functions the agent does, that you want exposed to the network, are stored as handlers. These live in the `self.handlers` registry. This registry is periodically sent to the entire network to tell the other agents what your agent is capable of doing. To add handlers to the registry, call `self.register_handlers({"h1": self.fn1, "h2": self.fn2})` in the constructor.
NEVER modify the `self.handlers` map in any other way.

The key of the handler is a descriptive name of what the function does, containing parameter information and return type if necessary. There will be a refactor to make the actual parameters and return value description registered separately, but until then, the key you provide is all that a language agent has to understand your function.

Any method must take a single dictionary parameter `msg` and can have any or no return type. The `msg` dict is a standard HALO network message, and you can extract information on the source of the message, as well as the parameters.

Here is the typical HALO `msg` schema:
```python
{"action": "fetch_current_deals_by_postcode",
 "source": "LanguageAgent",
 "target": "Food",
 "params": {"postcode": "EC1 1EE"},
 "on_success": {"action": "send_chat_msg",
                "source": "Food",
                "target": "LanguageAgent",
                "params": {"user_msg": "Found the following deals: $*"}}}
```

Note the wildcard `$*` in the `on-success` dict. The result of the first action is pasted in place of the wildcard once the function has completed. This allows for the chaining of commands through a simple message passing system.

Here is what registering handlers would look like:
```python
class FoodAgent(BaseAgent):
    def __init__(self):
        super().__init__("Food", "Aggregator")
        
        self.register_handlers({"fetch_current_deals_by_postcode": self.get_deals,
                                "get_takeaway_menu_by_name": self.get_menu})
    def get_deals(self, msg: dict) -> list:
        pass
        
    def get_menu(self, msg: dict) -> list:
        pass
```

### Passive agents

### Active agents
An active agent is an agent that sends messages directly to the network, usually for ongoing tasks. An example might be an agent that detects suspicious people on the security camera.
