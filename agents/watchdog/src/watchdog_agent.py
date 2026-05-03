import asyncio
import docker
from loguru import logger
from discovery.src.base_agent import BaseAgent

class WatchdogAgent(BaseAgent):
    def __init__(self, name="HALO-Watchdog", role="System"):
        super().__init__(name=name, role=role)
        
        # Initialize the Docker client (reads from /var/run/docker.sock)
        try:
            self.docker_client = docker.from_env()
            logger.success(f"[{self.name}] Successfully connected to Host Docker Daemon.")
        except Exception as e:
            logger.error(f"[{self.name}] CRITICAL: Could not connect to Docker socket. Is it mounted? {e}")
            self.docker_client = None

        # Map your HALO Network Names to your Docker Container Names
        self.node_to_container_map = {
            "Claude": "halo-language",
            "TV": "livingroomtv-1",
            "StreamingAggregator": "stream-1"
            # Add other nodes here as you scale!
        }

    async def on_peer_dead(self, peer_name: str) -> None:
        """
        This is the magic. Because Watchdog inherits from BaseAgent, 
        this function fires automatically the second a node stops sending heartbeats.
        """
        logger.warning(f"[{self.name}] 🚨 DETECTED CASUALTY: {peer_name} has flatlined!")
        
        if not self.docker_client:
            logger.error(f"[{self.name}] Cannot heal {peer_name}: Docker client offline.")
            return

        # Find the actual Docker container name for this node
        container_name = None
        for halo_name, docker_name in self.node_to_container_map.items():
            if peer_name.startswith(halo_name): # .startswith handles dynamic IDs (e.g., TV-a1b2c3)
                container_name = docker_name
                break
                
        if not container_name:
            logger.info(f"[{self.name}] No Docker container mapped for '{peer_name}'. Ignoring.")
            return

        # Execute the self-healing restart
        logger.info(f"[{self.name}] 💉 Administering CPR to container '{container_name}'...")
        
        try:
            # We run this in an executor so the blocking Docker HTTP request 
            # doesn't freeze our ZeroMQ heartbeat loop!
            loop = asyncio.get_running_loop()
            container = await loop.run_in_executor(None, self.docker_client.containers.get, container_name)
            await loop.run_in_executor(None, container.restart)
            
            logger.success(f"[{self.name}] ⚡ SHOCK DELIVERED. {container_name} restarted successfully.")
            
            # Optional: Tell Claude what just happened so you get a Telegram notification!
            alert_payload = {
                "action": "send_chat_message",
                "target": "Claude",
                "source": self.name,
                "params": {"message": f"🛠️ <b>Self-Healing Alert</b>\nNode <code>{peer_name}</code> crashed and was automatically rebooted by the Watchdog."}
            }
            import json
            await self.send_msg("Claude", json.dumps(alert_payload))
            
        except docker.errors.NotFound:
            logger.error(f"[{self.name}] Could not find a running container named '{container_name}'!")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to restart {container_name}: {e}")

async def main():
    agent = WatchdogAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())