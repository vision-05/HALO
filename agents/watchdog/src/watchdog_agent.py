import asyncio
import docker
import json
from loguru import logger
from discovery.src.base_agent import BaseAgent

class WatchdogAgent(BaseAgent):
    def __init__(self, name="HALO-Watchdog", role="System"):
        super().__init__(name=name, role=role)
        
        try:
            self.docker_client = docker.from_env()
            logger.success(f"[{self.name}] Successfully connected to Host Docker Daemon.")
        except Exception as e:
            logger.error(f"[{self.name}] CRITICAL: Could not connect to Docker socket. Is it mounted? {e}")
            self.docker_client = None

        # Map HALO Network Names to partial Docker container names
        self.node_to_container_map = {
            "Claude": "language",
            "TV": "tv",
            "StreamingAggregator": "stream" # Fuzzy match: will catch 'halo-stream-1' or 'stream_aggregator'
        }

    async def on_peer_dead(self, peer_name: str) -> None:
        logger.warning(f"[{self.name}] 🚨 DETECTED CASUALTY: {peer_name} has flatlined!")
        
        if not self.docker_client:
            logger.error(f"[{self.name}] Cannot heal {peer_name}: Docker client offline.")
            return

        container_hint = None
        for halo_name, docker_name in self.node_to_container_map.items():
            if peer_name.startswith(halo_name): 
                container_hint = docker_name
                break
                
        if not container_hint:
            logger.info(f"[{self.name}] No Docker mapping for '{peer_name}'. Ignoring.")
            return

        logger.info(f"[{self.name}] 🔍 Scanning Docker for any container matching '{container_hint}'...")
        
        try:
            loop = asyncio.get_running_loop()
            
            # THE FIX 1: Grab ALL containers, including completely shut down/exited ones!
            all_containers = await loop.run_in_executor(
                None, lambda: self.docker_client.containers.list(all=True)
            )

            target_container = None
            for c in all_containers:
                # THE FIX 2: Fuzzy matching bypasses Docker Compose's dynamic prefixes
                if container_hint in c.name:
                    target_container = c
                    break

            if not target_container:
                logger.error(f"[{self.name}] ❌ Could not find ANY container (running or stopped) matching '{container_hint}'!")
                return

            logger.info(f"[{self.name}] 💉 Found '{target_container.name}' (Status: {target_container.status}). Administering CPR...")

            # Execute the self-healing restart
            await loop.run_in_executor(None, target_container.restart)
            
            logger.success(f"[{self.name}] ⚡ SHOCK DELIVERED. {target_container.name} revived.")
            
            # Alert Claude
            alert_payload = {
                "action": "send_chat_message",
                "target": "Claude",
                "source": self.name,
                "params": {"message": f"🛠️ <b>Self-Healing Alert</b>\nNode <code>{peer_name}</code> crashed and was automatically revived by the Watchdog."}
            }
            await self.send_msg("Claude", json.dumps(alert_payload))
            
        except Exception as e:
            logger.error(f"[{self.name}] Failed CPR: {e}")

async def main():
    agent = WatchdogAgent()
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())