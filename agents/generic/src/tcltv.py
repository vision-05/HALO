from discovery.src.base_agent import BaseAgent
import subprocess
import asyncio
import json

import requests
import re
from ddgs import DDGS

def find_netflix_id(show_name: str) -> str:
    """
    Uses the DuckDuckGo Search API to find the official Netflix Show ID.
    This bypasses the anti-bot blocks triggered by raw HTML scraping.
    """
    print(f"[LanguageAgent] Searching the web for Netflix ID: '{show_name}'...")
    
    query = f"site:netflix.com/title {show_name}"
    
    try:
        # Initialize the DDGS client
        search_client = DDGS()
        
        # Fetch up to 5 results to give us options
        results = list(search_client.text(query, max_results=5))
        
        fallback_id = None
            
        for result in results:
            url = result.get("href", "")
            title = result.get("title", "").lower()
            
            # The Regex Pattern: Catch IDs of any length (Netflix uses 7, 8, or 9 digits)
            match = re.search(r'netflix\.com/title/(\d+)', url)
            
            if match:
                show_id = match.group(1)
                
                # Store the very first ID as a fallback just in case
                if not fallback_id:
                    fallback_id = show_id
                    
                # VERIFICATION: Ensure the requested show name is actually in the search result's title!
                # We split into words to handle variations (e.g. "The Office" matching "The Office (U.S.)")
                search_words = show_name.lower().split()
                if all(word in title for word in search_words):
                    print(f"[LanguageAgent] Verified title match! Found ID {show_id} for '{show_name}'")
                    return show_id

        # If no exact match was found, use the first Netflix ID we saw
        if fallback_id:
            print(f"[LanguageAgent] No exact title match. Best guess is ID {fallback_id}")
            return fallback_id

        print(f"[LanguageAgent] Could not find a Netflix ID for '{show_name}'.")
        return None
            
    except Exception as e:
        print(f"[LanguageAgent] Error scraping for ID: {e}")
        return None

class TclTv(BaseAgent):
    def __init__(self):
        super().__init__("TV", "Actuator")

        self.tv_ip = "192.168.1.161"
        self.local_state = {}
        self.handlers = {"power_on": self.turn_onoff,
                         "power_off": self.turn_onoff,
                         "netflix": self.start_netflix,
                         "netflix_play_show_by_name": self.netflix_play_show}

        subprocess.run(["adb", "connect", self.tv_ip], capture_output=True)

    async def turn_onoff(self, msg):
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "26"])

    async def start_netflix(self, msg):
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", "-n", "com.netflix.ninja/.MainActivity"])

    async def netflix_play_show(self, msg):
        show_name = msg["show_name"]

        show_id = find_netflix_id(show_name)
        print(show_id)
        print(f"playing show {show_id}")
        await self.start_netflix(msg)
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", 
            "-n", "com.netflix.ninja/.MainActivity",
            "-a", "android.intent.action.VIEW", 
            "-e", "amzn_deeplink_data", str(show_id)])

    def get_handlers(self):
        return list(self.handlers.keys())
    
    async def expose_handlers(self):
        while True:
            await self.send_msg("Claude", json.dumps({"action": "schema", "TV": self.get_handlers()}))
            await asyncio.sleep(5.0)

async def main():
    tv = TclTv()
    asyncio.create_task(tv.broadcast_and_discover())
    asyncio.create_task(tv.heartbeat())
    asyncio.create_task(tv.prune_network())
    asyncio.create_task(tv.expose_handlers())
    await tv.recv_msg()

asyncio.run(main())