from discovery.src.base_agent import BaseAgent
import subprocess
import asyncio
import json

import requests
import re
from ddgs import DDGS
from urllib.parse import unquote
import time

def find_luna_id(game_name: str) -> str:
    """
    Uses the DuckDuckGo Search API (HTML backend) to find the official Amazon Luna Game ID.
    Extracts the 10-character Amazon ASIN from the end of the URL.
    """
    print(f"[AggregatorAgent] Searching the web for Luna Game ID: '{game_name}'...")
    
    # Restrict the search specifically to Luna's game directory
    query = f"site:luna.amazon.com/game {game_name}"
    
    for attempt in range(3):
        try:
            search_client = DDGS()
            
            # Use backend="html" to bypass the strict API rate limit
            results = list(search_client.text(query, max_results=5, backend="html"))
            
            fallback_id = None
                
            for result in results:
                # Decode DuckDuckGo tracking proxy links
                url = unquote(result.get("href", ""))
                title = result.get("title", "").lower()
                
                # THE REGEX: Looks for luna.amazon.com/game/[game-name-slug]/[10-CHARACTER-ASIN]
                # We use {8,12} just in case Amazon ever slightly alters their ASIN length for digital goods
                match = re.search(r'luna\.amazon\.com/game/.*?/([a-zA-Z0-9]{8,12})(?:[/?#]|$)', url)
                
                if match:
                    game_id = match.group(1)
                    
                    # Store the very first ID as a fallback
                    if not fallback_id:
                        fallback_id = game_id
                        
                    # VERIFICATION: Strip punctuation so "Assassin's Creed" matches "assassins creed"
                    search_words = [re.sub(r'[^\w]', '', w) for w in game_name.lower().split()]
                    search_words = [w for w in search_words if w] 
                    
                    if all(word in title for word in search_words):
                        print(f"[AggregatorAgent] Verified title match! Found ID {game_id} for '{game_name}'")
                        return game_id

            # If no exact title match was found, use the first ID we saw
            if fallback_id:
                print(f"[AggregatorAgent] No exact title match. Best guess is ID {fallback_id}")
                return fallback_id

            print(f"[AggregatorAgent] Could not find a Luna ID for '{game_name}'.")
            return None
            
        except Exception as e:
            # If rate-limited, wait and retry
            wait_time = 2 * (attempt + 1)
            print(f"[AggregatorAgent] Search rate limit hit. Retrying in {wait_time} seconds... (Error: {e})")
            time.sleep(wait_time)
            
    print(f"[AggregatorAgent] Failed to find Luna ID after 3 attempts.")
    return None

def find_disney_id(show_name: str) -> str:
    """
    Uses the DuckDuckGo Search API to find the official Disney+ ID.
    Extracts the alphanumeric hash at the end of the URL.
    """
    print(f"[LanguageAgent] Searching the web for Disney+ ID: '{show_name}'...")
    
    # We restrict the search to the main Disney+ domain
    query = f"site:disneyplus.com {show_name}"
    
    import time
    for attempt in range(3):
        try:
            # Initialize the DDGS client
            search_client = DDGS()
            
            # THE FIX: Use backend="html" to bypass the strict API rate limit
            results = list(search_client.text(query, max_results=5, backend="html"))
            
            fallback_id = None
                
            for result in results:
                url = unquote(result.get("href", ""))
                title = result.get("title", "").lower()
            
            # THE FIX 2: Upgraded Regex
            # - Allows any number of path segments before the ID using (?:[^/]+/)*
            # - Forces the ID to be the absolute last part of the URL, safely ignoring ?query=strings
            match = re.search(r'disneyplus\.com/.*?(?:series|movies|video)/(?:[^/]+/)*([a-zA-Z0-9]{8,})(?:[/?#]|$)', url)
            
            if match:
                show_id = match.group(1)
                
                # Store the very first ID as a fallback just in case
                if not fallback_id:
                    fallback_id = show_id
                    
                # VERIFICATION: Strip punctuation so "Avatar:" doesn't fail against "Avatar"
                search_words = [re.sub(r'[^\w]', '', w) for w in show_name.lower().split()]
                search_words = [w for w in search_words if w] # Remove any empty strings
                
            # If no exact title match was found, use the first ID we saw
            if fallback_id:
                print(f"[LanguageAgent] No exact title match. Best guess is ID {fallback_id}")
                return fallback_id

            print(f"[LanguageAgent] Could not find a Disney+ ID for '{show_name}'.")
            return None
            
        except Exception as e:
            # If we get rate-limited, wait a few seconds and try again
            wait_time = 2 * (attempt + 1)
            print(f"[LanguageAgent] DuckDuckGo rate limit hit. Retrying in {wait_time} seconds... (Error: {e})")
            time.sleep(wait_time)
            
    print(f"[LanguageAgent] Failed to find Disney+ ID after 3 attempts.")
    return None
    
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
                         "netflix_play_show_by_name": self.netflix_play_show,
                         "disney+": self.start_disney,
                         "disney+_play_show_by_name": self.disney_play_show,
                         "luna_play_game_by_name": self.play_luna_game}

        subprocess.run(["adb", "connect", self.tv_ip], capture_output=True)

    async def turn_onoff(self, msg):
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "26"])

    async def start_disney(self, msg):
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", "-n" "com.disney.disneyplus/com.bamtechmedia.dominguez.main.MainActivity"])

    async def start_netflix(self, msg):
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", "-n", "com.netflix.ninja/.MainActivity"])

    async def disney_play_show(self, msg):
        show_name = msg["show_name"]
        show_id = find_disney_id(show_name)

        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", 
            "-a", "android.intent.action.VIEW", 
            "-d", f"https://www.disneyplus.com/video/{show_id}", 
            "com.disney.disneyplus"])

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
        
    async def play_luna_game(self, data):
        """
        Attempts to launch a specific game on Amazon Luna.
        Expects data["game_id"] to be the alphanumeric string from a Luna web URL.
        Example: https://luna.amazon.com/game/fortnite/B09M... -> ID is "B09M..."
        """

        game_name = data.get("game_name")
        game_id = find_luna_id(game_name)
    
        if not game_id:
            print(f"[{self.name}] Error: play_luna_game requires a game_id")
            return
        
        print(f"[{self.name}] Attempting to Deep Link Amazon Luna Game ID: {game_id}...", flush=True)
    
        # THE FIX: Using Amazon's internal codename for Luna
        LUNA_PACKAGE = "com.amazon.spiderpork"
    
        # 1. Kill Luna for a clean slate
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "force-stop", LUNA_PACKAGE])
        await asyncio.sleep(1.0) 
    
        # --- METHOD 1: The Native Web URI (Most common for Android TV) ---
        # Cloud gaming apps often intercept standard web URLs.
        print(f"[{self.name}] Trying Method 1: Web URI Intent...", flush=True)
        subprocess.run([
            "adb", "-s", self.tv_ip, "shell", "am", "start", 
            "-a", "android.intent.action.VIEW", 
            "-d", f"https://luna.amazon.com/game/{game_id}", 
            LUNA_PACKAGE
        ])
    
        await asyncio.sleep(5.0)
    
        # If the app didn't open the game, it might just be on the home screen.
        # --- METHOD 2: The Fire OS amzn_deeplink_data Extra ---
        # Because Luna is an Amazon product on an Amazon OS, it likely uses the same 
        # secret handshake that Netflix uses on Fire TV.
        print(f"[{self.name}] Firing Method 2 fallback (amzn_deeplink_data)...", flush=True)
        subprocess.run([
            "adb", "-s", self.tv_ip, "shell", "am", "start", 
            "-a", "android.intent.action.VIEW", 
            "-e", "amzn_deeplink_data", str(game_id),
            LUNA_PACKAGE
        ])
    
        # --- METHOD 3: The Profile Bypass (Standard procedure) ---
        # If Luna has a "Who's playing?" screen like Netflix/Disney, 
        # we need to simulate the Enter key to push past it.
        print(f"[{self.name}] Waiting for potential profile screen...", flush=True)
        await asyncio.sleep(8.0)
    
        print(f"[{self.name}] Sending Enter keyevent to clear UI...", flush=True)
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "66"])
    
        # Some cloud games require a second "Play" confirmation click
        await asyncio.sleep(3.0)
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "66"])
    
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