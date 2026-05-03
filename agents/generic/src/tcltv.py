from discovery.src.base_agent import BaseAgent
import subprocess
import asyncio
import json

import requests
import re
from ddgs import DDGS
from urllib.parse import unquote
import time
from loguru import logger

class TclTv(BaseAgent):
    def __init__(self) -> None:
        super().__init__("TV", "Actuator")

        self.tv_ip = "192.168.1.161"
        self.state = {"tv_status": "disconnected",
                      "media": {"application": None,
                                "title": None,
                                "id": None,
                                "status": None}}
        self.register_handlers({"power_on": self.turn_onoff,
                         "power_off": self.turn_onoff,
                         "netflix": self.start_netflix,
                         "netflix_play_show_by_id": self.netflix_play_show,
                         "disney+": self.start_disney,
                         "disney+_play_show_by_id": self.disney_play_show,
                         "luna_play_game_by_id": self.play_luna_game,
                         "pause": self.play_pause,
                         "resume": self.play_pause,
                         "volume_by_percent_level": self.volume_control,
                         "home": self.home,
                         "spotify_play_track_by_id": self.play_spotify_track,
                         "spotify_next_track": self.spotify_next,
                         "spotify_prev_track": self.spotify_prev})

        self.desc = "TV controlling agent, for all commands that involve playing media go through StreamAggregator first to get corresponding ID to title."

        subprocess.run(["adb", "connect", self.tv_ip], capture_output=True)
        self.state["tv_status"] = "connected"
        self.state["power"] = "on"

        res = subprocess.run(["adb", "-s", self.tv_ip, "shell", "dumpsys", "power"],
                             capture_output=True,
                             text=True,
                             timeout=5)
        
        output = res.stdout

        if "mWakefulness=Awake" in output:
            self.state["power"] = "on"
        elif "mWakefulness=Asleep" in output:
            self.state["power"] = "off"
        elif "Display Power: state=ON" in output:
            self.state["power"] = "on"
        else:
            self.state["power"] = "off"

    def sanititze(self, title):
        if title[0] == '[':
            return title[1:-1]
        return title



    async def turn_onoff(self, msg: dict) -> None:
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "26"])
        if self.state["power"] == "on":
            self.state["power"] = "off"
        else:
            self.state["power"] = "on"

    async def volume_control(self, msg: dict) -> None:
        level = msg["params"].get("level", 10)
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "media", "volume", "--stream", "3", "--set", str(level)])
        self.state["volume"] = level

    async def home(self, msg: dict) -> None:
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "3"])
        self.state["media"] = {"application": "home"}

    async def play_pause(self, msg: dict) -> None:
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "85"])
        if self.state["media"]["status"] == "Playing":
            self.state["media"]["status"] = "Paused"
        else:
            self.state["media"]["status"] = "Playing"

    async def start_disney(self, msg: dict) -> None:
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", "-n" "com.disney.disneyplus/com.bamtechmedia.dominguez.main.MainActivity"])

    async def start_netflix(self, msg: dict) -> None:
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", "-n", "com.netflix.ninja/.MainActivity"])

    async def play_spotify_track(self, msg: dict) -> None:
        track_id = msg["params"]["track_id"]
        track_id = self.sanititze(track_id)
        res = subprocess.run(["adb", 
                        "shell", "am", "start", "-a", "android.intent.action.VIEW",
                        "-d", f"spotify:track:{track_id}",
                        "-p", "com.spotify.tv.android"], capture_output=True, text=True)
        
        logger.debug(res)
        
        self.state["media"]["application"] = "Spotify"
        self.state["media"]["id"] = track_id
        self.state["media"]["status"] = "Playing"
        
    async def spotify_next(self, msg: dict) -> None:
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "87"])

    async def spotify_prev(self, msg: dict) -> None:
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "88"])

    async def disney_play_show(self, msg: dict) -> None:
        show_id = msg["params"]["show_id"][0]

        res = subprocess.run(["adb", "-s", self.tv_ip, "shell", "am", "start", 
            "-a", "android.intent.action.VIEW", 
            "-d", f"https://www.disneyplus.com/video/{show_id}", 
            "com.disney.disneyplus"], capture_output=True, text=True)
        logger.debug(str(res))

    async def netflix_play_show(self, msg: dict) -> None:
        show_id = msg["params"]["show_id"]
        show_id = self.sanitize(show_id)
        logger.debug(f"[{self.name}] Initiating {show_id}")

        # Define YOUR exact working ADB command from yesterday
        adb_start_cmd = [
            "adb", "-s", self.tv_ip, "shell", "am", "start", 
            "-f", "0x10000020", # <--- THE MISSING LINK
            "-n", "com.netflix.ninja/.MainActivity",
            "-a", "android.intent.action.VIEW", 
            "-e", "amzn_deeplink_data", str(show_id)
        ]

        # 0. WAKE TV
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "224"])
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "3"])
        await asyncio.sleep(2.0)

        # 1. FIRST STRIKE: Send the intent to wake the app and hit the profile screen
        logger.debug(f"[{self.name}] Waking up Netflix...")
        subprocess.run(adb_start_cmd, capture_output=True)
        
        # Give it time to hit the profile selection screen
        await asyncio.sleep(8.0)
        
        # 2. SELECT PROFILE: Press 'Enter'
        logger.debug(f"[{self.name}] Selecting default profile...")
        subprocess.run(["adb", "-s", self.tv_ip, "shell", "input", "keyevent", "66"])
        
        # Give the profile time to fully load the home catalog
        await asyncio.sleep(8.0)
        
        # 3. SECOND STRIKE: Send your exact deep-link AGAIN.
        # Now that the profile is active, it will actually play the show.
        logger.debug(f"[{self.name}] Sending deep-link to active profile...")
        res = subprocess.run(adb_start_cmd, capture_output=True, text=True)
        
        logger.debug(f"[{self.name}] Final Result: {res}")

    async def play_luna_game(self, data: dict) -> None:
        """
        Attempts to launch a specific game on Amazon Luna.
        Expects data["game_id"] to be the alphanumeric string from a Luna web URL.
        Example: https://luna.amazon.com/game/fortnite/B09M... -> ID is "B09M..."
        """

        game_id = data["params"]["game_id"]
        game_id = self.sanititze(game_id)
    
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

async def main() -> None:
    tv = TclTv()
    await tv.run()

asyncio.run(main())