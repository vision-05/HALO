from discovery.src.base_agent import BaseAgent
from telegram import Bot

import requests
import re

def find_netflix_id(show_name: str) -> str:
    """
    Scrapes DuckDuckGo to find the official Netflix Show ID.
    We use DuckDuckGo's basic HTML version because it does not block simple Python scripts like Google does.
    """
    print(f"[LanguageAgent] Searching the web for Netflix ID: '{show_name}'...")
    
    # Force the search engine to only return official Netflix title pages
    query = f"site:netflix.com/title {show_name}"
    url = f"https://html.duckduckgo.com/html/?q={query}"
    
    # We must fake a web browser so DuckDuckGo doesn't reject the request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=5)
        
        # The Regex Pattern: Look for the exact URL structure and grab the 8 numbers at the end
        match = re.search(r'netflix\.com/title/(\d{8})', response.text)
        
        if match:
            show_id = match.group(1)
            print(f"[LanguageAgent] Found ID {show_id} for '{show_name}'!")
            return show_id
        else:
            print(f"[LanguageAgent] Could not find a Netflix ID for '{show_name}'.")
            return None
            
    except Exception as e:
        print(f"[LanguageAgent] Error scraping for ID: {e}")
        return None


class LanguageAgent(BaseAgent):
    def __init__(self, telegram_token, admin_chat_id):
        super().__init__("Claude", "Language")
        self.telegram_token = telegram_token
        self.admin_chat_id = admin_chat_id

        self.bot = Bot(telegram_token)

        self.pending_peers = {}
        self.handlers = {"schema": self.get_peer_schema}
        self.schemas = {}

    async def verification_prompt(self, peername, peerdata):
        clean_name = peername.split('.')[0]
        self.pending_peers[clean_name] = peerdata
        
        message = (
            f"🌐 <b>New HALO Node Discovered</b>\n"
            f"Name: {clean_name}\n"
            f"IP: {peerdata['ip']}\n\n"
            f"Reply with <code>/accept {clean_name}</code> to pair."
        )

        await self.bot.send_message(
            chat_id = self.admin_chat_id,
            text=message,
            parse_mode="HTML"
        )

    def get_peer_info(self):
        return list(self.peers.keys())

    def get_peer_schema(self, msg):
        schemas = list(msg.keys())
        self.schemas[schemas[1]] = msg[schemas[1]]
        print(self.schemas[schemas[1]])
