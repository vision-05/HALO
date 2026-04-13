from discovery.src.base_agent import BaseAgent
from telegram import Bot

class LanguageAgent(BaseAgent):
    def __init__(self, telegram_token, admin_chat_id):
        super().__init__("Claude", "Language")
        self.telegram_token = telegram_token
        self.admin_chat_id = admin_chat_id

        self.bot = Bot(telegram_token)

        self.pending_peers = {}

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
