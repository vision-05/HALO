from discovery.src.base_agent import BaseAgent

from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler

import asyncio
import json
import typing
import uuid
import requests
import re
import os
from loguru import logger
import aiohttp
from ollama import AsyncClient
CLAUDE_API_KEY = os.environ['CLAUDE_KEY']

class LanguageAgent(BaseAgent):
    def __init__(self, name, role):
        super().__init__(name, role)

        self.llm = AsyncClient(host='http://127.0.0.1:11434')

        self.sys_prompt = ""

    async def self_prompt(self, msg: dict):
        logger.debug(f"Message: {msg}")
        instruction = msg["params"].get("instruction", None)
        if instruction is None:
            instruction = msg["params"].get("prompt", None) 
        if instruction is None:
            instruction = msg["params"].get("task", None)
        if not instruction:
            instruction = str(msg["params"])
        logger.debug(f"======={instruction}")
        network_context = {
            "your_name": self.name,
            "your_capabilities": list(self.handlers.keys()),
            "connected_agents": self.get_peer_info(),
            "available_device_schemas": self.schemas
        }

        context_string = json.dumps(network_context, indent=2)

        response = await self.llm.chat(
            model='phi4-mini',
            format='json',
            messages=[
                {
                    'role': 'system',
                    'content': self.sys_prompt + f"\n\n### CURRENT NETWORK STATE ###\n{context_string}"
                },
                {
                    'role': 'system',
                    'content': instruction # (or 'instruction' if in self_prompt)
                }
            ])
        response = response["message"]["content"]
        if response[0] == "`":
            response = response[7:-3]
        human_resp = json.loads(response)
        net_commands = human_resp.get("network_payload", None)
        commands = net_commands

        if not isinstance(net_commands, List):
            commands = [net_commands]

        if commands[0] is None:
            return

        target = commands[0].get("target", None)
        if target is not None:
            for command in commands:
                await self.send_msg(command["target"], json.dumps(command))


class TelegramAgent(LanguageAgent):
    def __init__(self, telegram_token: str, admin_chat_id: str) -> None:
        super().__init__("LanguageAgent", "Language")
        self.telegram_token = telegram_token
        self.admin_chat_id = admin_chat_id

        self.bot = Bot(telegram_token)
        self.desc = "DescriptionStart: MASTER ROUTER. Use this agent to send chat messages to the user, evaluate complex logic, or trigger a self_prompt. DescriptionEnd"

        self.pending_peers = {}
        self.register_handlers({"schema": self.get_peer_schema,
                         "send_chat_message": self.send_message_to_user,
                         "self_prompt": self.self_prompt})
        self.schemas = {}

        self.route_prompt = """You are the Triage Router for the HALO smart home network.
Your ONLY job is to determine if a user's request requires a single action or a complex, multi-step chain of actions.

RULES FOR COMPLEXITY:
1. CROSS-AGENT DEPENDENCY: If the task requires getting data from one agent to pass to another (e.g., playing media requires fetching an ID from StreamingAggregator FIRST, then sending it to TV), it is COMPLEX.
2. STATE VERIFICATION: If the task requires fetching a state key before acting, it is COMPLEX.
3. SINGLE ACTION: If the task requires only one command sent to one agent, it is NOT complex.
4. DO NOT OVER-ENGINEER: You are strictly FORBIDDEN from using `upsert_state`, `get_state_keys`, or `fetch_state_by_keys` to pass data between agents. Data is passed directly using wildcards.

You MUST think step-by-step before making your decision. Output ONLY valid JSON matching this exact schema:
{{
  "required_actions": ["list", "the", "exact", "action", "names", "you", "will", "need"],
  "reasoning": "Explain step-by-step how these actions map to the available agents and if they trigger the complexity rules.",
  "complex_task": true_or_false
}}"""

        self.sys_prompt = """You are the HALO Orchestration formatting engine.
Your ONLY job is to format a chained network payload using the EXACT execution plan provided.

CRITICAL RULES:
1. You MUST use the `$*` wildcard in the on_success params to pass the ID to the TV.
2. DO NOT invent new keys like "next_step". Stick EXACTLY to the JSON template below.

{
  "telegram_reply": "Your friendly, human-readable message to the user. This is where you exhibit your personality. Be decisive, fun, and helpful. If asked to solve a social conundrum (e.g., who cleans the house), suggest a fun game or activity here.",
  "network_payload": {
    "action": "the_actuation_command",
    "delay": 0.5,
    "target": "TargetAgentName",
    "source": "LanguageAgent",
    "params": {"param1": "value1"},
    "on_success": {
      "action": "next_action",
      "target": "NextTargetName",
      "time": "Scheduled time strictly as '%b %d %Y %I:%M%p' OR omit if immediate",
      "source": "LanguageAgent",
      "params": {"result_data": "$*"}
    },
    "on_failure": {
      "action": "fallback_action",
      "target": "LanguageAgent",
      "source": "LanguageAgent",
      "params": {"error_data": "$*"}
    }
  }
}

NETWORK RULES & BEHAVIORS:
1. DELAYS: For immediate actions, set "delay": 0.5. To schedule, use the exact datetime format '%b %d %Y %I:%M%p' (e.g., 'Oct 25 2026 03:30PM') in the "time" key.
2. DUMB AGENTS: All other agents on the network are strictly functional. Do not send conversational text to them. Use exact data keys. 
3. SCHEMA DISCOVERY: Always use 'get_state_schema' chained before fetching data so you know exactly which keys exist in an agent's state. 
4. WILDCARDS ($*): Use the wildcard "$*" in the "params" of "on_success" or "on_failure" to pass the dynamic result of the current action into the next action.
5. SELF-PROMPTING: If you need to evaluate data from another agent, send an action to that agent and use "on_success" to trigger a "self_prompt" back to yourself (target "LanguageAgent"). Inject the "$*" wildcard into the self-prompt so you receive the data.
6. ENDING CHAINS: If you have finished an analysis or created a final plan based on data, put your final recommendations in "telegram_reply", send the message, and DO NOT trigger another self-prompt. 
7. STATE UPDATES: To persist useful data to the network state, the action is "receive_state" and the params MUST be structured with the "new_state" wrapper: {"new_state": {"key1": "value1"}}. Do not attempt to fetch a key named 'state_update'. """

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        await update.message.reply_html(rf"Hi {user.mention_html()}!",
                                       reply_markup = ForceReply(selective=True))
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        current_chat_id = update.effective_chat.id
        await update.message.reply_text(f"Help! The ID for this chat is: {current_chat_id}")

    async def handle_location(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Catches Telegram location pins and reverse-geocodes them."""
        lat = update.message.location.latitude
        lon = update.message.location.longitude
        
        processing_msg = await update.message.reply_text("📍 Pinning your location...")
        
        # Free Reverse Geocoding via OpenStreetMap (No API Key Required!)
        try:
            headers = {"User-Agent": "HALO-SmartHome/1.0"}
            url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    data = await response.json()
                    address = data.get("address", {})
                    
                    # Extract the most useful local identifiers
                    borough = address.get("borough") or address.get("city_district", "")
                    city = address.get("city") or address.get("town") or address.get("village", "")
                    neighborhood = address.get("neighbourhood") or address.get("suburb", "")
                    
                    # Format a beautiful string like "Camden Town, London"
                    if neighborhood and city:
                        self.state["weather_area"] = city[8:] if city.startswith("Greater") else city
                        self.state["user_location"] = f"{neighborhood}, {city}"
                    else:
                        self.state["user_location"] = city or f"Lat: {lat}, Lon: {lon}"
                        
            await processing_msg.edit_text(f"✅ Location synced to: **{self.user_location}**.\n\nYou can now ask me to find food or weather nearby!")
            
        except Exception as e:
            logger.error(f"Geocoding failed: {e}")
            self.user_location = f"Lat: {lat}, Lon: {lon}"
            await processing_msg.edit_text(f"✅ GPS Coordinates saved.")

    async def respond(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        is_group = update.message.chat.type in ['group', 'supergroup']
        bot_username = (await context.bot.get_me()).username
        
        text = update.message.text

        logger.debug("Responding")
        
        if is_group:
            # If in a group, ONLY respond if the bot is @mentioned
            if f"@{bot_username}" not in text:
                return # Ignore normal human-to-human group chatter
            
            # Strip the mention out so Claude just gets the raw command
            text = text.replace(f"@{bot_username}", "").strip()

        # 1. Build a clean dictionary of the current network reality
        network_context = {
            "your_name": self.name,
            "your_capabilities": list(self.handlers.keys()),
            "connected_agents": self.get_peer_info(),
            "available_device_schemas": self.schemas
        }

        # 2. Convert it to a pretty JSON string
        context_string = json.dumps(network_context, indent=2)

        triage_schema = {
            "type": "object",
            "properties": {
                "required_actions": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "reasoning": {"type": "string"},
                "complex_task": {"type": "boolean"}
            },
            "required": ["required_actions", "reasoning", "complex_task"]
        }

        # 3. Inject it cleanly into the prompt with clear boundaries
        response = await self.llm.chat(
            model = 'phi4-mini',
            format=triage_schema,
            messages=[
                {
                    'role': 'system',
                    'content': self.route_prompt + f"\n\n### CURRENT NETWORK STATE ###\n{context_string}",
                },
                {
                    'role': 'user',
                    'context': text
                }
            ])
        
        logger.debug(response)

        msg = json.loads(response["message"]["content"])

        actions = msg.get("required_actions", [])
        reasoning = msg.get("reasoning", "")

        orchestration_schema = {
            "type": "object",
            "properties": {
                "telegram_reply": {"type": "string"},
                "network_payload": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "delay": {"type": "number"},
                        "source": {"type": "string"},
                        "params": {"type": "object"},
                        "on_success": {"type": "object"},
                        "on_failure": {"type": "object"}
                    },
                    "required": ["action", "target", "source", "params"]
                }
            },
            "required": ["telegram_reply", "network_payload"]
        }

        response = await self.llm.chat(
            model='phi4-mini',
            format=orchestration_schema,
            messages=[
                {
                    'role': 'system',
                    'content': f"Execution plan (DO NOT DEVIATE): actions={actions}, reasoning={reasoning}. Format ONLY, with context {context_string}"
                },
                {
                    'role': 'user',
                    'content': text
                }
            ])

        response = response["message"]["content"]
        logger.debug(response)
        if response[0] == "`":
            response = response[7:-3]
        human_resp = json.loads(response)
        await update.message.reply_text(human_resp["telegram_reply"])
        net_commands = human_resp.get("network_payload", None)
        commands = net_commands

        if not isinstance(net_commands, List):
            commands = [net_commands]

        target = commands[0].get("target", None)
        if target is not None:
            for command in commands:
                await self.send_msg(command["target"], json.dumps(command))

    async def accept_peer(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not context.args:
            await update.message.reply_text("Please specify a node name to connect to: ")
            return

        target_node = context.args[0]

        if target_node in self.pending_peers:
            peerdata = self.pending_peers.pop(target_node)
            self.connect_peer(target_node, peerdata)

            await update.message.reply_text(f"Connected to {target_node}")
        else:
            await update.message.reply_text(f"Not recognised {target_node}")


    async def verification_prompt(self, peername: str, peerdata: dict) -> None:
        clean_name = peername.split('.')[0]

        short_id = uuid.uuid4().hex[:12]
        self.pending_peers[short_id] = {"name": clean_name,
                                        "data": peerdata}
        
        message = (
            f"🌐 <b>New HALO Node Discovered</b>\n"
            f"Name: {clean_name}\n"
            f"IP: {peerdata['ip']}\n\n"
            f"Reply with <code>/accept {clean_name}</code> to pair."
        )

        keyboard = [[InlineKeyboardButton("✅ Accept", callback_data=f"acc_{short_id}"),
                     InlineKeyboardButton("❌ Reject", callback_data=f"rej{short_id}")]]

        reply_markup = InlineKeyboardMarkup(keyboard)

        await self.bot.send_message(
            chat_id = self.admin_chat_id,
            text=message,
            parse_mode="HTML",
            reply_markup=reply_markup
        )

    async def send_message_to_user(self, msg) -> None:
        await self.bot.send_message(
            chat_id = self.admin_chat_id,
            text = str(msg["params"]["message"]),
            parse_mode="HTML"
        )

    async def handle_button_press(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        await query.answer()

        data = query.data

        if data.startswith("acc_"):
            short_id = data.split("acc_")[1]

            if short_id in self.pending_peers:
                pending_data = self.pending_peers[short_id]
                clean_name = pending_data["name"]
                peerdata = pending_data["data"]
                self.connect_peer(clean_name, peerdata)
                await query.edit_message_text(text=f"Accepted connection from {clean_name}")

                del self.pending_peers[short_id]
            else:
                await query.edit_message_text(text=f"{clean_name} is no longer available")
        elif data.startswith("rej_"):
            short_id = data.split("rej_")[1]
            pending_data = self.pending_peers[short_id]
            clean_name = pending_data["name"]
            await query.edit_message_text(text=f"Rejected {clean_name}")
            self.pending_peers.pop(short_id, None)

    def get_peer_info(self) -> None:
        return list(self.peers.keys())

    def get_peer_schema(self, msg: dict) -> None:
        schemas = list(msg.keys())
        self.schemas[schemas[1]] = msg[schemas[1]]
