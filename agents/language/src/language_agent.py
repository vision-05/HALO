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
CLAUDE_API_KEY = os.environ['CLAUDE_KEY']


class LanguageAgent(BaseAgent):
    def __init__(self, telegram_token: str, admin_chat_id: str) -> None:
        super().__init__("Claude", "Language")
        self.telegram_token = telegram_token
        self.admin_chat_id = admin_chat_id

        self.bot = Bot(telegram_token)

        self.pending_peers = {}
        self.handlers = {"schema": self.get_peer_schema}
        self.schemas = {}

        self.llm = ChatOpenAI(
            model='bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0',
            temperature=0.7,
            max_tokens=1024,
            api_key=CLAUDE_API_KEY,
            base_url='https://litellm.prod.outshift.ai/'
        )

        self.sys_prompt = """You are a HALO management assistant. You must process the user's request and output your response ONLY as a raw JSON object. Do not include markdown formatting or conversational filler outside the JSON.

                        Use this exact schema:
                        {
                        "telegram_reply": "The friendly, human-readable message to send to the user",
                        "network_payload": {
                        "action": "the actuation",
                        "target": "the target",
                        "source": "yourself",
                        "params": {"p1": "dict of params", "p2": "more params"},
                        "on_success": {"action": "next action",
                                       "target": "the next target",
                                       "source": "yourself",
                                       "params": {"p1": "$*", "p2": "other param"}},
                        "on_failure": {"action": "next action",
                                       "target": "the next target",
                                       "source": "yourself",
                                       "params": {"p1": "$*", "p2": "other param"}}
                        }} Where you can pass results of actions as parametsr by the wildcard $* 
                         Do NOT wrap your response in markdown blocks or include any backticks or the word json """

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user
        await update.message.reply_html(rf"Hi {user.mention_html()}!",
                                       reply_markup = ForceReply(selective=True))
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        current_chat_id = update.effective_chat.id
        await update.message.reply_text(f"Help! The ID for this chat is: {current_chat_id}")

    async def respond(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        is_group = update.message.chat.type in ['group', 'supergroup']
        bot_username = (await context.bot.get_me()).username
        
        text = update.message.text
        
        if is_group:
            # If in a group, ONLY respond if the bot is @mentioned
            if f"@{bot_username}" not in text:
                return # Ignore normal human-to-human group chatter
            
            # Strip the mention out so Claude just gets the raw command
            text = text.replace(f"@{bot_username}", "").strip()

        response = self.llm.invoke(self.sys_prompt + f"Current connected agents: {self.get_peer_info()}, current available actions for connected devices {self.schemas}" + text).content
        if response[0] == "`":
            response = response[7:-3]
        human_resp = json.loads(response)
        await update.message.reply_text(human_resp["telegram_reply"])
        net_commands = human_resp.get("network_payload", None)
        print(net_commands)
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
        print(self.schemas[schemas[1]])
