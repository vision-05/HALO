import os
TELEGRAM_BOT_KEY = os.environ['TELEGRAM_KEY']
TELEGRAM_CHAT_ID = os.environ['TEL_CHAT_ID']

from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler

import asyncio
import json

from discovery.src.base_agent import BaseAgent
from language_agent import LanguageAgent

lang_agent = None


async def start_mesh(application: Application) -> None:
    global lang_agent
    lang_agent = LanguageAgent(TELEGRAM_BOT_KEY, TELEGRAM_CHAT_ID) 

    asyncio.create_task(lang_agent.broadcast_and_discover())
    asyncio.create_task(lang_agent.heartbeat())
    asyncio.create_task(lang_agent.prune_network())
    asyncio.create_task(lang_agent.recv_msg())

    application.add_handler(CommandHandler("start", lang_agent.start))
    application.add_handler(CommandHandler("help", lang_agent.help_command))
    application.add_handler(CommandHandler("accept", lang_agent.accept_peer))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, lang_agent.respond))
    application.add_handler(CallbackQueryHandler(lang_agent.handle_button_press))

def start_telegram_bot() -> None:
    global lang_agent
    
    application = Application.builder().token(TELEGRAM_BOT_KEY).post_init(start_mesh).build()
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    start_telegram_bot()
