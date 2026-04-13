import os

CLAUDE_API_KEY = os.environ['CLAUDE_KEY']
TELEGRAM_BOT_KEY = os.environ['TELEGRAM_KEY']
TELEGRAM_CHAT_ID = os.environ['TEL_CHAT_ID']

from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import asyncio

from discovery.src.base_agent import BaseAgent
from language_agent import LanguageAgent

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_html(rf"Hi {user.mention_html()}!",
                                    reply_markup = ForceReply(selective=True))
    
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Help!")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(update.message.text)

llm = ChatOpenAI(
        model='bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0',
        temperature=0.7,
        max_tokens=1024,
        api_key=CLAUDE_API_KEY,
        base_url='https://litellm.prod.outshift.ai/'
        )

sys_prompt = """You are a HALO management assistant. You must process the user's request and output your response ONLY as a raw JSON object. Do not include markdown formatting or conversational filler outside the JSON.

Use this exact schema:
{
"telegram_reply": "The friendly, human-readable message to send to the user",
"network_payload": {
"action": "the actuation",
"target": "LightA1"
}
} """



async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(llm.invoke(sys_prompt + update.message.text).content)

def main():
    application = Application.builder().token(TELEGRAM_BOT_KEY).post_init(start_mesh).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("accept", accept_peer))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

    application.run_polling(allowed_updates=Update.ALL_TYPES)

lang_agent = None

async def accept_peer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Please specify a node name to connect to: ")
        return

    target_node = context.args[0]

    if target_node in lang_agent.pending_peers:
        peerdata = lang_agent.pending_peers.pop(target_node)
        lang_agent.connect_peer(target_node, peerdata)

        await update.message.reply_text(f"Connected to {target_node}")
    else:
        await update.message.reply_text(f"Not recognised {target_node}")

async def start_mesh(app: Application):
    global lang_agent
    lang_agent = LanguageAgent(TELEGRAM_BOT_KEY, TELEGRAM_CHAT_ID)

    asyncio.create_task(lang_agent.broadcast_and_discover())
    asyncio.create_task(lang_agent.recv_msg())

if __name__ == "__main__":
    main()
