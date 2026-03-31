import os

CLAUDE_API_KEY = os.environ['CLAUDE_KEY']

from typing import List, Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        max_tokens=1024,
        api_key=CLAUDE_API_KEY
        )

prompt = "What steps needed to turn on a lightbulb"
response = llm.invoke(prompt)

print(response.content)
