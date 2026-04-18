# Reaserch AI Agent

from datetime import datetime

# LangChain Agent & Middleware
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_tool_call,
    ToolRetryMiddleware
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# LangChain Community 
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun
)
from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper
)

# LangGraph
from langgraph.checkpoint.memory import MemorySaver
# from langgraph.checkpoint.postgres import PostgresSaver 

# Model
from langchain_ollama import ChatOllama


# --- RESEARCH TOOLS ---
ddgs_warpper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_tool = DuckDuckGoSearchResults(
   api_wrapper=ddgs_warpper,
   name="dggs_web_search",
   description=""
)

ddgs_warpper = WikipediaAPIWrapper(max_results=5)
wiki_tool = WikipediaQueryRun(
   api_wrapper=ddgs_warpper,
   name="dggs_web_search",
   description=""
)

ddgs_warpper = ArxivAPIWrapper(max_results=5)
arxiv_tool = ArxivQueryRun(
   api_wrapper=ddgs_warpper,
   name="dggs_web_search",
   description=""
)

def get_current_datetime():
    """"""
    current_datetime = datetime.now()
    return current_datetime.strftime(
        ""
    )


tools = [ddgs_tool, wiki_tool, arxiv_tool, get_current_datetime]

SYSTEM_RESEARCH_PROMPT = """

"""

def create_reasearch_agent():
    llm = ""

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_RESEARCH_PROMPT,
        middleware="",
        name="research_agent",
        checkpointer=""
    )