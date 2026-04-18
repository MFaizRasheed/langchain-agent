"""
Research AI Agent

A AI-based research assistant that leverages web search, Wikipedia,
and arXiv to gather and synthesize information on any topic.

Features:
- Web search via DuckDuckGo for current information
- Wikipedia queries for encyclopedia-style facts
- arXiv search for academic papers and research
- Persistent conversation memory across sessions

Usage:
    uv run research_agent.py

Type 'quit', 'exit', or 'q' to exit the interactive session.
"""

from datetime import datetime

# LangChain Agent & Middleware
from langchain.agents import create_agent
# from langchain.agents.middleware import (
#     wrap_tool_call,
#     ToolRetryMiddleware
# )
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool

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
# Web search tool - searches the web for current information
ddgs_warpper = DuckDuckGoSearchAPIWrapper(max_results=5)
ddgs_tool = DuckDuckGoSearchResults(
   api_wrapper=ddgs_warpper,
   name="web_search",
   description="Search the web using DuckDuckGo for current information, news, and general web content. Use this when you need up-to-date information or content not available on Wikipedia."
)

# Wikipedia tool - queries Wikipedia for encyclopedia-style information
wiki_warpper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(
   api_wrapper=wiki_warpper,
   name="wikipedia",
   description="Search Wikipedia for encyclopedia-style information, facts, and summaries. Use this for quick factual queries and well-established knowledge."
)

# arXiv tool - searches for academic papers and research
arxiv_warpper = ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=2000)
arxiv_tool = ArxivQueryRun(
   api_wrapper=arxiv_warpper,
   name="arxiv",
   description="Search arXiv for academic papers, scientific research, and scholarly articles. Use this for technical and academic research queries."
)

@tool
def get_current_datetime():
    """Get the current date and time.

    Returns:
        str: The current datetime formatted as a string.
    """
    current_datetime = datetime.now()
    return current_datetime.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


tools = [ddgs_tool, wiki_tool, arxiv_tool, get_current_datetime]

SYSTEM_RESEARCH_PROMPT = '''You are a Research AI Agent, an intelligent assistant specialized in conducting research and gathering information from multiple sources.

Your capabilities:
- Web Search: Use DuckDuckGo to find current information, news, and web content
- Wikipedia: Query for encyclopedia-style facts and well-established knowledge
- arXiv: Search for academic papers and scientific research
- DateTime: Get the current date and time when needed

Guidelines:
1. Always use the most appropriate tool for the type of information needed
2. For factual queries, start with Wikipedia
3. For current events or recent information, use web search
4. For academic or technical research, use arXiv
5. Synthesize information from multiple sources when possible
6. Provide clear, well-structured responses with proper citations
7. If a query is ambiguous, ask for clarification before searching

When responding, cite your sources and provide accurate, up-to-date information.
'''

def create_reasearch_agent():
    """Create and configure the research agent with LLM, tools, and memory.

    Returns:
        Configured LangChain agent with web search, Wikipedia, and arXiv tools.
    """
    llm = ChatOllama(
        model="minimax-m2.5:cloud",
        temperature=0
    )

    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_RESEARCH_PROMPT,
        checkpointer=memory,
        name="research_agent",
    )

    return agent

def banner():
    """Display the agent banner."""
    print("Research AI Agent")

def stream_response(agent, query: str, config: dict):
    """Stream and print the agent's response to a query.

    Args:
        agent: The LangChain agent instance.
        query: The user's query string.
        config: Configuration dictionary with thread_id for memory.
    """
    for chunk in agent.stream({"messages": [HumanMessage(content=query)]}, 
                              config=config,
                              stream_mode="values"):
        # Each chunk contains the full state at that point
        latest_message = chunk["messages"][-1]
        if latest_message.content:
            if isinstance(latest_message, HumanMessage):
                # print(f"User: {latest_message.content}")
                pass
            elif isinstance(latest_message, AIMessage):
                print(f"Agent: {latest_message.content}")
        elif latest_message.tool_calls:
            print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

def main():
    """Main entry point for the Research AI Agent.

    Runs an interactive CLI session where users can query the research agent.
    The agent maintains conversation history across sessions.
    """
    banner()
    agent = create_reasearch_agent()
    config = {"configurable": {"thread_id": "research-session-1"}}

    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit", "q"):
            print("\nGoodBye! Happy Researching!")
            break

        try:
            stream_response(agent, query, config)
        except Exception as err:
            print(f"Error: {err}")


main()