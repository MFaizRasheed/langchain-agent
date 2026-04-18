"""
Research Agent Pro (HITL - Stable Version)

Fixes:
- Handles arXiv 429 errors
- Adds retry & fallback
- Prevents tool overuse
"""

from datetime import datetime
import json
import time

# LangChain
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain.tools import tool

# Tools
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    WikipediaQueryRun,
    ArxivQueryRun,
)

from langchain_community.utilities import (
    DuckDuckGoSearchAPIWrapper,
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
)

# Memory
from langgraph.checkpoint.memory import MemorySaver

# Model
from langchain_ollama import ChatOllama


# =========================
# 🔧 TOOLS SETUP
# =========================

ddgs_wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
ddgs_tool = DuckDuckGoSearchResults(
    api_wrapper=ddgs_wrapper,
    name="web_search",
    description="Search the web for current information",
)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=2)
wiki_tool = WikipediaQueryRun(
    api_wrapper=wiki_wrapper,
    name="wikipedia",
    description="Search Wikipedia for factual information",
)

# ⚠️ Reduced arXiv usage to avoid 429
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1)
arxiv_tool = ArxivQueryRun(
    api_wrapper=arxiv_wrapper,
    name="arxiv",
    description="Search academic papers from arXiv (use sparingly)",
)


@tool
def get_current_datetime():
    """Get current date and time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [ddgs_tool, wiki_tool, arxiv_tool, get_current_datetime]


# =========================
# 🧠 SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are a Research AI Agent.

Rules:
- Use Wikipedia for factual knowledge
- Use web_search for current info
- Use arxiv ONLY when user explicitly asks for research papers
- If any tool fails, answer directly without tools
- Always give clear and structured answers
"""


# =========================
# 🧠 CREATE AGENT
# =========================


def create_research_agent():
    llm = ChatOllama(model="minimax-m2.5:cloud", temperature=0)

    memory = MemorySaver()

    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    return agent


# =========================
# 📝 LOGGING (HITL)
# =========================


def log_decision(tool, decision):
    with open("hitl_log.json", "a") as f:
        f.write(
            json.dumps(
                {"tool": tool, "decision": decision, "time": str(datetime.now())}
            )
            + "\n"
        )


# =========================
# 🔁 SAFE INVOKE (Retry + Fallback)
# =========================


def safe_invoke(agent, state, config, retries=2):
    for attempt in range(retries):
        try:
            time.sleep(1)  # prevent rate limit
            return agent.invoke(state, config=config)
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1} failed: {e}")

    # fallback
    print("⚠️ Tool failed. Switching to direct answer...\n")
    state["messages"].append(
        HumanMessage(content="Answer without using any external tools.")
    )
    return agent.invoke(state, config=config)


# =========================
# 🔥 HUMAN-IN-THE-LOOP CORE
# =========================


def hitl_run(agent, query, config):
    state = {"messages": [HumanMessage(content=query)]}

    while True:
        result = safe_invoke(agent, state, config)
        last_msg = result["messages"][-1]

        # 🔴 TOOL INTERCEPT
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            tool_call = last_msg.tool_calls[0]

            print("\n🔎 Tool Request Detected")
            print(f"Tool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")

            decision = input("\nApprove? (y/n/edit): ").strip().lower()

            if decision == "y":
                print("✅ Approved\n")
                log_decision(tool_call["name"], "approved")
                state = result
                continue

            elif decision == "n":
                print("❌ Rejected\n")
                log_decision(tool_call["name"], "rejected")

                state["messages"].append(
                    HumanMessage(content="Do not use tool. Answer directly.")
                )
                continue

            elif decision == "edit":
                new_query = input("✏️ Modify query: ")
                log_decision(tool_call["name"], "edited")

                state["messages"].append(HumanMessage(content=new_query))
                continue

            else:
                print("Invalid input")
                continue

        # ✅ FINAL RESPONSE
        if isinstance(last_msg, AIMessage):
            print(f"\n🤖 Agent:\n{last_msg.content}\n")
            break


# =========================
# 🚀 MAIN LOOP
# =========================


def main():
    print("🔬 Research Agent Pro (Stable HITL)\n")

    agent = create_research_agent()
    config = {"configurable": {"thread_id": "session-1"}}

    while True:
        query = input("You: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            hitl_run(agent, query, config)
        except Exception as e:
            print(f"❌ Fatal Error: {e}")


if __name__ == "__main__":
    main()
