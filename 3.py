import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from datetime import datetime


load_dotenv()


MODEL_NAME = os.getenv("MODEL_NAME", "minimax-m2.5")
TEMPERATURE = float(os.getenv("TEMPERATURE", "1.0"))

# LLM
llm = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE)


# Tool(s)
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


def get_time(city: str) -> str:
    # """Get the current time for a given city."""
    current_time = datetime.now().strftime("%I:%M %p")
    return f"The current time in {city} is {current_time}."


# Agent
agent = create_agent(
    model=llm,
    tools=[get_weather, get_time],
    system_prompt="You are a helpful assistant. Use the available tools when needed.",
)

# Run the agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather and current time in lahore?",
            }
        ]
    }
)


print(response["messages"][-1].content)
