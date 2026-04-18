import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from datetime import datetime

# Load variables from .env
load_dotenv()

# Read configuration
MODEL_NAME = os.getenv("MODEL_NAME", "ollama:minimax-m2.5")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


# Tool 1
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


# Tool 2
def get_time(city: str) -> str:
    """Get the current time for a given city."""
    current_time = datetime.now().strftime("%I:%M %p")
    return f"The current time in {city} is {current_time}."


# Agent
agent = create_agent(
    model="ollama:minimax-m2.5:cloud",
    tools=[get_weather, get_time],
    system_prompt="You are a helpful assistant",
)

# Run the agent
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is the weather and current time in Lahore?",
            }
        ]
    }
)

# Print final response
print(response["messages"][-1].content)
