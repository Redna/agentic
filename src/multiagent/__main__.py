import os

from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, List
from langgraph.graph import END, StateGraph
from langchain_core.messages import ChatMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from langchain_groq import ChatGroq
from typing import Optional

from langchain_core.tools import tool

from langgraph.prebuilt.tool_executor import ToolExecutor, ToolInvocation

@tool
def get_current_weather(location: str, unit: Optional[str]):
    """Get the current weather in a given location"""
    return "Cloudy with a chance of rain."


model = ChatGroq(temperature=0,
                    model="llama3-70b-8192",)

ChatPromptTemplate.from_messages([("human", "Write a haiku about {topic}")])

tool_model = model.bind_tools([get_current_weather], tool_choice="auto")

res = tool_model.invoke("What is the weather like in San Francisco and Tokyo?")

res.tool_calls

tool_executor = ToolExecutor([get_current_weather])


result = []
for tool_call in res.tool_calls:
    name = tool_call["name"]
    args = tool_call["args"]

    invocation = ToolInvocation(tool=name, tool_input=args)
    ToolMessage(content=tool_executor.invoke(invocation))

    

print("Hello, World!")