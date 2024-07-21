import os

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from typing import Optional, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain.globals import set_debug

set_debug(True)

os.environ["GROQ_API_KEY"] = "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"
memory = SqliteSaver.from_conn_string(":memory:")


class AgentState(TypedDict):
    task: str 
    messages: Annotated[list[AnyMessage], operator.add]
    plan_outline: str
    action: AIMessage
    intermediate_steps: Annotated[list[str], operator.add]
    answer: str
    critique: str
    task_done: bool

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.base_system = system
        self.system = system
        workflow = StateGraph(AgentState)
        workflow.add_node("plan", self.plan)
        workflow.add_node("act", self.act)
        workflow.add_node("execute", self.execute)
        workflow.add_node("observe", self.observe)
        workflow.add_node("reflect", self.reflect)
        workflow.add_node("conclude", self.conclude)

        workflow.add_edge("plan", "act")
        workflow.add_edge("act", "execute")
        workflow.add_edge("execute", "observe")
        workflow.add_edge("observe", "reflect")
        workflow.add_conditional_edges("reflect", self.is_final_answer, {True: "conclude", False: "plan"})
        workflow.add_edge("conclude", END)
        workflow.set_entry_point("plan")

        self.graph = workflow.compile(checkpointer=checkpointer)

        self.tools = tools
        self.tools_lookup = {t.name: t for t in tools}
        self.model = model

    def plan(self, state: AgentState):
        system = self.base_system + """
Your task is to outline the necessary steps to solve the given task."""

        template = ChatPromptTemplate.from_messages([
            ("system", system),
            ("placeholder", "{messages}"),
            ("human", "{task_input}"),
        ])
        
        chat_prompt = template.invoke({
            "task_input": state['task'],
            "messages": state['messages']
        })

        model = self.model.bind_tools(self.tools, tool_choice="none")

        message = model.invoke(chat_prompt)
        print(f"Plan: {message.content}")
        return {"plan_outline": message.content, "intermediate_steps": ["Thought: " + message.content]}
    
    def act(self, state: AgentState):
        system = self.base_system + """
You can make calls to the tools you have available.        
"""

        template = ChatPromptTemplate.from_messages([
            ("system", system),
            ("placeholder", "{messages}"),
            ("human", "{task_input}"),
        ])

        task = f"""
# Your initial objective: 
{state['task']}

# Your current outline on how to solve the task:
{state['plan_outline']}
 
Run the necessary tools to solve the task."""

        chat_prompt = template.invoke({
            "task_input": task,
            "messages": state['messages']
        })

        model = self.model.bind_tools(self.tools)
        return {"action": model.invoke(chat_prompt)}

    def execute(self, state: AgentState):
        tool_calls = state['action'].tool_calls
        intermediate_steps = []
        
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools_lookup[t['name']].invoke(t['args'])
            intermediate_steps.append(f"Action: Calling {t['name']} with args: {t['args']}")#
            intermediate_steps.append(f"Observation: ### {result} ###")
        return {'intermediate_steps': intermediate_steps}

    def observe(self, state: AgentState):
        system = self.base_system + """
Observe the result of the action, the given taks and the outline."""

        template = ChatPromptTemplate.from_messages([
            ("system", system),
            ("placeholder", "{messages}"),
            ("human", "{task_input}"),
        ])

        task = f"""
# Your initial objective: 
{state['task']}

# Current scratchpad:
{"\n".join(state['intermediate_steps'])}

Make your observation the current state of the task and the actions taken to solve it and reason if you have enough information to give a final answer."""
        
        chat_prompt = template.invoke({
            "task_input": task,
            "messages": state['messages']
        })

        message = self.model.invoke(chat_prompt)

        print(f"Observation: {message.content}")

        return {"intermediate_steps": ["Observation: " + message.content]}
    
    def reflect(self, state: AgentState):

        template = ChatPromptTemplate.from_messages([
            ("system", self.base_system),
            ("placeholder", "{messages}"),
            ("human", "{task_input}"),
        ])

        class Critique(BaseModel):
            critique: str = Field(description="Critique of the current state of the task and the actions taken to solve it.")
            task_done: bool = Field(description="Is the task done?")

        task = f"""
# Your initial objective:
{state['task']}

# Current scratchpad:
{"\n".join(state['intermediate_steps'])}

Your task is to reflect on your work. Based on the observation, critique the current state of the task and the actions taken to solve it. Is the information sufficient to give a final answer? Is the path to the solution clear and understandable? 
Is there more research needed?"""

        chat_prompt = template.invoke({
            "task_input": task,
            "messages": state['messages']
        })

        message = self.model.with_structured_output(Critique).invoke(chat_prompt)

        print(f"Critique: {message.critique}")
        print(f"Task done: {message.task_done}")

        return {"critique": message.critique, "task_done": message.task_done}


    def is_final_answer(self, state: AgentState):
        return state['task_done']

    def conclude(self, state: AgentState):
        system = self.base_system + """
Beaed on your observation, give a final answer to the task. Outline your path to the solution to make it clear and understandable."""

        template = ChatPromptTemplate.from_messages([
            ("system", system),
            ("placeholder", "{messages}"),
            ("human", "{task_input}"),
        ])

        task = f"""
# Your initial objective: 
{state['task']}

# Current scratchpad:
{"\n".join(state['intermediate_steps'])}

Give the final answer and outline your path to the solution."""

        chat_prompt = template.invoke({
            "task_input": task,
            "messages": state['messages']
        })

        model = self.model.bind_tools(self.tools, tool_choice="none")
        message = model.invoke(chat_prompt)

        return {"answer": message.content, "messages": [message]}

prompt = """You are a smart research assistant. You are given a certain set of tools to solve the given task.\
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

tool = DuckDuckGoSearchResults()
model = ChatGroq(temperature=0,
                    model="llama3-groq-8b-8192-tool-use-preview" #"llama3-groq-70b-8192-tool-use-preview"
                    )
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}

task = "What is the largest city of France and how much inhabitants live there. Compare the number city with the largest city of Europe."
task = "How can you help me?"

task = "Create a report of the top 5 frameworks for creating LLM agents. Compare the frameworks based on their popularity and ease of use."


final_output = ""
for event in abot.graph.stream({"task": task, 
                   "messages": [], 
                   "intermediate_steps": [], 
                   "plan_outline": "",
                   "action": None, 
                   "answer": "", 
                   "final_answer": False}, thread):
    for v in event.values():
        print(v)
        final_output = v

print(final_output['messages'][-1].content)

with open("workflow.png", "wb") as fh:
    fh.write(abot.graph.get_graph().draw_mermaid_png())



def run_step_by_step():
    plan = abot.plan({"task": "Find the capital of France and how much inhabitants live there. Compare the number with the German capital.", "messages": [], "intermediate_steps": []})

    act = abot.act({"task": "Find the capital of France and how much inhabitants live there. Compare the number with the German capital.", 
                    "messages": [], 
                    "intermediate_steps": plan['intermediate_steps'],
                    "plan_outline": plan['plan_outline']})

    execute = abot.execute({"task": "Find the capital of France and how much inhabitants live there. Compare the number with the German capital.",
                            "messages": [], 
                            "intermediate_steps": plan['intermediate_steps'],
                            "plan_outline": plan['plan_outline'],
                            "action": act['action']})

    observe = abot.observe({"task": "Find the capital of France and how much inhabitants live there. Compare the number with the German capital.",
                            "messages": [], 
                            "intermediate_steps": plan["intermediate_steps"] + execute['intermediate_steps'],
                            "plan_outline": plan['plan_outline'],
                            "action": act['action']})

    conclude = abot.conclude({"task": "Find the capital of France and how much inhabitants live there. Compare the number with the German capital.",
                            "messages": [], 
                            "intermediate_steps": plan["intermediate_steps"] + execute['intermediate_steps'] + observe['intermediate_steps'],
                            "plan_outline": plan['plan_outline'],
                            "action": act['action'],
                            "final_answer": False})


