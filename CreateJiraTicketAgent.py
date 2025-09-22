from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage 
from langchain_core.messages import ToolMessage 
from langchain_core.messages import SystemMessage 
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages 
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import os
from jira import JIRA
load_dotenv()

jira_server = os.getenv("JIRA_SERVER")
jira_username = os.getenv("JIRA_USERNAME")
jira_api_token = os.getenv("JIRA_API_TOKEN")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 

@tool
def create_jira_ticket(
    project_key: str,
    summary: str,
    description: str ="",
    issue_type: str = "Task"
) -> str:
    """
    This is a create Jira's issue tool
    inputs: dict (project_key, summary, description, issue_type, ...)
    context: dict (JIRA_SERVER, JIRA_USERNAME, API_TOKEN)
    """

    jira = JIRA(server=jira_server, 
                basic_auth=(jira_username, jira_api_token))
    issue_dict = {
        'project': {'key': project_key},
        'summary': summary,
        'description': description,
        'issuetype': {'name': issue_type},
    }
    try:
        new_issue = jira.create_issue(fields=issue_dict)
        return f"Issue created: {jira_server}/browse/{new_issue.key}"
    except Exception as e:
        return f"Failed to create issue: {str(e)}"

tools = [create_jira_ticket]


model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are my AI assistant, please answer my query to the best of your ability."
    )

    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else: 
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue, 
    {
        "continue": "tools",
        "end": END,
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "Create a Jira ticket for project LEARNJIRA, with summary 'Testing', description 'Create testing issue', type 'Task'")]}
print_stream(app.stream(inputs, stream_mode="values")) 