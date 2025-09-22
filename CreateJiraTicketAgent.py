from typing import TypedDict, Annotated, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage 
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
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

# jira_server = os.getenv("JIRA_SERVER")
# jira_username = os.getenv("JIRA_USERNAME")
# jira_api_token = os.getenv("JIRA_API_TOKEN")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages] 

issue_info = ""
@tool
def update(content: str) -> str:
    """Update the issue with the provide content."""
    global issue_info
    issue_info = content
    return f"The issue has been updated! The current issue is: \n{issue_info}"

    
@tool
def create_jira_ticket(
    project_key: str,
    summary: str,
    description: str ="",
    issue_type: str = "Task",
    jira_server: str = "",
    jira_username: str = "",
    jira_api_token: str = ""
) -> str:
    """
    This is a create Jira's issue tool
    inputs: dict (project_key, summary, description, issue_type,
                    JIRA_SERVER, JIRA_USERNAME, API_TOKEN...)
    Format:
        JIRA_SERVER: Jira server URL (https://<domain>.atlassian.net)
        JIRA_USERNAME: Jira username or Email
        API_TOKEN: Jira API token
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

tools = [update, create_jira_ticket]


model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are a Jira Assistant, please do all the query to the best of your ability.
        - Make sure the user provides enough information for you to take action with the tool 'create_jira_ticket'.
        - Always show the current information of the issue for the user in case they might want to change something.
        - Only create the issue after the user satisfies and confirms the current information if not then use the 'update' tool.
        """
    )

    if not state['messages']:
        user_input = "How can i help you? "
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like me to do?\n")
        # print(f"\nUser: {user_input}")
        user_message = HumanMessage(content=user_input)
        
    all_message = [system_prompt] + list(state["messages"]) + [user_message]
    response = model.invoke(all_message)
    
    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")
    return {"messages": list(state["messages"]) + [user_message, response]}


def should_continue(state: AgentState) -> AgentState:
    """
    Determine if you should continue or end the conversation
    """
    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage)
            and "create" in message.content.lower()
            and "issue" in message.content.lower()):
            return "end"

    else: 
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("our_agent")

graph.add_edge("our_agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue, 
    {
        "continue": "our_agent",
        "end": END,
    }
)

app = graph.compile()

def print_messages(messages):
    """Function i make to print the message in a more readable format"""
    if not messages:
        return
    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n TOOL RESULT: {message.content}")

def run_agent():
    print("\n ===== JIRA ASSISTANT =====")
    
    state = {"messages": []}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
            
    print("\n ===== FINISHED =====")

# inputs = {"messages": [("user", "Create a Jira ticket for project LEARNJIRA, with summary 'Testing', description 'Create testing issue', type 'Task'")]}
# print_stream(app.stream(inputs, stream_mode="values"))
if __name__ == "__main__":
    run_agent()