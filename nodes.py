import os
import time
import requests
import logging
from typing import Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import tool
from langchain_core.exceptions import LangChainException
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor, TimeoutError  # For simple sandbox simulation
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Define Input Schema (from Abstraction Layer)
class NodeInput(BaseModel):
    query: str = Field(description="Query string to extract data")
    api_url: str = Field(default="https://api.example.com/data", description="API endpoint")
    max_retries: int = Field(default=3, description="Max retries on error")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v

# Define Output Schema
class NodeOutput(BaseModel):
    result: Dict[str, Any]
    partials: Optional[list] = None  # For streaming partials

# Secret Resolver (using dotenv)
def resolve_secrets(secret_ref: str) -> str:
    return os.getenv(secret_ref, "")  # Now pulls from .env via load_dotenv

# Error Handler with retry policy
def error_handler(error: Exception, retries_left: int) -> Any:
    if retries_left > 0:
        logger.info(f"Retrying... ({retries_left} left)")
        time.sleep(1)  # Exponential backoff in real prod
        return None  # Signal retry
    logger.error(f"Mapped error: {str(error)}")
    raise LangChainException(f"Mapped error: {str(error)}")

# Core Logic Executor (as a LangChain tool)
@tool
def data_extractor_tool(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Core logic to extract data from API."""
    api_key = resolve_secrets("API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        logger.info(f"Calling API: {input_data['api_url']} with query: {input_data['query']}")
        response = requests.get(input_data["api_url"], params={"q": input_data["query"]}, headers=headers, timeout=10)
        response.raise_for_status()
        return {"data": response.json()}
    except requests.RequestException as e:
        logger.error(f"API call failed: {str(e)}")
        raise e

# Simulate Sandbox: Run with timeout and resource limits
def sandbox_execute(func: callable, args: tuple, timeout: int = 10) -> Any:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:
            logger.error("Execution timed out (sandbox limit)")
            raise TimeoutError("Execution timed out (sandbox limit)")
        except Exception as e:
            logger.error(f"Sandbox error: {str(e)}")
            raise e

# MyNodeAdapter (implements INodeAdapter-like)
class MyNodeAdapter(Runnable):
    def __init__(self):
        self.param_parser = NodeInput  # Use Pydantic for validation/coercion

    def invoke(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None) -> NodeOutput:
        # Validate + coerce params
        try:
            parsed_input = self.param_parser(**input)
        except ValueError as e:
            logger.error(f"Param validation failed: {e}")
            raise ValueError(f"Param validation failed: {e}")

        retries = parsed_input.max_retries
        partials = []  # For partial yields

        while retries > 0:
            try:
                # Resolve secrets (if needed in core logic)
                # Run in sandbox
                result = sandbox_execute(data_extractor_tool.invoke, (parsed_input.model_dump(),))  # Changed .dict() to .model_dump()
                partials.append({"partial": "Processing complete"})  # Simulate partial
                return NodeOutput(result=result, partials=partials)
            except Exception as e:
                error_handler(e, retries)
                retries -= 1

        logger.error("Max retries exceeded")
        raise LangChainException("Max retries exceeded")

    def stream(self, input: Dict[str, Any], config: Optional[RunnableConfig] = None):
        # Streaming support (yield partials)
        parsed_input = self.param_parser(**input)
        yield {"partial": "Starting extraction..."}
        # Simulate streaming
        for i in range(3):
            time.sleep(1)
            yield {"partial": f"Chunk {i+1}"}
        result = data_extractor_tool.invoke(parsed_input.model_dump())  # Changed .dict() to .model_dump()
        yield {"final": result}

# Define AgentState using TypedDict
class AgentState(TypedDict):
    input: Dict[str, Any]
    output: Optional[NodeOutput]

def node_step(state: AgentState) -> Dict[str, Any]:
    adapter = MyNodeAdapter()
    output = adapter.invoke(state["input"])
    return {"output": output}

graph = StateGraph(AgentState)
graph.add_node("specialist_node", node_step)
graph.set_entry_point("specialist_node")
graph.set_finish_point("specialist_node")  # Simple single-node graph for demo
compiled_graph = graph.compile()

# Example Run (with LangSmith tracing automatic if env set)
input_data = {"query": "test data", "api_url": "https://jsonplaceholder.typicode.com/todos/1"}
result = compiled_graph.invoke({"input": input_data})
print(result["output"])