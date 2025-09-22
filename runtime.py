# runtime.py
from nodes import SpecialistNode
from langgraph.graph import StateGraph, END
from typing import Dict, Any, Optional

# State for Graph
class AgentState(Dict):
    input: Dict
    output: Any
    node_metadata: Dict
    error: Optional[str] = None

def error_handler(state: AgentState) -> AgentState:
    """Handle errors by logging or modifying state."""
    if state.get("error"):
        print(f"Error handled: {state['error']}")
    return state  # Pass state to END

def build_graph():
    graph = StateGraph(AgentState)
    
    # Add nodes from catalog
    node1 = SpecialistNode("extractor")
    node2 = SpecialistNode("analyzer")
    graph.add_node("extractor", node1)
    graph.add_node("analyzer", node2)
    graph.add_node("error_handler", error_handler)  # Add error_handler node
    
    # Edges: Simple sequence for now
    graph.add_edge("extractor", "analyzer")
    graph.add_conditional_edges(
        "analyzer", 
        lambda state: END if not state.get("error") else "error_handler"  # Use END instead of "end"
    )
    graph.add_edge("error_handler", END)  # Edge from error_handler to END
    
    graph.set_entry_point("extractor")
    
    return graph.compile()

# (Optional) Export if needed for testing
if __name__ == "__main__":
    graph = build_graph()
    print("Graph compiled successfully!")