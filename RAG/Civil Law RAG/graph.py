"""
graph.py

Defines and constructs the complete LangGraph workflow for the Egyptian Civil Law AI Judge Assistant.

Responsibilities:
1. Adds all nodes (preprocessor, retrieval, grading, refinement, answer generation, etc.) to the StateGraph.
2. Defines edges and conditional edges between nodes based on router functions.
3. Compiles the graph into an executable app for processing user queries.
4. Provides visualization support (e.g., Mermaid diagrams) for inspecting the workflow.
5. Stores the default state template used to initialize queries in the system.

This file separates the graph structure and routing logic from node implementations,
making it easier to modify the workflow without touching node logic.
"""

from nodes import *
from routers import *
from config.rag import default_state_template, START, END
from langgraph.graph import StateGraph
from indexer import ensure_civil_law_indexed

# Ensure the civil law corpus is indexed in Qdrant before first query
ensure_civil_law_indexed()

# Initialize the graph
graph = StateGraph(State)

# Add nodes to the graph
graph.add_node("preprocessor_node", preprocessor_node)
graph.add_node("off_topic_node", off_topic_node)
graph.add_node("textual_node", textual_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("rule_grader_node", rule_grader_node)
graph.add_node("refine_node", refine_node)
graph.add_node("llm_grader_node", llm_grader_node)
graph.add_node("generate_answer_node", generate_answer_node)
graph.add_node("cannot_answer_node", cannot_answer_node)

# Define edges and conditional edges based on routers
graph.add_edge(START, "preprocessor_node")

graph.add_conditional_edges(
    "preprocessor_node",
    top_level_router,
    {
        "off_topic_node": "off_topic_node",
        "textual_node": "textual_node",
        "retrieve_node": "retrieve_node"
    })

graph.add_edge("textual_node", END)
graph.add_edge("off_topic_node", END)
graph.add_edge("retrieve_node", "rule_grader_node")

graph.add_conditional_edges(
    "rule_grader_node",
    rule_grader_router,
    {
        "generate_answer_node": "generate_answer_node",
        "refine_node": "refine_node",
        "llm_grader_node": "llm_grader_node",
        "cannot_answer_node": "cannot_answer_node"
    })

graph.add_edge("refine_node", "retrieve_node")

graph.add_conditional_edges(
    "llm_grader_node",
    llm_grader_router,
    {
        "generate_answer_node": "generate_answer_node",
        "refine_node": "refine_node"
    })

graph.add_edge("generate_answer_node", END)
graph.add_edge("cannot_answer_node", END)

# Compile the graph into an executable app
app = graph.compile()