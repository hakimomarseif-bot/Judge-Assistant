"""
routers.py

Contains all routing logic for the Egyptian Civil Law AI Judge Assistant.
Routers determine the next node in the LangGraph workflow based on the
current state, such as the query classification, retrieval results, and
grading outcomes.

Each router function returns the name of the next node to execute.
This separation of routing logic from node logic allows for easier
maintenance and flexible modification of paths without touching node code.
"""
from nodes import State

# ---------------------
# Top Level Router
# ---------------------
def top_level_router(state: State) -> str:
    """
    Determine the main path after the preprocessing node.

    Logic:
        - "off_topic" classification → off_topic_node
        - "textual" classification → textual_node
        - "analytical" classification → retrieve_node (start analytical path)
        - Any other case → cannot_answer_node

    Args:
        state (State): The current state of the AI system.

    Returns:
        str: The name of the next node to execute.
    """
    classification = state.get("classification")

    if classification == "off_topic":
        return "off_topic_node"
    elif classification == "textual":
        return "textual_node"
    elif classification == "analytical":
        return "retrieve_node"
    else:
        return "cannot_answer_node"
    
# ---------------------
# Rule Grader Router
# ---------------------
def rule_grader_router(state: State) -> str:
    """
    Decide the next node based on the outcome of the rule-based grading node.

    Logic:
        - If retry_count >= max_retries → cannot_answer_node
        - grade == "pass" → generate_answer_node
        - grade == "refine" → refine_node
        - grade == "fail" → llm_grader_node
        - Default → cannot_answer_node

    Args:
        state (State): The current state of the AI system.

    Returns:
        str: The name of the next node to execute.
    """
    if state.get("retry_count", 0) >= state.get("max_retries", 2):
        return "cannot_answer_node"

    grade = state.get("grade")
    if grade == "pass":
        return "generate_answer_node"
    elif grade == "refine":
        return "refine_node"
    elif grade == "fail":
        return "llm_grader_node"
    else:
        return "cannot_answer_node"
    
# ---------------------
# LLM Grader Router
# ---------------------
def llm_grader_router(state: State) -> str:
    """
    Decide the next node after the LLM grading node.

    Logic:
        - If retry_count >= max_retries → cannot_answer_node
        - If llm_pass is True → generate_answer_node
        - Otherwise → refine_node

    Args:
        state (State): The current state of the AI system.

    Returns:
        str: The name of the next node to execute.
    """
    if state.get("retry_count", 0) >= state.get("max_retries", 2):
        return "cannot_answer_node"

    if state.get("llm_pass", False):
        return "generate_answer_node"
    else:
        return "refine_node"