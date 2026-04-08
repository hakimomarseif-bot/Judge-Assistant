"""case_doc_rag.graph -- Graph wiring and compilation.

No node logic. No prompts. No infrastructure calls. Imports all node
functions and routers, wires them, and returns the compiled app from
build_graph(). No graph compilation at module import time.
"""

from langgraph.graph import END, START, StateGraph

from RAG.case_doc_rag.nodes.generation_nodes import (
    cannotAnswer,
    errorResponse,
    generateAnswer,
    mergeAnswers,
    refineQuestion,
)
from RAG.case_doc_rag.nodes.query_nodes import (
    offTopicResponse,
    questionClassifier,
    questionRewriter,
)
from RAG.case_doc_rag.nodes.retrieval_nodes import retrieve, retrievalGrader
from RAG.case_doc_rag.nodes.selection_nodes import DocumentFinalizer, documentSelector
from RAG.case_doc_rag.routers import docSelectorDispatchRouter, onTopicRouter, proceedRouter
from RAG.case_doc_rag.state import AgentState, SubQuestionState


def build_graph():
    """Build and compile the Case Doc RAG LangGraph workflow.

    Returns the compiled StateGraph app with .invoke() method.
    """
    # --- Branch sub-graph (operates on SubQuestionState) ---
    branch = StateGraph(SubQuestionState)
    branch.add_node("retrieve", retrieve)
    branch.add_node("retrievalGrader", retrievalGrader)
    branch.add_node("refineQuestion", refineQuestion)
    branch.add_node("generateAnswer", generateAnswer)
    branch.add_node("cannotAnswer", cannotAnswer)

    branch.add_edge(START, "retrieve")
    branch.add_edge("retrieve", "retrievalGrader")
    branch.add_conditional_edges(
        "retrievalGrader",
        proceedRouter,
        {
            "generateAnswer": "generateAnswer",
            "refineQuestion": "refineQuestion",
            "cannotAnswer": "cannotAnswer",
        },
    )
    branch.add_edge("refineQuestion", "retrieve")  # retry loop
    branch.add_edge("generateAnswer", END)
    branch.add_edge("cannotAnswer", END)

    branch_graph = branch.compile()

    # --- Main graph (operates on AgentState) ---
    workflow = StateGraph(AgentState)

    workflow.add_node("questionRewriter", questionRewriter)
    workflow.add_node("questionClassifier", questionClassifier)
    workflow.add_node("offTopicResponse", offTopicResponse)
    workflow.add_node("documentSelector", documentSelector)
    workflow.add_node("DocumentFinalizer", DocumentFinalizer)
    workflow.add_node("retrieve_branch", branch_graph)
    workflow.add_node("mergeAnswers", mergeAnswers)
    workflow.add_node("errorResponse", errorResponse)

    # Edges
    workflow.add_edge(START, "questionRewriter")
    workflow.add_edge("questionRewriter", "questionClassifier")
    workflow.add_conditional_edges(
        "questionClassifier",
        onTopicRouter,
        {
            "documentSelector": "documentSelector",
            "offTopicResponse": "offTopicResponse",
            "errorResponse": "errorResponse",
        },
    )
    workflow.add_edge("offTopicResponse", END)
    workflow.add_conditional_edges(
        "documentSelector",
        docSelectorDispatchRouter,
        {
            "DocumentFinalizer": "DocumentFinalizer",
            "errorResponse": "errorResponse",
        },
    )
    workflow.add_edge("DocumentFinalizer", END)
    workflow.add_edge("retrieve_branch", "mergeAnswers")
    workflow.add_edge("mergeAnswers", END)
    workflow.add_edge("errorResponse", END)

    return workflow.compile()
