"""case_doc_rag.routers -- Pure routing functions for graph conditional edges.

Zero side effects. No LLM calls. No imports from nodes/, infrastructure, or prompts.
Only imports: state types, langgraph Send, and standard library.

Rule 5: Router functions return a string key (or a list of Send objects for
fan-out) and nothing else. They read from state, compute the routing decision,
and return it. They never write to state.
"""

import logging
from typing import List, Union

from langgraph.types import Send

from RAG.case_doc_rag.state import AgentState, SubQuestionState

logger = logging.getLogger("case_doc_rag.routers")

# Single source of truth for the retry ceiling.
# proceedRouter is the sole enforcer. refineQuestion must not import this.
_MAX_REPHRASE: int = 2


def onTopicRouter(state: AgentState) -> str:
    """Route after questionClassifier: documentSelector / offTopicResponse / errorResponse."""
    if state.get("error"):
        return "errorResponse"
    if state.get("on_topic", False) is True:
        return "documentSelector"
    return "offTopicResponse"


def docSelectorRouter(state: AgentState) -> str:
    """Route after documentSelector: DocumentFinalizer / dispatchQuestions / errorResponse.

    NOTE: Kept for backwards-compatibility / reference. The graph now uses
    docSelectorDispatchRouter which merges routing + fan-out into one
    conditional-edge function.
    """
    if state.get("error"):
        return "errorResponse"
    if state.get("doc_selection_mode") == "retrieve_specific_doc":
        return "DocumentFinalizer"
    return "dispatchQuestions"


def docSelectorDispatchRouter(
    state: AgentState,
) -> Union[str, List[Send]]:
    """Conditional-edge function after documentSelector.

    Returns a string key for DocumentFinalizer / errorResponse, **or** a list
    of Send objects that fan-out each sub-question to ``retrieve_branch``.

    This replaces the old pattern where ``dispatchQuestions`` was registered as
    a regular node (which crashed because nodes must return dicts, not Sends).
    """
    if state.get("error"):
        return "errorResponse"
    if state.get("doc_selection_mode") == "retrieve_specific_doc":
        return "DocumentFinalizer"

    # --- Fan-out logic (previously in query_nodes.dispatchQuestions) ---
    request_id = state.get("request_id", "")
    sub_questions = state.get("sub_questions", [])

    if not sub_questions:
        logger.error("[%s] No sub-questions to dispatch", request_id)
        sub_questions = [""]

    sends: List[Send] = []
    for sub_q in sub_questions:
        sends.append(
            Send(
                "retrieve_branch",
                {
                    "sub_question": sub_q,
                    "case_id": state.get("case_id", ""),
                    "conversation_history": state.get("conversation_history", []),
                    "selected_doc_id": state.get("selected_doc_id"),
                    "doc_selection_mode": state.get(
                        "doc_selection_mode", "no_doc_specified"
                    ),
                    "request_id": request_id,
                    "retrieved_docs": [],
                    "proceedToGenerate": False,
                    "rephraseCount": 0,
                    "sub_answer": "",
                    "sources": [],
                    "found": False,
                    "sub_answers": [],
                },
            )
        )

    logger.info(
        "[%s] docSelectorDispatchRouter: dispatching %d branches",
        request_id,
        len(sends),
    )
    return sends


def proceedRouter(state: SubQuestionState) -> str:
    """Route after retrievalGrader: generateAnswer / refineQuestion / cannotAnswer.

    This is the ONLY location that checks the rephraseCount ceiling.
    """
    if state.get("proceedToGenerate", False) is True:
        return "generateAnswer"
    if state.get("rephraseCount", 0) >= _MAX_REPHRASE:
        return "cannotAnswer"
    return "refineQuestion"
