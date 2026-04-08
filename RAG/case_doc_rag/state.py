"""case_doc_rag.state -- TypedDict definitions for graph state.

Defines AgentState (main graph) and SubQuestionState (per-branch fan-out).
No functions, no logic. No imports from inside the package.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


def _last_value(existing, new):
    """Last-writer-wins reducer for fields shared between AgentState and SubQuestionState.

    When parallel Send branches complete and merge back into AgentState,
    fields present in both states receive one value per branch.  LangGraph
    requires a reducer to resolve the conflict.  Since these fields are
    read-only copies (set once by the caller), any branch value is
    identical -- so simply keeping the latest write is semantically correct.
    """
    return new


class AgentState(TypedDict):
    """Main graph state contract between all nodes and the Supervisor.

    Input fields are set once by the caller via run() and never modified.
    Processing fields are written by specific nodes as documented.
    """

    # -- Input fields (set once, never modified by nodes) --
    # Fields shared with SubQuestionState use _last_value reducer to
    # prevent INVALID_CONCURRENT_GRAPH_UPDATE when parallel branches
    # write back identical copies of these read-only values.
    query: str
    case_id: Annotated[str, _last_value]
    conversation_history: Annotated[List[Dict[str, str]], _last_value]
    request_id: Annotated[str, _last_value]

    # -- Query processing fields --
    sub_questions: List[str]
    on_topic: bool

    # -- Document selection fields --
    # doc_selection_mode and selected_doc_id are also in SubQuestionState,
    # so they need _last_value reducers to avoid concurrent-update errors.
    doc_selection_mode: Annotated[
        Literal["retrieve_specific_doc", "restrict_to_doc", "no_doc_specified"],
        _last_value,
    ]
    selected_doc_id: Annotated[Optional[str], _last_value]
    doc_titles: List[str]

    # -- Fan-out result field (CRITICAL: annotated reducer required) --
    sub_answers: Annotated[List[Dict[str, Any]], operator.add]

    # -- Output and error fields --
    final_answer: str
    error: Optional[str]


class SubQuestionState(TypedDict):
    """Per-branch state for parallel fan-out.

    Each parallel branch spawned by dispatchQuestions gets its own
    independent SubQuestionState. No field from AgentState is
    automatically inherited -- all required values are explicitly
    copied by dispatchQuestions at dispatch time.
    """

    # -- Fields copied from AgentState at dispatch time (read-only) --
    sub_question: str
    case_id: str
    conversation_history: List[Dict[str, str]]
    selected_doc_id: Optional[str]
    doc_selection_mode: str
    request_id: str

    # -- Retrieval field --
    retrieved_docs: List[Document]

    # -- Retry and grading fields (branch-local) --
    proceedToGenerate: bool
    rephraseCount: int

    # -- Output fields --
    sub_answer: str
    sources: List[str]
    found: bool

    # -- Fan-out result field (must match AgentState.sub_answers so that
    #    branch results propagate back to the parent graph via the
    #    operator.add reducer) --
    sub_answers: Annotated[List[Dict[str, Any]], operator.add]
