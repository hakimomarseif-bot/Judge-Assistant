"""case_doc_rag.state -- TypedDict definitions for graph state.

Defines AgentState (main graph) and SubQuestionState (per-branch fan-out).
No functions, no logic. No imports from inside the package.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Main graph state contract between all nodes and the Supervisor.

    Input fields are set once by the caller via run() and never modified.
    Processing fields are written by specific nodes as documented.
    """

    # -- Input fields (set once, never modified by nodes) --
    query: str
    case_id: str
    conversation_history: List[Dict[str, str]]
    request_id: str

    # -- Query processing fields --
    sub_questions: List[str]
    on_topic: bool

    # -- Document selection fields --
    doc_selection_mode: Literal[
        "retrieve_specific_doc", "restrict_to_doc", "no_doc_specified"
    ]
    selected_doc_id: Optional[str]
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
