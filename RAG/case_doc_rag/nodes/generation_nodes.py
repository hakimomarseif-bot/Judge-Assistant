"""case_doc_rag.nodes.generation_nodes -- Answer generation and merge nodes.

Nodes: generateAnswer, refineQuestion, cannotAnswer, mergeAnswers, errorResponse
"""

import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate

from RAG.case_doc_rag.infrastructure import get_llm
from RAG.case_doc_rag.prompts import REFINE_QUESTION_PROMPT, get_rag_chain
from RAG.case_doc_rag.state import AgentState, SubQuestionState

logger = logging.getLogger("case_doc_rag.generation_nodes")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _serialize_history(conversation_history: List[Dict[str, str]]) -> str:
    """Convert conversation history to a clean human-readable Arabic string.

    Fixes: Bug 6 (history serialized as Python object repr into the LLM).
    """
    if not conversation_history:
        return "لا يوجد سياق سابق."

    lines = []
    for turn in conversation_history:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if not role or not content:
            continue
        if role == "user":
            lines.append(f"القاضي: {content}")
        elif role == "assistant":
            lines.append(f"المساعد: {content}")

    if not lines:
        return "لا يوجد سياق سابق."

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def generateAnswer(state: SubQuestionState) -> Dict[str, Any]:
    """Generate an answer using the RAG chain with retrieved documents.

    Fixes: Bug 6 (history repr), removal of ValueError raises.
    """
    request_id = state.get("request_id", "")
    sub_question = state.get("sub_question", "")

    if not sub_question:
        logger.warning("[%s] generateAnswer: empty sub_question", request_id)
        entry = {
            "question": "",
            "answer": "لم يتم تحديد السؤال.",
            "sources": [],
            "found": False,
        }
        return {
            "sub_answer": "لم يتم تحديد السؤال.",
            "found": False,
            "sources": [],
            "sub_answers": [entry],
        }

    docs = state.get("retrieved_docs", [])
    context_str = "\n\n".join(doc.page_content for doc in docs) if docs else ""
    history_str = _serialize_history(state.get("conversation_history", []))

    try:
        response = get_rag_chain().invoke({
            "history": history_str,
            "context": context_str,
            "question": sub_question,
        })
        answer = response.content.strip()
        found = True
    except Exception:
        logger.exception("[%s] LLM error in generateAnswer", request_id)
        answer = "حدث خطأ أثناء توليد الإجابة. يرجى المحاولة مرة أخرى."
        found = True  # docs were found, generation failed

    # Build sources from document metadata
    sources = []
    for doc in docs:
        src = doc.metadata.get("source_file", "unknown")
        idx = doc.metadata.get("chunk_index", "?")
        sources.append(f"{src}:chunk_{idx}")

    # Include actual document text for downstream evaluation (RAGAS etc.)
    contexts = [doc.page_content for doc in docs] if docs else []

    answer_entry = {
        "question": sub_question,
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
        "found": found,
    }

    logger.info(
        "[%s] generateAnswer: answer=%s...",
        request_id, answer[:100],
    )
    return {
        "sub_answer": answer,
        "sources": sources,
        "found": True,
        "sub_answers": [answer_entry],
    }


def refineQuestion(state: SubQuestionState) -> Dict[str, Any]:
    """Rephrase a sub-question for better document retrieval.

    Fixes: Bug 7 (prompt_template.format() passed to llm.invoke()).
    Does NOT check rephraseCount ceiling -- that is proceedRouter's job.
    """
    request_id = state.get("request_id", "")
    sub_question = state.get("sub_question", "")
    rephrase_count = state.get("rephraseCount", 0)

    if not sub_question:
        return {"rephraseCount": rephrase_count + 1}

    try:
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", REFINE_QUESTION_PROMPT),
            ("human", "{question}"),
        ])
        chain = prompt_template | get_llm("high")
        response = chain.invoke({"question": sub_question})
        refined_question = response.content.strip()
    except Exception:
        logger.warning(
            "[%s] LLM error in refineQuestion, keeping original",
            request_id,
            exc_info=True,
        )
        refined_question = sub_question

    logger.debug(
        "[%s] refineQuestion: '%s' -> '%s'",
        request_id, sub_question[:80], refined_question[:80],
    )
    return {
        "sub_question": refined_question,
        "rephraseCount": rephrase_count + 1,
    }


def cannotAnswer(state: SubQuestionState) -> Dict[str, Any]:
    """Return a standard Arabic 'cannot answer' message.

    Includes sub_answers in the return dict so the branch contributes
    its result to AgentState.sub_answers via the operator.add reducer.
    """
    message = (
        "المستندات المتاحة لا تتضمن معلومة يمكن استخدامها للإجابة عن هذا السؤال. "
        "يرجى تحديد المستند أو النقطة القانونية المراد البحث فيها."
    )
    answer_entry = {
        "question": state.get("sub_question", ""),
        "answer": message,
        "sources": [],
        "found": False,
    }
    return {
        "sub_answer": message,
        "found": False,
        "sources": [],
        "sub_answers": [answer_entry],
    }


def mergeAnswers(state: AgentState) -> Dict[str, Any]:
    """Merge parallel branch results into final_answer.

    Runs on AgentState after all parallel branches complete.
    For single-question queries, sets final_answer directly.
    For multi-question queries, leaves final_answer empty (Supervisor
    reads sub_answers directly).
    """
    request_id = state.get("request_id", "")
    sub_answers = state.get("sub_answers", [])

    if not sub_answers:
        logger.warning("[%s] mergeAnswers: no sub_answers", request_id)
        return {"final_answer": ""}

    if len(sub_answers) == 1:
        final_answer = sub_answers[0].get("answer", "")
    else:
        # Multi-question: Supervisor reads sub_answers directly
        final_answer = ""

    found_count = sum(1 for sa in sub_answers if sa.get("found"))
    logger.info(
        "[%s] mergeAnswers: %d sub-answers, %d found",
        request_id, len(sub_answers), found_count,
    )
    return {"final_answer": final_answer}


def errorResponse(state: AgentState) -> Dict[str, Any]:
    """Return a standard Arabic system error message.

    Does NOT expose the internal error string to the judge.
    """
    request_id = state.get("request_id", "")
    error = state.get("error", "An internal error occurred")
    logger.error("[%s] errorResponse: %s", request_id, error)

    return {
        "final_answer": (
            "عذراً، واجه النظام مشكلة أثناء معالجة طلبك. "
            "يرجى المحاولة مرة أخرى أو التواصل مع الدعم الفني."
        )
    }
