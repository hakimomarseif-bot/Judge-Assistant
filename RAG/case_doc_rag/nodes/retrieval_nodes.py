"""case_doc_rag.nodes.retrieval_nodes -- Retrieval and grading nodes.

Nodes: retrieve, retrievalGrader
Both operate on SubQuestionState.
"""

import concurrent.futures
import logging
from typing import Any, Dict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client.models import FieldCondition, Filter, MatchValue

from RAG.case_doc_rag.infrastructure import get_llm, get_retriever, get_vectorstore
from RAG.case_doc_rag.models import GradeDocument
from RAG.case_doc_rag.state import SubQuestionState

logger = logging.getLogger("case_doc_rag.retrieval_nodes")

# Module-level constant for score threshold -- easy to tune
_RETRIEVAL_K = 15

# Grading system prompt -- tightly coupled to GradeDocument schema, used only here
_GRADING_SYSTEM_PROMPT = (
    "أنت مقيّم لمدى صلة المستندات القانونية المسترجعة بسؤال القاضي.\n\n"
    "المستندات مستخرجة من ملفات قضايا مدنية مصرية (صحف دعاوى، تقارير خبراء، أحكام محاكم).\n"
    "المستندات مكتوبة بالعربية، وسؤال القاضي أيضاً بالعربية.\n\n"
    "إذا احتوى المستند على كلمات مفتاحية أو وقائع أو أسماء أو تواريخ أو مراجع قانونية\n"
    "يمكن أن تساعد في الإجابة على سؤال القاضي، قيّمه بـ 'Yes'.\n"
    "قيّم بـ 'No' فقط إذا كان المستند غير مرتبط تماماً بالسؤال.\n\n"
    "أعط تقييماً ثنائياً 'Yes' أو 'No'."
)


def retrieve(state: SubQuestionState) -> Dict[str, Any]:
    """Retrieve documents from Qdrant with multi-level fallback.

    Fixes: Arch 7 (metadata key inconsistency), Perf 3 (no score threshold).
    Uses metadata.title as canonical filter key with metadata.type fallback.
    """
    request_id = state.get("request_id", "")
    sub_question = state.get("sub_question", "")
    case_id = state.get("case_id", "")
    doc_target = state.get("selected_doc_id")
    doc_mode = state.get("doc_selection_mode", "no_doc_specified")

    docs = []
    vs = get_vectorstore()

    if doc_target and doc_mode == "restrict_to_doc":
        # --- restrict_to_doc mode ---

        # Attempt 1: case_id + metadata.title
        if case_id:
            meta_filter = Filter(must=[
                FieldCondition(key="metadata.case_id", match=MatchValue(value=case_id)),
                FieldCondition(key="metadata.title", match=MatchValue(value=doc_target)),
            ])
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": _RETRIEVAL_K, "filter": meta_filter},
            )
            docs = retriever.invoke(sub_question)
            logger.debug(
                "[%s] retrieve attempt1 (case_id+title): %d docs",
                request_id, len(docs),
            )

        # Attempt 2: case_id only (drop title filter)
        if not docs and case_id:
            meta_filter = Filter(must=[
                FieldCondition(key="metadata.case_id", match=MatchValue(value=case_id)),
            ])
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": _RETRIEVAL_K, "filter": meta_filter},
            )
            docs = retriever.invoke(sub_question)
            logger.debug(
                "[%s] retrieve attempt2 (case_id only): %d docs",
                request_id, len(docs),
            )

        # Attempt 3: metadata.type (legacy data, no case_id -- breaks isolation)
        if not docs:
            meta_filter = Filter(must=[
                FieldCondition(key="metadata.type", match=MatchValue(value=doc_target)),
            ])
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": _RETRIEVAL_K, "filter": meta_filter},
            )
            docs = retriever.invoke(sub_question)
            if docs:
                logger.warning(
                    "[%s] retrieve attempt3 (metadata.type, no case isolation): %d docs",
                    request_id, len(docs),
                )

        # Attempt 4: unfiltered last resort
        if not docs:
            docs = get_retriever({"k": _RETRIEVAL_K}).invoke(sub_question)
            if docs:
                logger.warning(
                    "[%s] retrieve attempt4 (unfiltered): %d docs",
                    request_id, len(docs),
                )

    else:
        # --- no_doc_specified mode ---

        # Attempt 1: case_id filter
        if case_id:
            meta_filter = Filter(must=[
                FieldCondition(key="metadata.case_id", match=MatchValue(value=case_id)),
            ])
            retriever = vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": _RETRIEVAL_K, "filter": meta_filter},
            )
            docs = retriever.invoke(sub_question)
            logger.debug(
                "[%s] retrieve (case_id): %d docs", request_id, len(docs),
            )

        # Attempt 2: unfiltered fallback
        if not docs:
            docs = get_retriever({"k": _RETRIEVAL_K}).invoke(sub_question)
            logger.warning(
                "[%s] retrieve (unfiltered fallback): %d docs",
                request_id, len(docs),
            )

    return {"retrieved_docs": docs}


def retrievalGrader(state: SubQuestionState) -> Dict[str, Any]:
    """Grade each retrieved document for relevance in parallel.

    Fixes: Perf 1 (sequential grading -- up to 8 blocking LLM calls).
    Uses ThreadPoolExecutor for embarrassingly parallel grading.
    """
    request_id = state.get("request_id", "")
    docs = state.get("retrieved_docs", [])
    sub_question = state.get("sub_question", "")

    if not docs:
        logger.debug("[%s] retrievalGrader: no documents to grade", request_id)
        return {"proceedToGenerate": False}

    def _grade_single(doc: Document, question: str) -> bool:
        """Grade a single document. Returns True if relevant."""
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", _GRADING_SYSTEM_PROMPT),
                ("human", "User question: {question}\n\nRetrieved document:\n{content}"),
            ])
            chain = prompt | get_llm("high").with_structured_output(GradeDocument)
            result = chain.invoke({"question": question, "content": doc.page_content})
            return result.score.strip().lower() == "yes"
        except Exception:
            logger.warning(
                "[%s] Grading exception for doc, including by default",
                request_id,
                exc_info=True,
            )
            # Fail open: include the doc, let generation handle irrelevance
            return True

    max_workers = min(8, len(docs))
    relevant_docs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_doc = {
            executor.submit(_grade_single, doc, sub_question): doc
            for doc in docs
        }
        for future in concurrent.futures.as_completed(future_to_doc):
            doc = future_to_doc[future]
            if future.result():
                relevant_docs.append(doc)

    logger.info(
        "[%s] retrievalGrader: %d/%d docs passed grading",
        request_id, len(relevant_docs), len(docs),
    )
    return {
        "retrieved_docs": relevant_docs,
        "proceedToGenerate": len(relevant_docs) > 0,
    }
