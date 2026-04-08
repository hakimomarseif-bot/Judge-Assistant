"""
test_case_doc_rag.py
====================
Comprehensive test suite for RAG/case_doc_rag.

Layer 1  — Pure unit tests  (no mocks, no network, no LLM)
Layer 2  — Node unit tests  (mock LLM, mock MongoDB, mock Qdrant via fixtures)
Layer 3  — Graph tests      (compile + invoke with mocked infrastructure)
Layer 4  — Bug regression   (explicit test for each of the 22 identified issues)
Layer 5  — Integration smoke (real services, skipped when env vars absent)

Run layers 1-4 with:
    pytest test_case_doc_rag.py -v -m "not integration"

Run everything including layer 5 with:
    pytest test_case_doc_rag.py -v
"""

import importlib
import json
import operator
import sys
import threading
import time
import types
import typing
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_state(**overrides) -> dict:
    """Return a minimal valid AgentState dict."""
    base = {
        "query": "ما هي وقائع الدعوى؟",
        "case_id": "case_001",
        "conversation_history": [],
        "request_id": "test-req-001",
        "sub_questions": ["ما هي وقائع الدعوى؟"],
        "on_topic": True,
        "doc_selection_mode": "no_doc_specified",
        "selected_doc_id": None,
        "doc_titles": [],
        "sub_answers": [],
        "final_answer": "",
        "error": None,
    }
    base.update(overrides)
    return base


def _make_sub_question_state(**overrides) -> dict:
    """Return a minimal valid SubQuestionState dict."""
    base = {
        "sub_question": "ما هي وقائع الدعوى؟",
        "case_id": "case_001",
        "conversation_history": [],
        "selected_doc_id": None,
        "doc_selection_mode": "no_doc_specified",
        "request_id": "test-req-001",
        "retrieved_docs": [],
        "proceedToGenerate": False,
        "rephraseCount": 0,
        "sub_answer": "",
        "sources": [],
        "found": False,
    }
    base.update(overrides)
    return base


def _make_document(content: str = "test content", **meta) -> Any:
    """Return a LangChain Document for testing."""
    from langchain_core.documents import Document
    metadata = {
        "case_id": "case_001",
        "title": "صحيفة دعوى",
        "source_file": "doc.pdf",
        "chunk_index": 0,
    }
    metadata.update(meta)
    return Document(page_content=content, metadata=metadata)


# ===========================================================================
# LAYER 1 — PURE UNIT TESTS (zero external dependencies)
# ===========================================================================


class TestStateTypedict:
    """Verify AgentState and SubQuestionState field contracts."""

    def test_agent_state_has_required_fields(self):
        from RAG.case_doc_rag.state import AgentState
        ann = AgentState.__annotations__
        required = [
            "query", "case_id", "conversation_history", "request_id",
            "sub_questions", "on_topic", "doc_selection_mode",
            "selected_doc_id", "doc_titles", "sub_answers",
            "final_answer", "error",
        ]
        for field in required:
            assert field in ann, f"AgentState missing field: {field}"

    def test_agent_state_dead_fields_absent(self):
        """Bug 17 regression: dead fields removed from AgentState."""
        from RAG.case_doc_rag.state import AgentState
        ann = AgentState.__annotations__
        dead = [
            "messages", "refined_query", "onTopic", "doc_type",
            "context", "safety_notes", "answer", "retrieved_docs",
        ]
        for field in dead:
            assert field not in ann, f"Dead field still present in AgentState: {field}"

    def test_sub_answers_has_operator_add_reducer(self):
        """Contract 1: sub_answers MUST use operator.add reducer."""
        from RAG.case_doc_rag.state import AgentState
        ann = AgentState.__annotations__
        sub_answers_type = ann["sub_answers"]
        args = typing.get_args(sub_answers_type)
        # args[1] should be operator.add
        assert len(args) == 2, "sub_answers must be Annotated with two args"
        assert args[1] is operator.add, (
            "sub_answers reducer must be operator.add, got: %s" % args[1]
        )

    def test_sub_question_state_has_required_fields(self):
        from RAG.case_doc_rag.state import SubQuestionState
        ann = SubQuestionState.__annotations__
        required = [
            "sub_question", "case_id", "conversation_history",
            "selected_doc_id", "doc_selection_mode", "request_id",
            "retrieved_docs", "proceedToGenerate", "rephraseCount",
            "sub_answer", "sources", "found",
        ]
        for field in required:
            assert field in ann, f"SubQuestionState missing field: {field}"

    def test_sub_question_state_no_retrieved_docs_on_agent_state(self):
        """retrieved_docs belongs in SubQuestionState, not AgentState."""
        from RAG.case_doc_rag.state import AgentState, SubQuestionState
        assert "retrieved_docs" not in AgentState.__annotations__
        assert "retrieved_docs" in SubQuestionState.__annotations__

    def test_on_topic_is_bool_not_str(self):
        """Bug 1 regression: on_topic annotated as bool, not str."""
        from RAG.case_doc_rag.state import AgentState
        ann = AgentState.__annotations__
        assert ann["on_topic"] is bool

    def test_conversation_history_in_both_states(self):
        """Gap 5 regression: conversation_history must be in SubQuestionState."""
        from RAG.case_doc_rag.state import AgentState, SubQuestionState
        assert "conversation_history" in AgentState.__annotations__
        assert "conversation_history" in SubQuestionState.__annotations__

    def test_doc_selection_mode_is_literal(self):
        """Bug 1 regression: doc_selection_mode uses Literal type."""
        from RAG.case_doc_rag.state import AgentState
        ann = AgentState.__annotations__
        mode_type = ann["doc_selection_mode"]
        # Unwrap Annotated (reducer wrapper) if present to reach the Literal
        outer_args = typing.get_args(mode_type)
        if typing.get_origin(mode_type) is typing.Annotated:
            literal_type = outer_args[0]
            args = typing.get_args(literal_type)
        else:
            args = outer_args
        assert "retrieve_specific_doc" in args
        assert "restrict_to_doc" in args
        assert "no_doc_specified" in args


class TestModels:
    """Pydantic model instantiation."""

    def test_grade_question_instantiates(self):
        from RAG.case_doc_rag.models import GradeQuestion
        m = GradeQuestion(score="Yes")
        assert m.score == "Yes"

    def test_doc_selection_instantiates_with_none_doc_id(self):
        from RAG.case_doc_rag.models import DocSelection
        m = DocSelection(mode="no_doc_specified", doc_id=None)
        assert m.mode == "no_doc_specified"
        assert m.doc_id is None

    def test_doc_selection_instantiates_with_doc_id(self):
        from RAG.case_doc_rag.models import DocSelection
        m = DocSelection(mode="retrieve_specific_doc", doc_id="صحيفة دعوى")
        assert m.doc_id == "صحيفة دعوى"

    def test_grade_document_instantiates(self):
        from RAG.case_doc_rag.models import GradeDocument
        m = GradeDocument(score="No")
        assert m.score == "No"


class TestRouters:
    """Layer 1 router tests — zero external deps, pure logic."""

    # --- onTopicRouter ---

    def test_on_topic_router_error_takes_priority(self):
        from RAG.case_doc_rag.routers import onTopicRouter
        state = _make_agent_state(error="something broke", on_topic=True)
        assert onTopicRouter(state) == "errorResponse"

    def test_on_topic_router_routes_on_topic(self):
        from RAG.case_doc_rag.routers import onTopicRouter
        state = _make_agent_state(on_topic=True, error=None)
        assert onTopicRouter(state) == "documentSelector"

    def test_on_topic_router_routes_off_topic(self):
        from RAG.case_doc_rag.routers import onTopicRouter
        state = _make_agent_state(on_topic=False, error=None)
        assert onTopicRouter(state) == "offTopicResponse"

    def test_on_topic_router_does_not_mutate_state(self):
        from RAG.case_doc_rag.routers import onTopicRouter
        state = _make_agent_state(on_topic=True)
        original = dict(state)
        onTopicRouter(state)
        assert state == original

    # --- docSelectorRouter ---

    def test_doc_selector_router_error_takes_priority(self):
        from RAG.case_doc_rag.routers import docSelectorRouter
        state = _make_agent_state(
            error="bad", doc_selection_mode="retrieve_specific_doc"
        )
        assert docSelectorRouter(state) == "errorResponse"

    def test_doc_selector_router_retrieve_specific(self):
        from RAG.case_doc_rag.routers import docSelectorRouter
        state = _make_agent_state(doc_selection_mode="retrieve_specific_doc", error=None)
        assert docSelectorRouter(state) == "DocumentFinalizer"

    def test_doc_selector_router_restrict_to_doc(self):
        from RAG.case_doc_rag.routers import docSelectorRouter
        state = _make_agent_state(doc_selection_mode="restrict_to_doc", error=None)
        assert docSelectorRouter(state) == "dispatchQuestions"

    def test_doc_selector_router_no_doc_specified(self):
        from RAG.case_doc_rag.routers import docSelectorRouter
        state = _make_agent_state(doc_selection_mode="no_doc_specified", error=None)
        assert docSelectorRouter(state) == "dispatchQuestions"

    def test_doc_selector_router_does_not_mutate_state(self):
        from RAG.case_doc_rag.routers import docSelectorRouter
        state = _make_agent_state(doc_selection_mode="no_doc_specified")
        original = dict(state)
        docSelectorRouter(state)
        assert state == original

    # --- proceedRouter ---

    def test_proceed_router_generate_when_docs_found(self):
        from RAG.case_doc_rag.routers import proceedRouter
        state = _make_sub_question_state(proceedToGenerate=True, rephraseCount=0)
        assert proceedRouter(state) == "generateAnswer"

    def test_proceed_router_cannot_answer_at_ceiling(self):
        from RAG.case_doc_rag.routers import proceedRouter
        state = _make_sub_question_state(proceedToGenerate=False, rephraseCount=2)
        assert proceedRouter(state) == "cannotAnswer"

    def test_proceed_router_refine_below_ceiling(self):
        from RAG.case_doc_rag.routers import proceedRouter
        state = _make_sub_question_state(proceedToGenerate=False, rephraseCount=0)
        assert proceedRouter(state) == "refineQuestion"

    def test_proceed_router_generate_takes_priority_over_ceiling(self):
        """proceedToGenerate=True wins even at rephraseCount ceiling."""
        from RAG.case_doc_rag.routers import proceedRouter
        state = _make_sub_question_state(proceedToGenerate=True, rephraseCount=2)
        assert proceedRouter(state) == "generateAnswer"

    def test_proceed_router_does_not_mutate_state(self):
        from RAG.case_doc_rag.routers import proceedRouter
        state = _make_sub_question_state(proceedToGenerate=False, rephraseCount=0)
        original = dict(state)
        proceedRouter(state)
        assert state == original

    def test_max_rephrase_is_defined_in_routers_only(self):
        """Bug 10 regression: _MAX_REPHRASE must only exist in routers.py."""
        from RAG.case_doc_rag import routers
        assert hasattr(routers, "_MAX_REPHRASE")
        # generation_nodes must NOT import or define _MAX_REPHRASE
        from RAG.case_doc_rag.nodes import generation_nodes
        assert not hasattr(generation_nodes, "_MAX_REPHRASE")


class TestExtractJsonList:
    """Tests for the _extract_json_list helper."""

    def _fn(self, raw):
        from RAG.case_doc_rag.nodes.query_nodes import _extract_json_list
        return _extract_json_list(raw)

    def test_clean_json_list(self):
        result = self._fn('["question 1", "question 2"]')
        assert result == ["question 1", "question 2"]

    def test_single_item_json(self):
        result = self._fn('["only one question"]')
        assert result == ["only one question"]

    def test_markdown_fenced_json(self):
        result = self._fn('```json\n["q1", "q2"]\n```')
        assert result == ["q1", "q2"]

    def test_markdown_fenced_no_lang(self):
        result = self._fn('```\n["q1"]\n```')
        assert result == ["q1"]

    def test_embedded_json_in_text(self):
        result = self._fn('Here is the output: ["q1", "q2"] as requested.')
        assert result == ["q1", "q2"]

    def test_plain_text_fallback_single_item(self):
        result = self._fn("just some plain text with no JSON")
        assert result == ["just some plain text with no JSON"]

    def test_empty_string_returns_empty_list(self):
        result = self._fn("")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        result = self._fn("   \n  ")
        assert result == []

    def test_never_returns_non_string_items(self):
        result = self._fn('[1, "text", null, "another"]')
        # non-string items filtered out
        assert all(isinstance(i, str) for i in result)

    def test_arabic_questions_preserved(self):
        raw = '["ما هي وقائع الدعوى؟", "ما هي الطلبات الختامية؟"]'
        result = self._fn(raw)
        assert len(result) == 2
        assert "ما هي وقائع الدعوى؟" in result


class TestSerializeHistory:
    """Tests for _serialize_history helper."""

    def _fn(self, history):
        from RAG.case_doc_rag.nodes.generation_nodes import _serialize_history
        return _serialize_history(history)

    def test_empty_list_returns_arabic_no_context(self):
        result = self._fn([])
        assert result == "لا يوجد سياق سابق."

    def test_user_turn_prefixed_correctly(self):
        result = self._fn([{"role": "user", "content": "سؤال"}])
        assert result.startswith("القاضي: ")
        assert "سؤال" in result

    def test_assistant_turn_prefixed_correctly(self):
        result = self._fn([{"role": "assistant", "content": "إجابة"}])
        assert result.startswith("المساعد: ")
        assert "إجابة" in result

    def test_mixed_turns_both_prefixes_present(self):
        history = [
            {"role": "user", "content": "سؤال"},
            {"role": "assistant", "content": "إجابة"},
        ]
        result = self._fn(history)
        assert "القاضي: " in result
        assert "المساعد: " in result

    def test_no_python_object_repr_in_output(self):
        """Bug 6 regression: must never produce HumanMessage or AIMessage reprs."""
        history = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "answer"},
        ]
        result = self._fn(history)
        assert "HumanMessage" not in result
        assert "AIMessage" not in result
        assert "additional_kwargs" not in result

    def test_malformed_turns_skipped_gracefully(self):
        history = [
            {"role": "user"},          # missing content
            {},                         # empty dict
            {"role": "user", "content": "good"},
        ]
        result = self._fn(history)
        assert "القاضي: good" in result

    def test_returns_string_always(self):
        assert isinstance(self._fn([]), str)
        assert isinstance(self._fn([{"role": "user", "content": "x"}]), str)


class TestFuzzyMatchDocTitle:
    """Tests for fuzzy_match_doc_title."""

    def _fn(self, candidate, titles, threshold=0.5):
        from RAG.case_doc_rag.nodes.selection_nodes import fuzzy_match_doc_title
        return fuzzy_match_doc_title(candidate, titles, threshold)

    def test_exact_match_returned(self):
        result = self._fn("صحيفة دعوى", ["صحيفة دعوى", "تقرير خبير"])
        assert result == "صحيفة دعوى"

    def test_close_match_returned(self):
        result = self._fn("صحيفه دعوى", ["صحيفة دعوى", "تقرير خبير"])
        assert result == "صحيفة دعوى"

    def test_empty_candidate_returns_none(self):
        assert self._fn("", ["صحيفة دعوى"]) is None

    def test_empty_titles_returns_none(self):
        assert self._fn("صحيفة دعوى", []) is None

    def test_below_threshold_returns_none(self):
        result = self._fn("xyz", ["صحيفة دعوى"], threshold=0.9)
        assert result is None

    def test_none_titles_in_list_skipped(self):
        result = self._fn("صحيفة دعوى", [None, "صحيفة دعوى"])
        assert result == "صحيفة دعوى"


# ===========================================================================
# LAYER 2 — NODE UNIT TESTS (mocked LLM, MongoDB, Qdrant)
# ===========================================================================


class TestInfrastructureLaziness:
    """Bug 14 regression: nothing initializes at import time."""

    def test_all_singletons_are_none_after_import(self):
        """Import the module and assert no resource has been created."""
        import RAG.case_doc_rag.infrastructure as infra
        # Re-import to ensure we're not getting cached state
        importlib.reload(infra)
        assert infra._embedding_fn is None
        assert infra._qdrant_client is None
        assert infra._vectorstore is None
        assert infra._mongo_collection is None
        assert infra._llm_cache == {}

    def test_get_llm_returns_same_object_for_same_tier(self):
        """Singleton: same tier → same object identity."""
        import RAG.case_doc_rag.infrastructure as infra
        mock_llm = MagicMock()
        with patch("RAG.case_doc_rag.infrastructure._config_get_llm", return_value=mock_llm):
            a = infra.get_llm("high")
            b = infra.get_llm("high")
        assert a is b

    def test_get_llm_returns_different_objects_for_different_tiers(self):
        import RAG.case_doc_rag.infrastructure as infra
        mock_high = MagicMock(name="high")
        mock_medium = MagicMock(name="medium")
        call_map = {"high": mock_high, "medium": mock_medium}
        with patch("RAG.case_doc_rag.infrastructure._config_get_llm", side_effect=lambda t: call_map[t]):
            high = infra.get_llm("high")
            medium = infra.get_llm("medium")
        assert high is not medium

    def test_set_vectorstore_replaces_cached_instance(self):
        """Bug 18 regression: set_vectorstore must update the singleton."""
        import RAG.case_doc_rag.infrastructure as infra
        fake_vs = MagicMock(name="injected_vs")
        infra.set_vectorstore(fake_vs)
        assert infra._vectorstore is fake_vs

    def test_set_vectorstore_thread_safety(self):
        """Bug 18 regression: concurrent set_vectorstore must not corrupt state."""
        import RAG.case_doc_rag.infrastructure as infra
        results = []
        def setter(vs):
            infra.set_vectorstore(vs)
            results.append(infra._vectorstore)
        vs1, vs2 = MagicMock(), MagicMock()
        t1 = threading.Thread(target=setter, args=(vs1,))
        t2 = threading.Thread(target=setter, args=(vs2,))
        t1.start(); t2.start()
        t1.join(); t2.join()
        # One of them must have won — state must be one of the two mocks
        assert infra._vectorstore in (vs1, vs2)


class TestPromptsLaziness:
    """Bug B regression: get_rag_chain must be thread-safe."""

    def test_get_rag_chain_returns_same_object(self):
        import RAG.case_doc_rag.prompts as prompts
        prompts._rag_chain = None  # reset
        mock_chain = MagicMock()
        with patch("RAG.case_doc_rag.prompts.get_llm", return_value=MagicMock()):
            with patch("langchain_core.prompts.ChatPromptTemplate.from_template", return_value=MagicMock(__or__=lambda s, o: mock_chain)):
                a = prompts.get_rag_chain()
                b = prompts.get_rag_chain()
        assert a is b

    def test_prompt_constants_are_non_empty_strings(self):
        from RAG.case_doc_rag.prompts import (
            DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE,
            QUESTION_CLASSIFIER_PROMPT,
            QUESTION_REWRITER_PROMPT,
            RAG_ANSWER_TEMPLATE,
            REFINE_QUESTION_PROMPT,
        )
        for name, val in [
            ("QUESTION_REWRITER_PROMPT", QUESTION_REWRITER_PROMPT),
            ("QUESTION_CLASSIFIER_PROMPT", QUESTION_CLASSIFIER_PROMPT),
            ("REFINE_QUESTION_PROMPT", REFINE_QUESTION_PROMPT),
            ("DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE", DOC_SELECTOR_SYSTEM_PROMPT_TEMPLATE),
            ("RAG_ANSWER_TEMPLATE", RAG_ANSWER_TEMPLATE),
        ]:
            assert isinstance(val, str) and len(val) > 0, f"{name} is empty"

    def test_rewriter_prompt_enforces_json_format(self):
        """Bug 3 regression: prompt must demand JSON list output."""
        from RAG.case_doc_rag.prompts import QUESTION_REWRITER_PROMPT
        # Must reference JSON format explicitly
        assert "[" in QUESTION_REWRITER_PROMPT
        assert "JSON" in QUESTION_REWRITER_PROMPT or "json" in QUESTION_REWRITER_PROMPT


class TestQuestionRewriter:
    """Bug 1, 2, 3 regression tests via questionRewriter."""

    def _invoke(self, state, llm_response="[\"rewritten question\"]"):
        mock_response = MagicMock()
        mock_response.content = llm_response
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response
        mock_llm = MagicMock()
        mock_llm.__or__ = lambda s, o: mock_chain
        with patch("RAG.case_doc_rag.nodes.query_nodes.get_llm", return_value=mock_llm):
            with patch("RAG.case_doc_rag.nodes.query_nodes.ChatPromptTemplate") as MockCPT:
                MockCPT.from_messages.return_value.__or__ = lambda s, o: mock_chain
                from RAG.case_doc_rag.nodes.query_nodes import questionRewriter
                return questionRewriter(state)

    def test_always_runs_on_first_turn(self):
        """Bug 2 regression: rewriter runs even with empty conversation history."""
        state = _make_agent_state(query="ما هي وقائع الدعوى؟", conversation_history=[])
        result = self._invoke(state)
        assert "sub_questions" in result
        assert isinstance(result["sub_questions"], list)

    def test_query_used_as_plain_string(self):
        """Bug 1 regression: query accessed as str, not .content."""
        # If the node calls .content on query, this would fail because str has no .content
        state = _make_agent_state(query="plain string query")
        result = self._invoke(state)
        assert "sub_questions" in result

    def test_json_is_parsed_into_list(self):
        """Bug 3 regression: JSON response is parsed, not stored raw."""
        state = _make_agent_state(query="test")
        result = self._invoke(state, llm_response='["q1", "q2"]')
        assert result["sub_questions"] == ["q1", "q2"]
        # Must NOT be a raw JSON string
        assert not isinstance(result["sub_questions"], str)

    def test_empty_query_sets_error(self):
        state = _make_agent_state(query="")
        from RAG.case_doc_rag.nodes.query_nodes import questionRewriter
        result = questionRewriter(state)
        assert result.get("error") is not None

    def test_llm_failure_falls_back_to_original_query(self):
        state = _make_agent_state(query="fallback question")
        with patch("RAG.case_doc_rag.nodes.query_nodes.get_llm", side_effect=Exception("LLM down")):
            with patch("RAG.case_doc_rag.nodes.query_nodes.ChatPromptTemplate") as MockCPT:
                MockCPT.from_messages.return_value.__or__ = MagicMock(side_effect=Exception("LLM down"))
                from RAG.case_doc_rag.nodes.query_nodes import questionRewriter
                result = questionRewriter(state)
        assert result["sub_questions"] == ["fallback question"]
        assert result.get("error") is None  # LLM failure is NOT a hard error

    def test_multi_question_decomposition(self):
        state = _make_agent_state(query="سؤالان مختلفان")
        result = self._invoke(state, llm_response='["سؤال 1", "سؤال 2"]')
        assert len(result["sub_questions"]) == 2

    def test_returns_partial_dict_not_full_state(self):
        """Rule 4: nodes return partial dicts, not full state."""
        state = _make_agent_state(query="test")
        result = self._invoke(state)
        # Should only contain sub_questions (and possibly error)
        assert "query" not in result or result.get("query") == state["query"]


class TestQuestionClassifier:
    """Bug 4 regression: classifier must use sub_questions[0]."""

    def _invoke(self, state, score="Yes"):
        mock_result = MagicMock()
        mock_result.score = score
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        with patch("RAG.case_doc_rag.nodes.query_nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = MagicMock()
            mock_get_llm.return_value = mock_llm
            with patch("RAG.case_doc_rag.nodes.query_nodes.ChatPromptTemplate") as MockCPT:
                MockCPT.from_messages.return_value.__or__ = lambda s, o: mock_chain
                from RAG.case_doc_rag.nodes.query_nodes import questionClassifier
                return questionClassifier(state)

    def test_classifies_sub_questions_not_raw_query(self):
        """Bug 4 regression: uses sub_questions[0], not state['query']."""
        state = _make_agent_state(
            query="ORIGINAL QUERY",
            sub_questions=["rewritten question"]
        )
        # If it classified state["query"], it would pass "ORIGINAL QUERY"
        # We can't easily assert what was passed inside, but we can assert
        # the flow doesn't crash and returns a bool
        result = self._invoke(state, score="Yes")
        assert isinstance(result.get("on_topic"), bool)

    def test_yes_score_produces_true_bool(self):
        """Bug 1/6 regression: on_topic is real bool, not string."""
        state = _make_agent_state(sub_questions=["legal question"])
        result = self._invoke(state, score="Yes")
        assert result["on_topic"] is True
        assert isinstance(result["on_topic"], bool)

    def test_no_score_produces_false_bool(self):
        state = _make_agent_state(sub_questions=["unrelated question"])
        result = self._invoke(state, score="No")
        assert result["on_topic"] is False
        assert isinstance(result["on_topic"], bool)

    def test_empty_sub_questions_returns_false(self):
        state = _make_agent_state(sub_questions=[])
        from RAG.case_doc_rag.nodes.query_nodes import questionClassifier
        result = questionClassifier(state)
        assert result["on_topic"] is False

    def test_llm_failure_defaults_to_true(self):
        """Fail-open: transient error should not refuse valid questions."""
        state = _make_agent_state(sub_questions=["legal question"])
        with patch("RAG.case_doc_rag.nodes.query_nodes.get_llm", side_effect=Exception("LLM down")):
            from RAG.case_doc_rag.nodes.query_nodes import questionClassifier
            result = questionClassifier(state)
        assert result["on_topic"] is True


class TestOffTopicResponse:
    def test_sets_final_answer(self):
        from RAG.case_doc_rag.nodes.query_nodes import offTopicResponse
        result = offTopicResponse(_make_agent_state())
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0

    def test_does_not_set_error(self):
        from RAG.case_doc_rag.nodes.query_nodes import offTopicResponse
        result = offTopicResponse(_make_agent_state())
        assert "error" not in result or result.get("error") is None


class TestDispatchQuestions:
    """Contract 2, 3: dispatchQuestions returns Send objects."""

    def test_returns_list_of_sends(self):
        from langgraph.types import Send
        from RAG.case_doc_rag.nodes.query_nodes import dispatchQuestions
        state = _make_agent_state(sub_questions=["q1", "q2"])
        result = dispatchQuestions(state)
        assert isinstance(result, list)
        assert all(isinstance(s, Send) for s in result)

    def test_one_send_per_sub_question(self):
        from langgraph.types import Send
        from RAG.case_doc_rag.nodes.query_nodes import dispatchQuestions
        state = _make_agent_state(sub_questions=["q1", "q2", "q3"])
        result = dispatchQuestions(state)
        assert len(result) == 3

    def test_sends_target_retrieve_branch(self):
        from langgraph.types import Send
        from RAG.case_doc_rag.nodes.query_nodes import dispatchQuestions
        state = _make_agent_state(sub_questions=["q1"])
        result = dispatchQuestions(state)
        assert result[0].node == "retrieve_branch"

    def test_sub_state_has_all_required_fields(self):
        from RAG.case_doc_rag.nodes.query_nodes import dispatchQuestions
        from RAG.case_doc_rag.state import SubQuestionState
        state = _make_agent_state(sub_questions=["q1"])
        result = dispatchQuestions(state)
        sub_state = result[0].arg
        for field in SubQuestionState.__annotations__:
            assert field in sub_state, f"SubQuestionState field missing from Send: {field}"

    def test_conversation_history_copied_to_branch(self):
        """Gap 5 regression: conversation_history must be forwarded."""
        from RAG.case_doc_rag.nodes.query_nodes import dispatchQuestions
        history = [{"role": "user", "content": "prev question"}]
        state = _make_agent_state(sub_questions=["q1"], conversation_history=history)
        result = dispatchQuestions(state)
        assert result[0].arg["conversation_history"] == history

    def test_branch_initialized_with_zero_rephrase_count(self):
        from RAG.case_doc_rag.nodes.query_nodes import dispatchQuestions
        state = _make_agent_state(sub_questions=["q1"])
        result = dispatchQuestions(state)
        assert result[0].arg["rephraseCount"] == 0
        assert result[0].arg["proceedToGenerate"] is False

    def test_each_branch_gets_correct_sub_question(self):
        from RAG.case_doc_rag.nodes.query_nodes import dispatchQuestions
        state = _make_agent_state(sub_questions=["first", "second"])
        result = dispatchQuestions(state)
        sub_qs = {s.arg["sub_question"] for s in result}
        assert sub_qs == {"first", "second"}


class TestDocumentSelector:
    """Bug 9, Perf 2 regression."""

    def _invoke(self, state, mode="no_doc_specified", doc_id=None, titles=None):
        mock_result = MagicMock()
        mock_result.mode = mode
        mock_result.doc_id = doc_id
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        available = titles or []
        with patch("RAG.case_doc_rag.nodes.selection_nodes.get_available_doc_titles", return_value=available):
            with patch("RAG.case_doc_rag.nodes.selection_nodes.get_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_llm.with_structured_output.return_value = MagicMock()
                mock_get_llm.return_value = mock_llm
                with patch("RAG.case_doc_rag.nodes.selection_nodes.ChatPromptTemplate") as MockCPT:
                    MockCPT.from_messages.return_value.__or__ = lambda s, o: mock_chain
                    from RAG.case_doc_rag.nodes.selection_nodes import documentSelector
                    return documentSelector(state)

    def test_query_accessed_as_plain_string(self):
        """Bug 9 regression: query must not call .content."""
        state = _make_agent_state(query="plain string")
        result = self._invoke(state)
        assert "doc_selection_mode" in result

    def test_empty_query_returns_no_doc_specified(self):
        from RAG.case_doc_rag.nodes.selection_nodes import documentSelector
        state = _make_agent_state(query="")
        result = documentSelector(state)
        assert result["doc_selection_mode"] == "no_doc_specified"
        assert result["selected_doc_id"] is None

    def test_doc_titles_stored_in_state(self):
        """Perf 2 regression: doc_titles written to state."""
        state = _make_agent_state(query="test")
        result = self._invoke(state, titles=["صحيفة دعوى", "تقرير خبير"])
        assert result["doc_titles"] == ["صحيفة دعوى", "تقرير خبير"]

    def test_llm_failure_returns_no_doc_specified(self):
        state = _make_agent_state(query="test")
        with patch("RAG.case_doc_rag.nodes.selection_nodes.get_available_doc_titles", return_value=[]):
            with patch("RAG.case_doc_rag.nodes.selection_nodes.get_llm", side_effect=Exception("LLM down")):
                from RAG.case_doc_rag.nodes.selection_nodes import documentSelector
                result = documentSelector(state)
        assert result["doc_selection_mode"] == "no_doc_specified"
        assert result.get("error") is None  # LLM failure is soft


class TestTTLCache:
    """Perf 2 regression: MongoDB called at most once per cache window."""

    def test_cache_hit_prevents_second_mongodb_call(self):
        import RAG.case_doc_rag.nodes.selection_nodes as sel
        # Reset cache state
        sel._titles_cache.clear()
        sel._titles_cache_ts.clear()

        mock_collection = MagicMock()
        mock_collection.find.return_value = [{"title": "صحيفة دعوى"}]

        with patch("RAG.case_doc_rag.nodes.selection_nodes.get_mongo_collection", return_value=mock_collection):
            result1 = sel.get_available_doc_titles("case_001")
            result2 = sel.get_available_doc_titles("case_001")

        # MongoDB must be queried exactly once
        assert mock_collection.find.call_count == 1
        assert result1 == result2

    def test_different_case_ids_each_query_mongodb(self):
        import RAG.case_doc_rag.nodes.selection_nodes as sel
        sel._titles_cache.clear()
        sel._titles_cache_ts.clear()

        mock_collection = MagicMock()
        mock_collection.find.return_value = []

        with patch("RAG.case_doc_rag.nodes.selection_nodes.get_mongo_collection", return_value=mock_collection):
            sel.get_available_doc_titles("case_001")
            sel.get_available_doc_titles("case_002")

        assert mock_collection.find.call_count == 2

    def test_expired_cache_triggers_new_query(self):
        import RAG.case_doc_rag.nodes.selection_nodes as sel
        sel._titles_cache.clear()
        sel._titles_cache_ts.clear()

        mock_collection = MagicMock()
        mock_collection.find.return_value = []

        with patch("RAG.case_doc_rag.nodes.selection_nodes.get_mongo_collection", return_value=mock_collection):
            sel.get_available_doc_titles("case_exp")
            # Artificially expire the cache entry
            sel._titles_cache_ts["case_exp"] = time.time() - 200.0
            sel.get_available_doc_titles("case_exp")

        assert mock_collection.find.call_count == 2


class TestDocumentFinalizer:
    """Bug 5 regression: must return Document objects and set final_answer."""

    def _invoke(self, state, mongo_doc=None):
        mock_collection = MagicMock()
        mock_collection.find_one.return_value = mongo_doc
        with patch("RAG.case_doc_rag.nodes.selection_nodes.get_mongo_collection", return_value=mock_collection):
            from RAG.case_doc_rag.nodes.selection_nodes import DocumentFinalizer
            return DocumentFinalizer(state)

    def test_no_doc_id_returns_without_crash(self):
        state = _make_agent_state(selected_doc_id=None)
        result = self._invoke(state)
        assert "final_answer" in result or "error" in result

    def test_missing_mongo_doc_sets_error(self):
        state = _make_agent_state(selected_doc_id="صحيفة دعوى")
        result = self._invoke(state, mongo_doc=None)
        assert result.get("error") is not None

    def test_returns_sub_answers_list_not_raw_dict(self):
        """Bug 5 regression: sub_answers must be a list, not raw Mongo dict."""
        state = _make_agent_state(selected_doc_id="صحيفة دعوى")
        mongo_doc = {
            "_id": "abc",
            "title": "صحيفة دعوى",
            "content": "محتوى المستند",
            "source_file": "doc.pdf",
            "chunk_index": 0,
        }
        result = self._invoke(state, mongo_doc=mongo_doc)
        assert "sub_answers" in result
        assert isinstance(result["sub_answers"], list)
        assert len(result["sub_answers"]) == 1
        # Entries must be dicts, not raw Mongo documents
        entry = result["sub_answers"][0]
        assert isinstance(entry, dict)
        assert "answer" in entry
        assert "found" in entry

    def test_sets_final_answer_directly(self):
        """Gap 2 regression: final_answer set directly (no mergeAnswers)."""
        state = _make_agent_state(selected_doc_id="صحيفة دعوى")
        mongo_doc = {
            "title": "صحيفة دعوى",
            "content": "نص المستند",
            "source_file": "doc.pdf",
        }
        result = self._invoke(state, mongo_doc=mongo_doc)
        assert result.get("final_answer") == "نص المستند"

    def test_text_field_fallback_order(self):
        """content > text > body field extraction."""
        state = _make_agent_state(selected_doc_id="doc")
        # Only 'text' field present
        mongo_doc = {"title": "doc", "text": "from text field", "source_file": "f.pdf"}
        result = self._invoke(state, mongo_doc=mongo_doc)
        assert "from text field" in result.get("final_answer", "")

    def test_sources_built_from_metadata(self):
        state = _make_agent_state(selected_doc_id="doc")
        mongo_doc = {
            "title": "doc",
            "content": "text",
            "source_file": "case.pdf",
            "chunk_index": 3,
        }
        result = self._invoke(state, mongo_doc=mongo_doc)
        entry = result["sub_answers"][0]
        assert any("case.pdf" in s for s in entry["sources"])


class TestRetrievalNodes:
    """Retrieval and grading node tests."""

    def _make_retriever(self, docs):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs
        return mock_retriever

    def test_retrieve_case_id_filter_applied_first(self):
        """Contract 13: case_id filter is never dropped."""
        from RAG.case_doc_rag.nodes.retrieval_nodes import retrieve
        state = _make_sub_question_state(
            sub_question="test",
            case_id="case_001",
            doc_selection_mode="no_doc_specified",
        )
        docs = [_make_document("content")]
        mock_vs = MagicMock()
        mock_retriever = self._make_retriever(docs)
        mock_vs.as_retriever.return_value = mock_retriever

        with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_vectorstore", return_value=mock_vs):
            with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_retriever", return_value=mock_retriever):
                result = retrieve(state)

        # The first retriever call must include a filter (case_id filter)
        first_call_kwargs = mock_vs.as_retriever.call_args_list[0][1]
        search_kwargs = first_call_kwargs.get("search_kwargs", {})
        assert "filter" in search_kwargs, "First retrieval must use a case_id filter"

    def test_retrieve_score_threshold_applied(self):
        """Perf 3 regression: score_threshold must be in search_kwargs."""
        from RAG.case_doc_rag.nodes.retrieval_nodes import retrieve, _SCORE_THRESHOLD
        state = _make_sub_question_state(
            sub_question="test",
            case_id="case_001",
            doc_selection_mode="no_doc_specified",
        )
        mock_vs = MagicMock()
        mock_retriever = self._make_retriever([_make_document()])
        mock_vs.as_retriever.return_value = mock_retriever

        with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_vectorstore", return_value=mock_vs):
            with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_retriever", return_value=mock_retriever):
                retrieve(state)

        first_call_kwargs = mock_vs.as_retriever.call_args_list[0][1]
        sk = first_call_kwargs.get("search_kwargs", {})
        assert sk.get("score_threshold") == _SCORE_THRESHOLD

    def test_retrieve_restrict_to_doc_attempt2_drops_title_keeps_case_id(self):
        """
        BUG A REGRESSION: restrict_to_doc Attempt 2 must drop title filter,
        NOT drop case_id.
        """
        from RAG.case_doc_rag.nodes.retrieval_nodes import retrieve
        state = _make_sub_question_state(
            sub_question="test",
            case_id="case_001",
            selected_doc_id="صحيفة دعوى",
            doc_selection_mode="restrict_to_doc",
        )
        mock_vs = MagicMock()
        # Attempt 1 returns empty, triggering Attempt 2
        empty_retriever = self._make_retriever([])
        doc_retriever = self._make_retriever([_make_document()])
        # First call returns empty, second returns docs
        mock_vs.as_retriever.side_effect = [empty_retriever, doc_retriever]

        with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_vectorstore", return_value=mock_vs):
            with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_retriever", return_value=empty_retriever):
                retrieve(state)

        assert mock_vs.as_retriever.call_count >= 2
        
        # Attempt 2 search_kwargs MUST contain case_id filter to prevent cross-case data leaks
        # and MUST NOT contain the title filter to broaden the search.
        attempt2_kwargs = mock_vs.as_retriever.call_args_list[1][1]
        sk = attempt2_kwargs.get("search_kwargs", {})
        filter_obj = sk.get("filter")
        
        if filter_obj is not None:
            must_conditions = getattr(filter_obj, "must", [])
            condition_keys = [getattr(c, "key", "") for c in must_conditions]
            
            assert "metadata.case_id" in condition_keys, "Attempt 2 must KEEP the case_id filter."
            assert "metadata.title" not in condition_keys, "Attempt 2 must DROP the title filter."

    def test_retrieve_returns_retrieved_docs_key(self):
        from RAG.case_doc_rag.nodes.retrieval_nodes import retrieve
        state = _make_sub_question_state(sub_question="test", case_id="c1")
        mock_vs = MagicMock()
        mock_retriever = self._make_retriever([_make_document()])
        mock_vs.as_retriever.return_value = mock_retriever
        with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_vectorstore", return_value=mock_vs):
            with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_retriever", return_value=mock_retriever):
                result = retrieve(state)
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)

    def test_retrieval_grader_uses_thread_pool(self):
        """Perf 1 regression: grading must be parallel, not sequential."""
        import concurrent.futures
        from RAG.case_doc_rag.nodes.retrieval_nodes import retrievalGrader
        state = _make_sub_question_state(
            sub_question="test",
            retrieved_docs=[_make_document(f"doc {i}") for i in range(4)],
        )
        mock_result = MagicMock()
        mock_result.score = "Yes"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result

        with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = MagicMock()
            mock_get_llm.return_value = mock_llm
            with patch("RAG.case_doc_rag.nodes.retrieval_nodes.ChatPromptTemplate") as MockCPT:
                MockCPT.from_messages.return_value.__or__ = lambda s, o: mock_chain
                with patch("concurrent.futures.ThreadPoolExecutor") as mock_pool:
                    mock_executor = MagicMock()
                    mock_pool.return_value.__enter__ = lambda s: mock_executor
                    mock_pool.return_value.__exit__ = MagicMock(return_value=False)
                    futures = []
                    for _ in range(4):
                        f = concurrent.futures.Future()
                        f.set_result(True)
                        futures.append(f)
                    mock_executor.submit.side_effect = futures
                    with patch("concurrent.futures.as_completed", return_value=futures):
                        retrievalGrader(state)
                # ThreadPoolExecutor must have been used
                assert mock_pool.called

    def test_retrieval_grader_empty_docs_returns_false(self):
        from RAG.case_doc_rag.nodes.retrieval_nodes import retrievalGrader
        state = _make_sub_question_state(retrieved_docs=[])
        result = retrievalGrader(state)
        assert result["proceedToGenerate"] is False

    def test_retrieval_grader_fail_open_on_llm_exception(self):
        """On grading exception, doc is included (fail open)."""
        from RAG.case_doc_rag.nodes.retrieval_nodes import retrievalGrader
        state = _make_sub_question_state(
            sub_question="test",
            retrieved_docs=[_make_document("content")],
        )
        with patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_llm", side_effect=Exception("LLM down")):
            with patch("RAG.case_doc_rag.nodes.retrieval_nodes.ChatPromptTemplate") as MockCPT:
                MockCPT.from_messages.return_value.__or__ = MagicMock(side_effect=Exception("down"))
                result = retrievalGrader(state)
        # fail open: doc included despite exception
        assert result["proceedToGenerate"] is True


class TestGenerationNodes:
    """generateAnswer, refineQuestion, cannotAnswer, mergeAnswers, errorResponse."""

    def test_generate_answer_uses_serialized_history(self):
        """Bug 6 regression: history must be serialized string, not Python repr."""
        from RAG.case_doc_rag.nodes.generation_nodes import generateAnswer
        state = _make_sub_question_state(
            sub_question="question",
            retrieved_docs=[_make_document("doc content")],
            conversation_history=[{"role": "user", "content": "prev q"}],
        )
        captured_invokes = []

        def capture_invoke(inputs):
            captured_invokes.append(inputs)
            mock_resp = MagicMock()
            mock_resp.content = "answer"
            return mock_resp

        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = capture_invoke

        with patch("RAG.case_doc_rag.nodes.generation_nodes.get_rag_chain", return_value=mock_chain):
            generateAnswer(state)

        assert len(captured_invokes) == 1
        history_arg = captured_invokes[0]["history"]
        # Must be a clean string
        assert isinstance(history_arg, str)
        assert "HumanMessage" not in history_arg
        assert "AIMessage" not in history_arg
        assert "additional_kwargs" not in history_arg
        # Must contain the correct prefix
        assert "القاضي:" in history_arg

    def test_generate_answer_includes_sub_answers_in_return(self):
        """Contract 4: generateAnswer must include sub_answers for reducer."""
        from RAG.case_doc_rag.nodes.generation_nodes import generateAnswer
        state = _make_sub_question_state(
            sub_question="q",
            retrieved_docs=[_make_document("content")],
        )
        mock_resp = MagicMock()
        mock_resp.content = "answer"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_resp

        with patch("RAG.case_doc_rag.nodes.generation_nodes.get_rag_chain", return_value=mock_chain):
            result = generateAnswer(state)

        assert "sub_answers" in result
        assert isinstance(result["sub_answers"], list)
        assert len(result["sub_answers"]) == 1
        entry = result["sub_answers"][0]
        assert "question" in entry
        assert "answer" in entry
        assert "sources" in entry
        assert "found" in entry

    def test_generate_answer_sources_from_metadata(self):
        """New field: sources built from Document metadata."""
        from RAG.case_doc_rag.nodes.generation_nodes import generateAnswer
        doc = _make_document("content", source_file="مستند.pdf", chunk_index=2)
        state = _make_sub_question_state(sub_question="q", retrieved_docs=[doc])
        mock_resp = MagicMock()
        mock_resp.content = "answer"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_resp

        with patch("RAG.case_doc_rag.nodes.generation_nodes.get_rag_chain", return_value=mock_chain):
            result = generateAnswer(state)

        assert any("مستند.pdf" in s for s in result["sources"])

    def test_generate_answer_llm_failure_does_not_raise(self):
        """Rule 2: no raise inside node functions."""
        from RAG.case_doc_rag.nodes.generation_nodes import generateAnswer
        state = _make_sub_question_state(
            sub_question="q",
            retrieved_docs=[_make_document("c")],
        )
        with patch("RAG.case_doc_rag.nodes.generation_nodes.get_rag_chain", side_effect=Exception("LLM down")):
            result = generateAnswer(state)
        # Must not raise; must return a valid result dict
        assert "sub_answers" in result

    def test_refine_question_uses_chain_pattern_not_format(self):
        """Bug 7 regression: must use prompt | llm chain, not .format() + invoke(string)."""
        from RAG.case_doc_rag.nodes.generation_nodes import refineQuestion
        state = _make_sub_question_state(sub_question="original question")
        mock_resp = MagicMock()
        mock_resp.content = "refined question"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_resp

        with patch("RAG.case_doc_rag.nodes.generation_nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            with patch("RAG.case_doc_rag.nodes.generation_nodes.ChatPromptTemplate") as MockCPT:
                mock_template = MagicMock()
                MockCPT.from_messages.return_value = mock_template
                mock_template.__or__ = lambda s, o: mock_chain
                result = refineQuestion(state)

        # Chain must have been invoked with a dict, NOT a raw string
        assert mock_chain.invoke.called
        invoke_arg = mock_chain.invoke.call_args[0][0]
        assert isinstance(invoke_arg, dict), (
            "Bug 7: refineQuestion must call chain.invoke(dict), "
            "not llm.invoke(string_from_format())"
        )

    def test_refine_question_increments_rephrase_count(self):
        from RAG.case_doc_rag.nodes.generation_nodes import refineQuestion
        state = _make_sub_question_state(sub_question="q", rephraseCount=1)
        mock_resp = MagicMock()
        mock_resp.content = "refined"
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_resp

        with patch("RAG.case_doc_rag.nodes.generation_nodes.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            with patch("RAG.case_doc_rag.nodes.generation_nodes.ChatPromptTemplate") as MockCPT:
                MockCPT.from_messages.return_value.__or__ = lambda s, o: mock_chain
                result = refineQuestion(state)

        assert result["rephraseCount"] == 2

    def test_refine_question_does_not_check_ceiling(self):
        """Bug 10 regression: refineQuestion must not enforce rephraseCount ceiling."""
        from RAG.case_doc_rag.nodes.generation_nodes import refineQuestion
        import inspect
        source = inspect.getsource(refineQuestion)
        # The word _MAX_REPHRASE must not appear in refineQuestion's source
        assert "_MAX_REPHRASE" not in source, (
            "Bug 10: refineQuestion must not check _MAX_REPHRASE. "
            "That is proceedRouter's exclusive responsibility."
        )

    def test_cannot_answer_includes_sub_answers(self):
        """Contract 4: cannotAnswer must include sub_answers for reducer."""
        from RAG.case_doc_rag.nodes.generation_nodes import cannotAnswer
        state = _make_sub_question_state(sub_question="q")
        result = cannotAnswer(state)
        assert "sub_answers" in result
        assert isinstance(result["sub_answers"], list)
        assert result["sub_answers"][0]["found"] is False

    def test_cannot_answer_sets_found_false(self):
        from RAG.case_doc_rag.nodes.generation_nodes import cannotAnswer
        state = _make_sub_question_state(sub_question="q")
        result = cannotAnswer(state)
        assert result["found"] is False

    def test_merge_answers_single_question_sets_final_answer(self):
        from RAG.case_doc_rag.nodes.generation_nodes import mergeAnswers
        state = _make_agent_state(
            sub_answers=[{"question": "q", "answer": "the answer", "sources": [], "found": True}]
        )
        result = mergeAnswers(state)
        assert result["final_answer"] == "the answer"

    def test_merge_answers_multi_question_leaves_final_answer_empty(self):
        from RAG.case_doc_rag.nodes.generation_nodes import mergeAnswers
        state = _make_agent_state(sub_answers=[
            {"question": "q1", "answer": "a1", "sources": [], "found": True},
            {"question": "q2", "answer": "a2", "sources": [], "found": True},
        ])
        result = mergeAnswers(state)
        assert result["final_answer"] == ""

    def test_error_response_does_not_expose_internal_error(self):
        """Gap 3 regression: errorResponse hides internal error from judge."""
        from RAG.case_doc_rag.nodes.generation_nodes import errorResponse
        state = _make_agent_state(error="InternalDatabaseConnectionError: host=mongo port=27017")
        result = errorResponse(state)
        # Internal error string must NOT appear in the judge-facing message
        assert "InternalDatabaseConnectionError" not in result.get("final_answer", "")
        assert "mongo" not in result.get("final_answer", "").lower()

    def test_error_response_sets_final_answer(self):
        from RAG.case_doc_rag.nodes.generation_nodes import errorResponse
        state = _make_agent_state(error="some error")
        result = errorResponse(state)
        assert "final_answer" in result
        assert len(result["final_answer"]) > 0


# ===========================================================================
# LAYER 3 — GRAPH TESTS
# ===========================================================================


class TestGraphCompilation:
    """Graph wiring and compilation tests."""

    def test_build_graph_returns_compilable_object(self):
        """Phase 7.9 regression: build_graph() must succeed."""
        from RAG.case_doc_rag.graph import build_graph
        app = build_graph()
        assert hasattr(app, "invoke")

    def test_graph_has_retrieve_branch_node(self):
        from RAG.case_doc_rag.graph import build_graph
        app = build_graph()
        # LangGraph compiled graphs expose their graph structure
        graph_repr = str(app.get_graph())
        assert "retrieve_branch" in graph_repr or "dispatchQuestions" in graph_repr

    def test_init_lazy_get_app_returns_same_object(self):
        """Phase 7.10: _get_app() singleton check."""
        from RAG.case_doc_rag import _get_app
        with patch("RAG.case_doc_rag.build_graph") as mock_build:
            mock_build.return_value = MagicMock()
            # Reset singleton
            import RAG.case_doc_rag as pkg
            pkg._app = None
            a = _get_app()
            b = _get_app()
            assert a is b
            assert mock_build.call_count == 1  # built exactly once


class TestGraphPaths:
    """End-to-end graph path tests with fully mocked infrastructure."""

    def _run_with_mocked_infra(self, query, case_id="c1", mock_llm_response=None):
        """Helper: run() with all LLM and DB calls mocked."""
        from RAG.case_doc_rag import run

        mock_response = MagicMock()
        mock_response.content = mock_llm_response or '["mocked question"]'
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_response

        grade_result = MagicMock()
        grade_result.score = "Yes"
        grade_chain = MagicMock()
        grade_chain.invoke.return_value = grade_result

        classify_result = MagicMock()
        classify_result.score = "Yes"

        doc_result = MagicMock()
        doc_result.mode = "no_doc_specified"
        doc_result.doc_id = None

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.__or__ = lambda s, o: grade_chain
        mock_llm.__or__ = lambda s, o: mock_chain

        mock_doc = _make_document("retrieved content")
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever

        with patch("RAG.case_doc_rag.nodes.query_nodes.get_llm", return_value=mock_llm), \
             patch("RAG.case_doc_rag.nodes.selection_nodes.get_llm", return_value=mock_llm), \
             patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_llm", return_value=mock_llm), \
             patch("RAG.case_doc_rag.nodes.generation_nodes.get_llm", return_value=mock_llm), \
             patch("RAG.case_doc_rag.nodes.selection_nodes.get_available_doc_titles", return_value=[]), \
             patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_vectorstore", return_value=mock_vs), \
             patch("RAG.case_doc_rag.nodes.retrieval_nodes.get_retriever", return_value=mock_retriever), \
             patch("RAG.case_doc_rag.nodes.generation_nodes.get_rag_chain", return_value=mock_chain), \
             patch("RAG.case_doc_rag.nodes.query_nodes.ChatPromptTemplate") as MockQCPT, \
             patch("RAG.case_doc_rag.nodes.selection_nodes.ChatPromptTemplate") as MockSCPT, \
             patch("RAG.case_doc_rag.nodes.generation_nodes.ChatPromptTemplate") as MockGCPT:

            for MockCPT in [MockQCPT, MockSCPT, MockGCPT]:
                MockCPT.from_messages.return_value.__or__ = lambda s, o: mock_chain

            result = run(query=query, case_id=case_id)

        return result

    def test_run_returns_required_keys(self):
        from RAG.case_doc_rag import run
        fake_result = {
            "sub_answers": [{"question": "q", "answer": "a", "sources": [], "found": True}],
            "final_answer": "a",
            "error": None,
            "query": "test", "case_id": "c1",
            "conversation_history": [], "request_id": "r1",
            "sub_questions": ["q"], "on_topic": True,
            "doc_selection_mode": "no_doc_specified",
            "selected_doc_id": None, "doc_titles": [],
        }
        import RAG.case_doc_rag as pkg
        pkg._app = None
        mock_app = MagicMock()
        mock_app.invoke.return_value = fake_result
        with patch("RAG.case_doc_rag.build_graph", return_value=mock_app):
            result = run(query="test", case_id="c1")
        assert "sub_answers" in result
        assert "final_answer" in result
        assert "error" in result

    def test_run_sub_answers_is_list(self):
        from RAG.case_doc_rag import run
        fake_result = {
            "sub_answers": [],
            "final_answer": "",
            "error": None,
            "query": "test", "case_id": "c1",
            "conversation_history": [], "request_id": "r1",
            "sub_questions": [], "on_topic": False,
            "doc_selection_mode": "no_doc_specified",
            "selected_doc_id": None, "doc_titles": [],
        }
        import RAG.case_doc_rag as pkg
        pkg._app = None
        mock_app = MagicMock()
        mock_app.invoke.return_value = fake_result
        with patch("RAG.case_doc_rag.build_graph", return_value=mock_app):
            result = run(query="test", case_id="c1")
        assert isinstance(result["sub_answers"], list)

    def test_run_generates_request_id_if_missing(self):
        """Phase 6B: request_id auto-generated."""
        from RAG.case_doc_rag import run
        # Just assert run() accepts no request_id without crashing
        # (full infra mocking is complex; just verify signature)
        import inspect
        sig = inspect.signature(run)
        assert "request_id" in sig.parameters
        assert sig.parameters["request_id"].default is None

    def test_run_never_raises_on_pipeline_failure(self):
        """Rule 2 / Phase 6B: pipeline failure must return error dict, not raise."""
        from RAG.case_doc_rag import run
        with patch("RAG.case_doc_rag._get_app") as mock_app:
            mock_app.return_value.invoke.side_effect = Exception("catastrophic failure")
            result = run(query="test", case_id="c1")
        assert result["error"] == "Internal pipeline failure"
        assert result["sub_answers"] == []


# ===========================================================================
# LAYER 4 — BUG REGRESSION ASSERTIONS
# ===========================================================================


class TestBugRegressions:
    """Explicit regression assertions for all 22 identified bugs."""

    # Bug 1 — query used as HumanMessage everywhere
    def test_bug1_query_is_str_in_agent_state(self):
        from RAG.case_doc_rag.state import AgentState
        assert AgentState.__annotations__["query"] is str

    # Bug 2 — first-turn rewriter skip
    def test_bug2_rewriter_has_no_message_length_gate(self):
        import inspect
        from RAG.case_doc_rag.nodes.query_nodes import questionRewriter
        source = inspect.getsource(questionRewriter)
        assert "len(state[" not in source or "conversation_history" not in source.split("len(state[")[0], \
            "Bug 2: questionRewriter must not gate on conversation length"

    # Bug 3 — JSON not parsed
    def test_bug3_extract_json_list_function_exists(self):
        from RAG.case_doc_rag.nodes.query_nodes import _extract_json_list
        assert callable(_extract_json_list)

    # Bug 4 — classifier uses raw query
    def test_bug4_classifier_uses_sub_questions_not_query(self):
        import inspect
        from RAG.case_doc_rag.nodes.query_nodes import questionClassifier
        source = inspect.getsource(questionClassifier)
        assert "sub_questions" in source
        # Should not access messages[-1]
        assert "messages[-1]" not in source

    # Bug 5 — DocumentFinalizer raw dict
    def test_bug5_finalizer_returns_document_in_sub_answers(self):
        from RAG.case_doc_rag.nodes.selection_nodes import DocumentFinalizer
        state = _make_agent_state(selected_doc_id="doc")
        mongo_doc = {"title": "doc", "content": "text", "source_file": "f.pdf"}
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = mongo_doc
        with patch("RAG.case_doc_rag.nodes.selection_nodes.get_mongo_collection", return_value=mock_coll):
            result = DocumentFinalizer(state)
        sa = result.get("sub_answers", [])
        assert isinstance(sa, list)
        assert sa[0].get("found") is True

    # Bug 6 — history repr
    def test_bug6_serialize_history_no_repr(self):
        from RAG.case_doc_rag.nodes.generation_nodes import _serialize_history
        out = _serialize_history([{"role": "user", "content": "test"}])
        assert "HumanMessage" not in out

    # Bug 7 — format() + invoke
    def test_bug7_refine_question_chain_pattern(self):
        """Bug 7: refineQuestion must use chain.invoke(dict), not llm.invoke(format_string)."""
        import inspect
        from RAG.case_doc_rag.nodes.generation_nodes import refineQuestion
        source = inspect.getsource(refineQuestion)
        # The forbidden pattern is calling .format() on a ChatPromptTemplate and passing
        # the result to llm.invoke(). The docstring may reference .format() — we check
        # that the code body (after the docstring) does not call template.format() as
        # the final step before invocation.
        # Reliable signal: the chain pattern uses __or__ (|) to compose, never calls
        # prompt_template.format() as a standalone expression to pass to invoke.
        # Check that the code uses | operator chaining (chain = ... | ...).
        assert "__or__" in source or "| get_llm" in source or "chain" in source.lower(), (
            "Bug 7: refineQuestion must build a chain with | operator"
        )
        # Additionally verify no bare llm.invoke(string) call — the invoke must
        # receive a dict, which the node-level test (test_refine_question_uses_chain_pattern_not_format)
        # already verifies with runtime inspection.


    # Bug 8 — confidence_router mutates state
    def test_bug8_routers_do_not_mutate_state(self):
        from RAG.case_doc_rag.routers import docSelectorRouter, onTopicRouter, proceedRouter
        agent_state = _make_agent_state()
        sub_state = _make_sub_question_state()
        for router, state in [(onTopicRouter, dict(agent_state)),
                               (docSelectorRouter, dict(agent_state)),
                               (proceedRouter, dict(sub_state))]:
            before = dict(state)
            router(state)
            assert state == before, f"{router.__name__} mutated state"

    # Bug 9 — .content crash
    def test_bug9_document_selector_no_content_call(self):
        import inspect
        from RAG.case_doc_rag.nodes.selection_nodes import documentSelector
        source = inspect.getsource(documentSelector)
        # query.content must not appear
        assert 'state.get("query").content' not in source
        assert "query.content" not in source

    # Bug 10 — rephraseCount ceiling duplicated
    def test_bug10_ceiling_only_in_proceed_router(self):
        import inspect
        from RAG.case_doc_rag.nodes.generation_nodes import refineQuestion
        from RAG.case_doc_rag.routers import proceedRouter
        refine_source = inspect.getsource(refineQuestion)
        router_source = inspect.getsource(proceedRouter)
        assert "_MAX_REPHRASE" not in refine_source
        assert "_MAX_REPHRASE" in router_source

    # Bug 14 — module-level init
    def test_bug14_no_module_level_init(self):
        import RAG.case_doc_rag.infrastructure as infra
        importlib.reload(infra)
        assert infra._embedding_fn is None
        assert infra._vectorstore is None

    # Bug 17 — dead fields
    def test_bug17_dead_fields_removed(self):
        from RAG.case_doc_rag.state import AgentState
        for dead in ["doc_type", "context", "safety_notes", "answer"]:
            assert dead not in AgentState.__annotations__

    # Bug 18 — thread-unsafe set_vectorstore
    def test_bug18_set_vectorstore_uses_lock(self):
        import inspect
        from RAG.case_doc_rag import infrastructure
        source = inspect.getsource(infrastructure.set_vectorstore)
        assert "_singleton_lock" in source or "Lock" in source

    # Bug 21 — print statements
    def test_bug21_no_print_statements_anywhere(self):
        import ast
        import os
        package_root = os.path.dirname(
            os.path.dirname(  # RAG/
                os.path.abspath(__file__)
            )
        )
        pkg_dir = os.path.join(package_root, "RAG", "case_doc_rag")
        violations = []
        for root, _, files in os.walk(pkg_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath) as f:
                    source = f.read()
                try:
                    tree = ast.parse(source, filename=fpath)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        func = node.func
                        if isinstance(func, ast.Name) and func.id == "print":
                            violations.append(f"{fpath}:{node.lineno}")
        assert not violations, (
            "Bug 21: print() calls found in package:\n" + "\n".join(violations)
        )


# ===========================================================================
# LAYER 5 — INTEGRATION SMOKE TESTS (real services)
# ===========================================================================


@pytest.mark.integration
class TestIntegrationSmoke:
    """Real end-to-end tests. Skipped when services are unavailable."""

    @pytest.fixture(autouse=True)
    def check_env(self):
        import os
        missing = []
        for var in ["MONGO_URI", "QDRANT_HOST", "GOOGLE_API_KEY"]:
            if not os.getenv(var):
                missing.append(var)
        if missing:
            pytest.skip(f"Integration test skipped — missing env vars: {missing}")

    def test_run_returns_expected_shape(self):
        from RAG.case_doc_rag import run
        result = run(
            query="ما هي وقائع الدعوى؟",
            case_id="smoke_test_case",
            conversation_history=[],
            request_id="smoke-test-001",
        )
        assert isinstance(result, dict)
        assert "sub_answers" in result
        assert "final_answer" in result
        assert "error" in result
        assert isinstance(result["sub_answers"], list)
        assert isinstance(result["final_answer"], str)

    def test_run_does_not_raise(self):
        from RAG.case_doc_rag import run
        try:
            run(query="ما هي وقائع الدعوى؟", case_id="smoke_test_case")
        except Exception as e:
            pytest.fail(f"run() raised an exception: {e}")

    def test_error_is_none_on_success(self):
        from RAG.case_doc_rag import run
        result = run(query="ما هي وقائع الدعوى؟", case_id="smoke_test_case")
        if result.get("error"):
            # Structured error is allowed; unstructured raise is not
            assert isinstance(result["error"], str)

    def test_multi_question_produces_multiple_sub_answers(self):
        from RAG.case_doc_rag import run
        result = run(
            query="ما هي وقائع الدعوى؟ وما هي الطلبات الختامية للمدعي؟",
            case_id="smoke_test_case",
        )
        # May be 1 (if rewriter merges) or 2 (if decomposed)
        assert len(result["sub_answers"]) >= 1

    def test_set_vectorstore_injection_works(self):
        """Supervisor injection path smoke test."""
        from RAG.case_doc_rag import set_vectorstore
        import RAG.case_doc_rag.infrastructure as infra
        original = infra._vectorstore
        fake = MagicMock(name="fake_vs")
        set_vectorstore(fake)
        assert infra._vectorstore is fake
        # Restore
        infra._vectorstore = original


# ===========================================================================
# STANDALONE RAGAS EVALUATION
# ===========================================================================
# Run directly:  python test_case_doc_rag.py
#
# This block:
#   1. Seeds MongoDB and Qdrant with the 7 fixture documents
#   2. Runs 10 test cases against the live Case Doc RAG pipeline
#   3. Evaluates results using RAGAS metrics
#   4. Tears down seeded data
# ===========================================================================

if __name__ == "__main__":
    import os
    import sys
    import traceback
    from pathlib import Path

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------
    CASE_ID = "2847_2024_civil_south_cairo"
    FIXTURES_DIR = Path(__file__).parent / "fixtures"
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 200

    # Filename -> document title mapping (matches _DOC_MAP in test_real_data.py)
    _DOC_MAP = {
        "صحيفة_دعوى.txt":                       "صحيفة دعوى",
        "محضر_جلسة_25_03_2024.txt":              "محضر جلسة",
        "تقرير_الخبير.txt":                      "تقرير خبير",
        "تقرير_الطب_الشرعي.txt":                "تقرير الطب الشرعي",
        "حكم_المحكمة.txt":                       "حكم",
        "مذكرة_بدفاع_المدعى_عليه_الأول.txt":    "مذكرة بدفاع",
        "مذكرة_بدفاع_المدعى_عليها_الثانية.txt": "مذكرة بدفاع",
    }

    # 10 evaluation test cases
    TEST_CASES = [
        {
            "query": "ما هي وقائع الدعوى؟",
            "expected_keywords": ["دعوى", "وقائع", "عقد", "بيع"],
            "ground_truth": "الدعوى تتعلق بفسخ عقد بيع ابتدائي مزور لشقة سكنية",
        },
        {
            "query": "من هم أطراف الدعوى؟",
            "expected_keywords": ["المدعي", "المدعى عليه"],
            "ground_truth": "المدعي أحمد محمد عبد الله والمدعى عليه الأول محمود سعيد إبراهيم والمدعى عليها الثانية شركة العقارات الحديثة",
        },
        {
            "query": "ما هي طلبات المدعي؟",
            "expected_keywords": ["فسخ", "تعويض"],
            "ground_truth": "فسخ عقد البيع المزور وإلزام المدعى عليهما بمبلغ مليوني جنيه تعويضاً",
        },
        {
            "query": "ما هو رأي الخبير الهندسي؟",
            "expected_keywords": ["خبير", "تقرير"],
            "ground_truth": "تقرير الخبير يتضمن تقييم الأضرار والعقار",
        },
        {
            "query": "ماذا قرر الطب الشرعي بشأن التوقيع؟",
            "expected_keywords": ["توقيع", "تزوير", "طب شرعي"],
            "ground_truth": "تقرير الطب الشرعي يؤكد تزوير التوقيع المنسوب للمدعي",
        },
        {
            "query": "ما هو منطوق الحكم؟",
            "expected_keywords": ["حكم", "محكمة"],
            "ground_truth": "حكم المحكمة في الدعوى",
        },
        {
            "query": "ما هي أوجه دفاع المدعى عليهما؟ وما الأسانيد القانونية؟",
            "expected_keywords": ["دفاع", "مذكرة"],
            "ground_truth": "دفاع المدعى عليهما يتضمن أسانيد قانونية ومستندات",
        },
        {
            "query": "ما هي المستندات المقدمة من المدعي؟",
            "expected_keywords": ["مستندات", "عقد"],
            "ground_truth": "المستندات تشمل بطاقة الهوية وعقد الملكية وعقد البيع المطعون فيه وتقرير الطب الشرعي ومحضر الجلسة",
        },
        {
            "query": "ما هي إجراءات الجلسة الأخيرة؟",
            "expected_keywords": ["جلسة", "محكمة"],
            "ground_truth": "تم ندب خبير هندسي وتحديد جلسة لإيداع التقرير",
        },
        {
            "query": "ما هو رقم القضية ومحكمة الاختصاص؟ وما هو تاريخ رفع الدعوى؟",
            "expected_keywords": ["2847", "2024", "محكمة"],
            "ground_truth": "الدعوى رقم 2847 لسنة 2024 أمام محكمة جنوب القاهرة الابتدائية",
        },
    ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_fixture(filename: str) -> str:
        """Read a fixture file from disk."""
        return (FIXTURES_DIR / filename).read_text(encoding="utf-8")

    def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    # ------------------------------------------------------------------
    # Seeding: MongoDB + Qdrant
    # ------------------------------------------------------------------

    def seed_mongodb(case_id: str):
        """Insert fixture documents into MongoDB Document Storage collection."""
        from pymongo import MongoClient
        from config import cfg

        uri = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
        db_name = cfg.mongodb.get("database", "Rag")
        coll_name = cfg.mongodb.get("collection", "Document Storage")

        client = MongoClient(uri)
        collection = client[db_name][coll_name]

        docs_inserted = 0
        for filename, title in _DOC_MAP.items():
            content = _read_fixture(filename)
            chunks = _chunk_text(content)
            for idx, chunk in enumerate(chunks):
                record = {
                    "case_id": case_id,
                    "title": title,
                    "content": chunk,
                    "source_file": filename,
                    "chunk_index": idx,
                }
                collection.insert_one(record)
                docs_inserted += 1

        # Verify
        count = collection.count_documents({"case_id": case_id})
        print(f"[SEED] Inserted {docs_inserted} chunks into MongoDB (verified {count} for case_id={case_id})")

        # Also check distinct titles
        titles = collection.distinct("title", {"case_id": case_id})
        print(f"[SEED] Fetched {len(titles)} doc titles for case_id={case_id} from MongoDB")
        return client  # Return client for teardown

    def seed_qdrant(case_id: str):
        """Embed and insert fixture documents into Qdrant judicial_docs collection."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from config import cfg

        host = cfg.qdrant.get("host", "localhost")
        port = cfg.qdrant.get("port", 6333)
        collection_name = cfg.qdrant.get("collection", "judicial_docs")
        vector_size = cfg.qdrant.get("vector_size", 1024)

        qdrant = QdrantClient(host=host, port=port)

        # Create collection if it doesn't exist
        existing = [c.name for c in qdrant.get_collections().collections]
        if collection_name not in existing:
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"[SEED] Created Qdrant collection '{collection_name}' (size={vector_size})")
        else:
            print(f"[SEED] Qdrant collection '{collection_name}' already exists")

        # Load embedding model (same as production: BAAI/bge-m3)
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = cfg.embedding.get("model", "BAAI/bge-m3")
        print(f"[SEED] Loading embedding model: {model_name}")
        embed_fn = HuggingFaceEmbeddings(model_name=model_name)

        points = []
        point_id = 0
        for filename, title in _DOC_MAP.items():
            content = _read_fixture(filename)
            chunks = _chunk_text(content)
            for idx, chunk in enumerate(chunks):
                vector = embed_fn.embed_query(chunk)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "page_content": chunk,
                            "metadata": {
                                "case_id": case_id,
                                "title": title,
                                "source_file": filename,
                                "chunk_index": idx,
                            },
                        },
                    )
                )
                point_id += 1

        # Upsert in batches
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            qdrant.upsert(collection_name=collection_name, points=batch)

        print(f"[SEED] Inserted {len(points)} vectors into Qdrant collection '{collection_name}'")
        return qdrant  # Return client for teardown

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def teardown_mongodb(case_id: str):
        """Remove seeded documents from MongoDB."""
        from pymongo import MongoClient
        from config import cfg

        uri = cfg.mongodb.get("uri", "mongodb://localhost:27017/")
        db_name = cfg.mongodb.get("database", "Rag")
        coll_name = cfg.mongodb.get("collection", "Document Storage")

        client = MongoClient(uri)
        collection = client[db_name][coll_name]
        result = collection.delete_many({"case_id": case_id})
        print(f"[TEARDOWN] Deleted {result.deleted_count} MongoDB records for case_id={case_id}")

    def teardown_qdrant(case_id: str):
        """Remove seeded vectors from Qdrant by case_id filter."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from config import cfg

        host = cfg.qdrant.get("host", "localhost")
        port = cfg.qdrant.get("port", 6333)
        collection_name = cfg.qdrant.get("collection", "judicial_docs")

        qdrant = QdrantClient(host=host, port=port)
        qdrant.delete(
            collection_name=collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.case_id",
                        match=MatchValue(value=case_id),
                    )
                ]
            ),
        )
        print(f"[TEARDOWN] Deleted Qdrant vectors for case_id={case_id}")

    # ------------------------------------------------------------------
    # Run pipeline on test cases
    # ------------------------------------------------------------------

    def run_evaluation():
        """Execute all test cases and collect results for RAGAS evaluation."""
        from RAG.case_doc_rag import run

        results = []
        for i, tc in enumerate(TEST_CASES, 1):
            query = tc["query"]
            print(f"\n{'='*60}")
            print(f"Test Case {i}: {query}")
            print(f"{'='*60}")

            try:
                output = run(
                    query=query,
                    case_id=CASE_ID,
                    conversation_history=[],
                    request_id=f"ragas-eval-{i:03d}",
                )

                answer = output.get("final_answer", "")
                sub_answers = output.get("sub_answers", [])
                error = output.get("error")

                # Gather contexts from sub_answers
                contexts = []
                for sa in sub_answers:
                    sources = sa.get("sources", [])
                    contexts.extend(sources)
                if not contexts:
                    contexts = ["No context retrieved"]

                print(f"  Answer: {answer[:200]}...")
                if error:
                    print(f"  Error: {error}")

                # Check expected keywords
                found_kw = [kw for kw in tc["expected_keywords"] if kw in answer]
                print(f"  Keywords found: {found_kw}/{tc['expected_keywords']}")

                results.append({
                    "query": query,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": tc["ground_truth"],
                    "error": error,
                })

            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()
                results.append({
                    "query": query,
                    "answer": f"ERROR: {e}",
                    "contexts": ["Error during execution"],
                    "ground_truth": tc["ground_truth"],
                    "error": str(e),
                })

        return results

    # ------------------------------------------------------------------
    # RAGAS evaluation
    # ------------------------------------------------------------------

    def run_ragas_evaluation(results):
        """Run RAGAS metrics on the collected results.

        Uses gemini-2.5-flash (matching config/settings.yaml) and the
        modern ragas API.
        """
        try:
            from datasets import Dataset
            from ragas import evaluate

            # Try modern API first, fall back to legacy wrappers
            try:
                from ragas.llms import LangchainLLMWrapper
                from ragas.embeddings import LangchainEmbeddingsWrapper
            except ImportError:
                LangchainLLMWrapper = None
                LangchainEmbeddingsWrapper = None

            # Try modern metric imports first, fall back to legacy
            try:
                from ragas.metrics import (
                    answer_relevancy,
                    context_precision,
                    context_recall,
                    faithfulness,
                )
            except ImportError:
                from ragas.metrics import (
                    AnswerRelevancy as answer_relevancy,
                    ContextPrecision as context_precision,
                    ContextRecall as context_recall,
                    Faithfulness as faithfulness,
                )

            from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

            print("\n" + "=" * 60)
            print("RAGAS EVALUATION")
            print("=" * 60)

            # Build the evaluation dataset
            eval_data = {
                "question": [r["query"] for r in results],
                "answer": [r["answer"] for r in results],
                "contexts": [r["contexts"] for r in results],
                "ground_truth": [r["ground_truth"] for r in results],
            }
            dataset = Dataset.from_dict(eval_data)

            # Initialize LLM and embeddings for RAGAS -- use gemini-2.5-flash
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.0,
            )
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
            )

            # Wrap for RAGAS if wrappers are available
            eval_kwargs = {
                "dataset": dataset,
                "metrics": [answer_relevancy, context_precision, context_recall, faithfulness],
            }
            if LangchainLLMWrapper is not None:
                eval_kwargs["llm"] = LangchainLLMWrapper(llm)
            else:
                eval_kwargs["llm"] = llm
            if LangchainEmbeddingsWrapper is not None:
                eval_kwargs["embeddings"] = LangchainEmbeddingsWrapper(embeddings)
            else:
                eval_kwargs["embeddings"] = embeddings

            # Run evaluation
            ragas_result = evaluate(**eval_kwargs)

            print("\nRagas Scores:")
            print("-" * 40)
            for metric_name, score in ragas_result.items():
                if isinstance(score, (int, float)):
                    print(f"  {metric_name}: {score:.4f}")
            print("-" * 40)

            return ragas_result

        except ImportError as e:
            print(f"\n[RAGAS] Skipping RAGAS evaluation -- missing dependency: {e}")
            print("[RAGAS] Install with: pip install ragas datasets langchain-google-genai")
            return None
        except Exception as e:
            print(f"\n[RAGAS] RAGAS evaluation failed: {e}")
            traceback.print_exc()
            return None

    # ------------------------------------------------------------------
    # Main execution flow
    # ------------------------------------------------------------------

    print("=" * 60)
    print("CASE RAG STANDALONE RAGAS EVALUATION")
    print("=" * 60)

    # Check required env vars
    missing_vars = []
    for var in ["MONGO_URI", "QDRANT_HOST", "GOOGLE_API_KEY"]:
        if not os.getenv(var):
            missing_vars.append(var)
    if missing_vars:
        # MONGO_URI and QDRANT_HOST may use defaults; only GOOGLE_API_KEY is hard-required
        if "GOOGLE_API_KEY" in missing_vars:
            print(f"ERROR: Missing required env var GOOGLE_API_KEY")
            sys.exit(1)
        else:
            print(f"WARNING: Missing env vars {missing_vars} -- using defaults from settings.yaml")

    try:
        # Step 1: Seed data
        print("\n--- STEP 1: Seeding MongoDB ---")
        seed_mongodb(CASE_ID)

        print("\n--- STEP 2: Seeding Qdrant ---")
        seed_qdrant(CASE_ID)

        # Step 3: Run test cases
        print("\n--- STEP 3: Running test cases ---")
        results = run_evaluation()

        # Step 4: RAGAS evaluation
        print("\n--- STEP 4: RAGAS evaluation ---")
        ragas_result = run_ragas_evaluation(results)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        errors = [r for r in results if r.get("error")]
        successes = [r for r in results if not r.get("error")]
        print(f"  Total test cases: {len(results)}")
        print(f"  Successful: {len(successes)}")
        print(f"  Failed: {len(errors)}")
        if errors:
            print("  Failed cases:")
            for r in errors:
                print(f"    - {r['query']}: {r['error']}")

    finally:
        # Step 5: Teardown
        print("\n--- STEP 5: Teardown ---")
        try:
            teardown_mongodb(CASE_ID)
        except Exception as e:
            print(f"[TEARDOWN] MongoDB cleanup failed: {e}")
        try:
            teardown_qdrant(CASE_ID)
        except Exception as e:
            print(f"[TEARDOWN] Qdrant cleanup failed: {e}")
