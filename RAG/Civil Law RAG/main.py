"""
main.py

Entry point for manual system operations.

Purpose:
---------
Allows controlled execution of:
- Law indexing
- Maintenance tasks
- Future administrative utilities

This file is NOT responsible for:
- Running the LangGraph pipeline
- Handling API requests
- Serving a mobile app

It is strictly an operational entry script.

Design Principle:
-----------------
Explicit execution.
Critical operations like indexing should not occur implicitly.
"""
from indexer import index_civil_law
from graph import app, default_state_template
from nodes import State
import sys
import io

# Force stdout/stderr to use UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def ask_question(query: str, db) -> str:
    """
    Processes a user query through the LangGraph workflow.

    Args:
        query (str): The user's question in Arabic, ideally related to Egyptian Civil Law.
        db: The Qdrant vectorstore instance.

    Returns:
        str: The final answer produced by the system.
    """
    # Initialize a fresh state for this query
    state: State = default_state_template.copy()
    state["last_query"] = query
    state["db"] = db

    # Run the query through the graph
    result_state = app.invoke(state)

    # Return the final answer
    return result_state.get("final_answer", "تعذر الحصول على إجابة.")


if __name__ == "__main__":
    # Ensure vectorstore exists and get its instance
    db = index_civil_law()  # now returns the db

    print("=== Egyptian Civil Law AI Judge Assistant ===\n")
    while True:
        user_input = input("ادخل سؤالك (أو 'خروج' لإنهاء البرنامج): ").strip()
        if user_input.lower() == "خروج":
            print("شكراً لاستخدام النظام. إلى اللقاء!")
            break

        answer = ask_question(user_input, db)  # pass db explicitly
        print("\n💡 الإجابة:\n", answer)
        print("\n" + "-"*50 + "\n")