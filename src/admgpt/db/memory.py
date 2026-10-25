"""
Database functions for long-term memory.

Location: src/admgpt/db/memory.py

These are used by the agent's "remember_this" and "recall_memory" tools.
"""

from admgpt.db.models import LongTermMemory, utc_now
from admgpt.db.session import get_db

# Maximum number of memories returned at once, so we do not send
# too much text back to the model
MAX_MEMORIES = 20


def save_memory(thread_id: str, memory: str):
    """
    Save one long-term memory (used by the "remember_this" tool).

    Args:
        thread_id: The chat the memory came from.
        memory: The text to remember.

    Returns:
        str: A confirmation message that the tool can give back to the model.
    """

    with get_db() as db:
        item = LongTermMemory(thread_id=thread_id, memory=memory, created_at=utc_now())

        db.add(item)

    # Reaching this line means the `with` block finished without errors,
    # so the memory has been saved.
    return "Memory saved successfully."


def search_memory(thread_id: str, query: str):
    """
    Find saved memories of a chat (used by the "recall_memory" tool).

    How it works:
        1. If `query` has text, return memories that CONTAIN that text
           (simple keyword match, newest first).
        2. If `query` is empty, or nothing matched, fall back to the most
           recent memories. This fallback helps with vague questions like
           "what do you know about me?", which would never match by keyword.

    Args:
        thread_id: Only memories saved in this chat are returned.
        query: The keyword to look for (may be empty).

    Returns:
        str: A bullet list of memories, one per line, or a message
            saying that nothing was found.
    """

    with get_db() as db:
        base_query = db.query(LongTermMemory).filter(
            LongTermMemory.thread_id == thread_id
        )

        memories = []

        # Step 1: keyword search (only if the query is not empty)
        if query and query.strip():
            memories = (
                base_query
                # autoescape=True makes characters like % and _ be treated
                # as normal text instead of SQL wildcards
                .filter(LongTermMemory.memory.contains(query.strip(), autoescape=True))
                .order_by(LongTermMemory.created_at.desc())  # newest first
                .limit(MAX_MEMORIES)
                .all()
            )

        # Step 2: fallback to the most recent memories
        if not memories:
            memories = (
                base_query.order_by(LongTermMemory.created_at.desc())
                .limit(MAX_MEMORIES)
                .all()
            )

        if not memories:
            return "No saved memory found."

        # Turn the list of rows into text like:
        # - I like Python
        # - My name is Ming
        return "\n".join([f"- {m.memory}" for m in memories])
