"""
Database functions for chat messages.

Location: src/admgpt/db/messages.py
"""

from admgpt.db.models import ChatMessage, Conversation, utc_now
from admgpt.db.session import get_db


def save_chat_message(thread_id: str, role: str, content: str):
    """
    Save one message and mark its conversation as recently updated.

    Args:
        thread_id: Which chat the message belongs to.
        role: Who sent it, e.g. "user" or "assistant".
        content: The message text.
    """

    with get_db() as db:
        msg = ChatMessage(
            thread_id=thread_id,
            role=role,
            content=content,
            created_at=utc_now(),
        )

        db.add(msg)

        # Also bump the conversation's updated_at so the sidebar order stays correct
        conversation = (
            db.query(Conversation).filter(Conversation.thread_id == thread_id).first()
        )

        if conversation:
            conversation.updated_at = utc_now()

        # No commit needed here: leaving the `with` block saves both the
        # message and the conversation update together in one go.


def get_chat_history(thread_id: str):
    """
    Get all messages of a chat in the order they were sent.

    Args:
        thread_id: Which chat to load.

    Returns:
        list[ChatMessage]: Oldest message first (asc() = ascending).
    """

    with get_db() as db:
        return (
            db.query(ChatMessage)
            .filter(ChatMessage.thread_id == thread_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
