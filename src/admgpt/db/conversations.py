"""
Database functions for conversations (the sidebar chat list).

Location: src/admgpt/db/conversations.py
"""

from admgpt.db.models import Conversation, utc_now
from admgpt.db.session import get_db

# Maximum number of characters of the first message used as the chat title
TITLE_MAX_LENGTH = 40


def create_or_update_conversation(thread_id: str, first_message: str | None = None):
    """
    Create a new conversation, or refresh the "updated_at" time of an existing one.

    Args:
        thread_id: The unique ID of the chat.
        first_message: The user's first message. Only used when the chat is
            created, to build a short title (first 40 characters).

    Behavior:
        - Chat not found -> create it with a title made from first_message
        - Chat found     -> only update updated_at (so it moves to the top of the list)
    """

    # get_db() opens the session and commits / closes it for us
    with get_db() as db:
        # Look for an existing conversation with this thread_id.
        # .first() returns the first match, or None if nothing matches.
        conversation = (
            db.query(Conversation).filter(Conversation.thread_id == thread_id).first()
        )

        if not conversation:
            # New chat: build the title
            title = "New Chat"

            if first_message:
                cleaned = first_message.strip()
                # Use the first 40 characters of the message as the title
                title = cleaned[:TITLE_MAX_LENGTH]
                # Add "..." if the message was longer than 40 characters
                if len(cleaned) > TITLE_MAX_LENGTH:
                    title += "..."

            now = utc_now()
            conversation = Conversation(
                thread_id=thread_id,
                title=title,
                created_at=now,
                updated_at=now,
            )

            # add() only stages the new row; it is saved when the `with` block ends
            db.add(conversation)

        else:
            # Existing chat: just mark it as recently active
            conversation.updated_at = utc_now()


def list_conversations():
    """
    Get all conversations, newest activity first.

    Returns:
        list[Conversation]: Used to fill the sidebar chat list.
    """

    with get_db() as db:
        # desc() = descending, so the most recently updated chat comes first
        return db.query(Conversation).order_by(Conversation.updated_at.desc()).all()
