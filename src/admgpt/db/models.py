"""
Database table definitions.

Location: src/admgpt/db/models.py

Each class below is one table. A column is one field of that table.

This is separate from LangGraph's checkpoint database
(langgraph_checkpoints.sqlite). LangGraph's checkpoint stores the agent's
internal state; these tables store data the app itself shows or searches.
"""

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from admgpt.db.session import Base


def utc_now() -> datetime:
    """
    Return the current time in UTC.

    This replaces datetime.utcnow(), which is deprecated since Python 3.12.
    It is also used as the `default=` of the created_at / updated_at columns,
    so we pass the function itself (without parentheses) and SQLAlchemy calls
    it each time a new row is inserted.
    """

    return datetime.now(UTC)


class Conversation(Base):
    """
    One chat session. Shown as an item in the sidebar chat list.

    Columns:
        id         : auto-increment primary key
        thread_id  : unique ID of the chat; the same ID is used by LangGraph
                     to find the chat's checkpoint
        title      : text shown in the sidebar (first part of the first message)
        created_at : when the chat was created (UTC time)
        updated_at : when the chat last had activity (used for sorting)
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)  # unique: one row per chat
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now)


class ChatMessage(Base):
    """
    One message inside a chat (from the user or from the assistant).

    Columns:
        id         : auto-increment primary key
        thread_id  : which chat this message belongs to
        role       : who said it, e.g. "user" or "assistant"
        content    : the message text
        created_at : when the message was saved (used to keep the order)
    """

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)  # index=True makes lookups by chat faster
    role = Column(String)
    content = Column(Text)  # Text = long text with no length limit
    created_at = Column(DateTime, default=utc_now)


class LongTermMemory(Base):
    """
    A fact the AI was asked to remember (e.g. "I like Python").

    Columns:
        id         : auto-increment primary key
        thread_id  : which chat the memory was saved from
        memory     : the text of the memory
        created_at : when the memory was saved
    """

    __tablename__ = "long_term_memory"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    memory = Column(Text)
    created_at = Column(DateTime, default=utc_now)
