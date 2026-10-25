"""
Database package.

Location: src/admgpt/db/__init__.py

This file re-exports the functions other parts of the app need, so you can write
    from admgpt.db import save_chat_message
instead of
    from admgpt.db.messages import save_chat_message
Both styles work.

Layout of this package:
    session.py        engine, session helper (get_db), init_db
    models.py         table definitions
    conversations.py  sidebar chat list functions
    messages.py       chat message functions
    memory.py         long-term memory functions
"""

from admgpt.db.conversations import create_or_update_conversation, list_conversations
from admgpt.db.memory import save_memory, search_memory
from admgpt.db.messages import get_chat_history, save_chat_message
from admgpt.db.session import init_db

# __all__ lists what `from admgpt.db import *` exposes, and tells linters
# that these imports are intentional
__all__ = [
    "create_or_update_conversation",
    "get_chat_history",
    "init_db",
    "list_conversations",
    "save_chat_message",
    "save_memory",
    "search_memory",
]
