"""
Tools that the AdmGPT agent can call.

Location: src/admgpt/tools/tools.py

What is a "tool"?
    A tool is a normal Python function that the AI model can ASK us to run.
    The model reads each tool's name and docstring to decide when to use it,
    so the docstring of every @tool function below is written for the MODEL,
    not only for humans. Changing a docstring changes how the model behaves!

How a tool call works:
    1. The user asks something (e.g. "what is 12 * 7?").
    2. The model replies "please run calculator('12 * 7')".
    3. ToolNode (in agent.py) runs the function and returns the result.
    4. The model reads the result and writes the final answer.

Tools in this file:
    calculator                -> simple math
    search_uploaded_documents -> RAG search over the user's uploaded files
    remember_this             -> save a long-term memory
    recall_memory             -> read long-term memories
    web_search                -> Tavily web search (for current information)
"""

import math

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

# Imported only for its side effects (loads .env and sets up logging).
# We rename it to `app_config` so it does not clash with the `config`
# parameter that the tools below receive from LangGraph.
from admgpt import config as app_config  # noqa: F401
from admgpt.db import save_memory, search_memory
from admgpt.rag.rag import retrieve_from_rag

# ---------------------------------------------------------------------------
# Getting the current chat ID
# ---------------------------------------------------------------------------

# Some tools (documents, memory) need to know WHICH chat they are working for.
#
# How it works now:
#   The API layer calls the agent like this:
#       agent.invoke(input, config={"configurable": {"thread_id": "abc"}})
#   LangGraph passes that same `config` to every tool that has a parameter
#   named `config` typed as RunnableConfig. So each request carries its OWN
#   thread_id, and two users at the same time can never mix up their chats.
#   (The old version used one shared global variable, which could be
#   overwritten by another request.)


def _get_thread_id(config: RunnableConfig) -> str:
    """
    Read the thread_id out of the config that LangGraph passes to a tool.

    The leading underscore means "internal helper, only used in this file".

    Args:
        config: The RunnableConfig injected by LangGraph.

    Returns:
        str: The thread_id of the chat that triggered this tool call.

    Raises:
        ValueError: If no thread_id was provided. We fail loudly on purpose:
            silently using a fake ID like "default" would mix data from
            different chats together without anyone noticing.
    """

    thread_id = (config or {}).get("configurable", {}).get("thread_id")

    if not thread_id:
        raise ValueError(
            "thread_id is missing. Call the agent with "
            "config={'configurable': {'thread_id': ...}}"
        )

    return thread_id


# ---------------------------------------------------------------------------
# Tool objects that are not written as functions
# ---------------------------------------------------------------------------

# Tavily web search tool, provided ready-made by langchain_tavily.
# It reads TAVILY_API_KEY from the environment (loaded by `app_config` above,
# which is why the config import must come before this line runs).
#   max_results   : return at most 5 results
#   topic         : "general" for normal searches ("news" is also available)
#   search_depth  : "advanced" gives better results but is slower and costs more
web_search = TavilySearch(max_results=5, topic="general", search_depth="advanced")


# ---------------------------------------------------------------------------
# Tools written as functions (the @tool decorator turns them into tools)
# ---------------------------------------------------------------------------


@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculation.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    # NOTE: the docstring above is read by the model. Keep it short and clear.

    try:
        # Whitelist of names the expression is allowed to use.
        # The model can write things like math.sqrt(16) or max(1, 2),
        # but nothing outside this list.
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        # eval() runs a string as Python code, which can be dangerous.
        # We limit the risk by removing all built-in functions
        # ({"__builtins__": {}}) and only giving it the `allowed` names.
        # This is a basic safeguard, NOT a perfect sandbox: do not expose this
        # tool to untrusted users without a safer math parser.
        result = eval(expression, {"__builtins__": {}}, allowed)

        # Tools must return strings (or things that become strings) to the model
        return str(result)

    except Exception as e:  # noqa: BLE001  (on purpose: any error goes back to the model)
        # Return the error as text instead of crashing, so the model can read it
        # and try again or explain the problem to the user.
        # `{e!s}` means str(e).
        return f"Calculation error: {e!s}"


@tool
def search_uploaded_documents(query: str, config: RunnableConfig) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """

    # `config` is filled in by LangGraph automatically. It is NOT shown to the
    # model, so the model still only sees one parameter: `query`.

    # RAG = Retrieval-Augmented Generation: find the document chunks most
    # related to the query. We pass thread_id so we only search the files
    # uploaded in THIS chat.
    return retrieve_from_rag(query=query, thread_id=_get_thread_id(config))


@tool
def remember_this(memory: str, config: RunnableConfig) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    # Saves the text into the long_term_memory table (see db/memory.py)
    return save_memory(thread_id=_get_thread_id(config), memory=memory)


@tool
def recall_memory(query: str, config: RunnableConfig) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    # Looks up saved memories of this chat (see db/memory.py)
    return search_memory(thread_id=_get_thread_id(config), query=query)


# ---------------------------------------------------------------------------
# Tool list
# ---------------------------------------------------------------------------

# This is the list agent.py imports and gives to the model (bind_tools)
# and to ToolNode. A tool that is NOT in this list cannot be used by the agent.
# When you write a new tool, remember to add it here!
tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search,
]
