"""
Core module for the AdmGPT agent.

Location: src/admgpt/agent/agent.py

This file builds an AI agent that can chat and use tools.
Overall flow (a mental picture for beginners):

    User message
        │
        ▼
    [chatbot node]  <- calls the Claude model, which decides to answer directly or use a tool
        │
        ├── tool needed ──▶ [tools node] runs the tool ──▶ result goes back to [chatbot node]
        │
        └── no tool needed ──▶ finish and return the answer

Key technologies:
- LangGraph: describes the agent's flow as nodes and edges.
- ChatAnthropic: LangChain's wrapper around Claude models.
- SqliteSaver: saves conversation state in SQLite so a chat can be resumed later (memory).

Where things live now:
- Settings and model validation -> admgpt/config.py
- System prompt                 -> admgpt/agent/prompts.py
- Tool list                     -> admgpt/tools
"""

import logging
import sqlite3

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from admgpt.agent.prompts import SYSTEM_PROMPT
from admgpt.config import DATA_DIR, NO_TEMPERATURE_MODELS, normalize_model_name

# `tools` is the list of tools defined in src/admgpt/tools/tools.py
from admgpt.tools.tools import tools

# One logger per module; log lines will show "admgpt.agent.agent" as the source
logger = logging.getLogger(__name__)


def build_agent(model_name: str):
    """
    Build a LangGraph agent for the selected Anthropic model.

    The agent's structure (a loop):
        START -> chatbot -> (tool needed?) -> tools -> chatbot -> ... -> END
    The model keeps going back and forth between "thinking" and "using tools"
    until no more tools are needed.

    Notes:
        - Every call builds a new graph and a new database connection, which is
          relatively expensive. In practice, get the agent through get_agent(),
          which caches the result and avoids rebuilding it.

    Args:
        model_name: The model you want to use (validated by normalize_model_name first).

    Returns:
        A compiled LangGraph object that you can run with .invoke() / .stream().
    """

    # Validate the model name first to make sure it is allowed
    selected_model = normalize_model_name(model_name)
    logger.info("Building agent for model: %s", selected_model)

    # Initialize the Claude model
    # temperature: 0 to 1. Lower = more stable answers, higher = more creative.
    #   Some newer models do not accept it (see NO_TEMPERATURE_MODELS in config.py).
    # streaming=True: lets the answer stream token by token
    #   (so the frontend can display it in real time).
    temperature = None if selected_model in NO_TEMPERATURE_MODELS else 0.3
    llm = ChatAnthropic(
        model_name=selected_model, temperature=temperature, streaming=True
    )

    # "Bind" the tools to the model so it knows which tools it can call
    llm_with_tools = llm.bind_tools(tools)

    def chatbot_node(state: MessagesState):
        """
        Chat node: calls the model to produce a response.

        Args:
            state: The current conversation state. state["messages"] is the
                list of previous messages.

        Returns:
            dict: {"messages": [model response]}. LangGraph automatically
                appends it to the conversation history.
        """
        # Put the system prompt before the history so the model knows its role and rules
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

        # Call the model. It may return a text answer, or a request like
        # "please run this tool".
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    # Tool node: actually runs the tools the model asked for and puts the results back in the conversation
    tool_node = ToolNode(tools)

    # Create the state graph. The state format is the built-in MessagesState
    # (just a list of messages).
    workflow = StateGraph(MessagesState)

    # Register the two nodes (name -> function to run)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("tools", tool_node)

    # Define the flow: start by going into the chatbot node
    workflow.add_edge(START, "chatbot")

    # Conditional edge: after chatbot, tools_condition checks whether the model asked for a tool
    #   yes -> go to the "tools" node
    #   no  -> end the flow
    workflow.add_conditional_edges("chatbot", tools_condition)

    # After tools finish, always return to chatbot so the model can
    # continue answering based on the tool results
    workflow.add_edge("tools", "chatbot")

    # Open a SQLite connection used to store conversation state (checkpoints)
    # check_same_thread=False: lets different threads share this connection
    #   (web servers are often multi-threaded; without this you get an error)
    conn = sqlite3.connect(
        f"{DATA_DIR}/langgraph_checkpoints.sqlite", check_same_thread=False
    )

    # SqliteSaver saves the conversation state to the database after each step,
    # so using the same thread_id later lets you resume the earlier conversation
    checkpoint = SqliteSaver(conn)
    logger.info(
        "Checkpoint database connected: %s", DATA_DIR / "langgraph_checkpoints.sqlite"
    )

    # Only after compile() do we get an agent that can actually run
    return workflow.compile(checkpointer=checkpoint)


# Cache dictionary: key = model name, value = an already-built agent.
# The leading underscore _ is a Python convention meaning
# "internal to this module; do not modify it from outside".
_AGENT_CACHE = {}


def get_agent(model_name: str | None = None):
    """
    Get (and cache) the LangGraph agent for the given model.

    The first time a model is requested, the agent is built and stored in the
    cache. After that, the same agent is reused, which avoids rebuilding the
    graph and database connection on every request and saves time and resources.

    Args:
        model_name: The model name. May be None, in which case the default model is used.

    Returns:
        A compiled LangGraph agent.

    Example:
        >>> agent = get_agent("claude-sonnet-5")
        >>> agent2 = get_agent("claude-sonnet-5")
        >>> agent is agent2   # same object, because of the cache
        True
    """

    # Normalize the name first, so None and invalid values both map to DEFAULT_MODEL,
    # and different spellings don't create duplicate agents for the same model
    selected_model = normalize_model_name(model_name)

    # Log when the requested name was invalid/missing and we fell back to the default
    if model_name and model_name.strip() != selected_model:
        logger.warning(
            "Model '%s' is not allowed, falling back to '%s'",
            model_name,
            selected_model,
        )

    # Build only if it is not in the cache yet (this happens just once per model)
    if selected_model not in _AGENT_CACHE:
        logger.info(
            "Agent cache miss for model: %s (creating a new one)", selected_model
        )
        _AGENT_CACHE[selected_model] = build_agent(selected_model)
    else:
        logger.info("Agent cache hit for model: %s", selected_model)

    return _AGENT_CACHE[selected_model]
