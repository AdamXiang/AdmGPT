"""
FastAPI web server for AdmGPT.

Location: src/admgpt/api/app.py

This file is the "front door" of the app. It receives HTTP requests from the
browser and calls the other layers to do the real work:

    Browser  ──▶  app.py (routes)  ──▶  agent/   (AI model + tools)
                                   ──▶  db/      (chat list, messages, memory)
                                   ──▶  rag/     (uploaded documents)

Routes:
    GET  /                  the chat web page
    GET  /conversations     list of chats for the sidebar
    GET  /history/{id}      all messages of one chat
    POST /upload            upload a document for RAG
    POST /chat/stream       send a message, get the answer streamed back (SSE)
"""

import json
import logging
import re
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

# Importing from config also loads .env and sets up logging, so we do not
# need a separate "import config" line here.
from admgpt.agent.agent import get_agent
from admgpt.config import ALLOWED_MODELS, DEFAULT_MODEL, DOCUMENT_DIR
from admgpt.db import (
    create_or_update_conversation,
    get_chat_history,
    init_db,
    list_conversations,
    save_chat_message,
)
from admgpt.rag.rag import add_document_to_rag

logger = logging.getLogger(__name__)

app = FastAPI()

# The templates folder sits next to the api/ folder (src/admgpt/templates).
# Building the path from __file__ means it works no matter where you start the app.
# (The old "templates" relative path only worked from one specific folder.)
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Create the database tables if they do not exist yet (safe to run every start)
init_db()


# ---------------------------------------------------------------------------
# Simple page and data routes
# ---------------------------------------------------------------------------


@app.get("/")
async def home(request: Request):
    """Return the chat web page (templates/index.html)."""

    # The model dropdown is built from ALLOWED_MODELS, so the page can only
    # offer models the backend actually accepts
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"models": sorted(ALLOWED_MODELS), "default_model": DEFAULT_MODEL},
    )


@app.get("/conversations")
async def conversations():
    """Return all chats, newest activity first, for the sidebar list."""

    items = list_conversations()

    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                # isoformat() turns a datetime into text like "2026-10-25T14:30:00"
                # because JSON cannot contain datetime objects
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
            }
            for item in items
        ]
    }


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    """Return every saved message of one chat, oldest first."""

    messages = get_chat_history(thread_id)

    return {
        "messages": [{"role": msg.role, "content": msg.content} for msg in messages]
    }


# ---------------------------------------------------------------------------
# Document upload (RAG)
# ---------------------------------------------------------------------------

# File types we can read (must match read_file_text in rag/rag.py)
ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]


def build_save_path(thread_id: str, filename: str) -> Path:
    """
    Decide where an uploaded file is saved on disk, in a safe way.

    The result looks like:  <project>/document/<thread_id>/<file name>

    Why this layout?
        - The file name stays the SAME as the original. rag.py uses the file
          name to recognise a re-upload (replace / skip). The old code added a
          random UUID in front of the name, so every upload looked like a brand
          new file and re-upload detection could never work.
        - One folder per chat keeps different chats' files apart, so two chats
          can both upload "notes.txt" without overwriting each other.

    Safety:
        thread_id and filename come from the browser, so we cannot trust them.
        A name like "../../secret" could otherwise write files outside the
        document folder (a "path traversal" attack). So we clean both parts.

    Args:
        thread_id: The chat ID sent by the browser.
        filename: The original file name sent by the browser.

    Returns:
        Path: The full path to save the file at (its folder is created).
    """

    # Folder name: keep only letters, digits, "_" and "-"; replace everything else
    safe_thread = re.sub(r"[^A-Za-z0-9_-]", "_", thread_id) or "default"

    # File name: Path(...).name drops any folder part ("a/b/c.pdf" -> "c.pdf").
    # We keep non-English characters (e.g. Chinese file names) as they are.
    safe_name = Path(filename).name.replace(" ", "_")

    # Names like ".." would point to the parent folder, so reject them
    if safe_name in ("", ".", ".."):
        safe_name = "uploaded_file"

    folder = DOCUMENT_DIR / safe_thread
    folder.mkdir(parents=True, exist_ok=True)

    return folder / safe_name


def build_upload_message(result: dict) -> str:
    """
    Turn the result of add_document_to_rag into a sentence for the user.

    Args:
        result: {"filename": ..., "chunks": ..., "status": "added" | "replaced" | "skipped"}

    Returns:
        str: A message the frontend can show directly.
    """

    name = result["filename"]
    chunks = result["chunks"]

    if result["status"] == "skipped":
        return f"{name} has not changed, so it was skipped ({chunks} chunks already stored)."

    if result["status"] == "replaced":
        return f"Updated {name}: the old version was replaced with {chunks} new chunks."

    return f"Uploaded {name} and created {chunks} chunks."


@app.post("/upload")
async def upload_document(file: UploadFile = File(...), thread_id: str = Form(...)):
    """
    Receive an uploaded file, save it, and index it for RAG.

    Form fields:
        file: The uploaded file.
        thread_id: Which chat the file belongs to.

    Returns:
        JSON {"success": bool, "message": str, "status": ...}
        status is "added", "replaced" or "skipped" (see rag/rag.py).
    """

    try:
        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        # Reject file types we cannot read
        if suffix not in ALLOWED_EXTENSIONS:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.",
                },
                status_code=400,
            )

        # Save the uploaded file to document/<thread_id>/<file name>
        file_path = build_save_path(thread_id, filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, "Uploaded document")

        # add_document_to_rag is slow (it calls the embeddings API) and is not
        # async. Running it directly here would freeze the whole server while it
        # works, including other users' streaming chats. run_in_threadpool runs
        # it in a background thread instead.
        result = await run_in_threadpool(
            add_document_to_rag, file_path=str(file_path), thread_id=thread_id
        )

        logger.info(
            "Upload %s: %s (%s chunks) in chat %s",
            result["status"],
            result["filename"],
            result["chunks"],
            thread_id,
        )

        return JSONResponse(
            {
                "success": True,
                "message": build_upload_message(result),
                "status": result["status"],
            }
        )

    except Exception as e:
        logger.exception("Upload failed")
        return JSONResponse({"success": False, "message": str(e)}, status_code=500)


# ---------------------------------------------------------------------------
# Chat streaming
# ---------------------------------------------------------------------------


def sse_data(payload: dict) -> str:
    """
    Format a dict as one Server-Sent Event (SSE) message.

    SSE is a simple way for the server to push many small messages to the
    browser over one connection. Each message must look like:
        data: <text>\\n\\n
    ensure_ascii=False keeps non-English characters readable instead of \\uXXXX.
    """

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    """
    Decide whether a piece of the agent's output should be shown to the user.

    This prevents raw tool/search/RAG JSON from appearing in the frontend.

    We only stream normal AI text chunks.
    We do NOT stream:
    - ToolMessage
    - messages from tool nodes
    - tool call chunks
    - raw tool outputs

    Args:
        chunk: One message piece produced while the agent runs.
        metadata: Extra info from LangGraph, including which node produced it.

    Returns:
        bool: True if the chunk is normal answer text that should be shown.
    """

    metadata = metadata or {}

    # Which graph node produced this chunk ("chatbot" or "tools")
    node_name = str(metadata.get("langgraph_node", "")).lower()

    if "tool" in node_name:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    # Only AI-written messages are answers
    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    # If the AI is asking to run a tool, that is not text for the user
    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}

    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    """
    Get the plain text out of a message chunk.

    A chunk's `content` can be a string, or a list of parts (text parts, tool
    parts, ...) depending on the model. This function handles both and returns
    just the text.

    Args:
        chunk: A message chunk from the agent.

    Returns:
        str: The text, or "" if the chunk has no text.
    """

    content = getattr(chunk, "content", "")

    if not content:
        return ""

    # Simple case: the content is already a string
    if isinstance(content, str):
        return content

    # List case: collect the text from every part and join them
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])

        return "".join(text_parts)

    return ""


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    Receive a user message and stream the AI's answer back token by token.

    JSON body:
        message  : the user's text (required)
        thread_id: which chat this belongs to (default "default")
        model    : which Claude model to use (optional; unknown or missing
                   values fall back to the default model, see config.py)

    Returns:
        A text/event-stream response. Each event is one of:
            {"token": "..."}   a piece of the answer
            {"error": "..."}   something went wrong
            {"done": true}     the answer is finished
    """

    try:
        data = await request.json()
    except Exception:  # noqa: BLE001  (any parsing problem means a bad request)
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"error": "Invalid JSON body."}, status_code=400)

    user_message = data.get("message", "")
    thread_id = str(data.get("thread_id") or "default")

    # No hard-coded default here any more. If the frontend sends nothing (None),
    # get_agent() picks DEFAULT_MODEL; if it sends a name that is not in
    # ALLOWED_MODELS, get_agent() logs a warning and also uses DEFAULT_MODEL.
    selected_model = data.get("model")
    if not isinstance(selected_model, str):
        selected_model = None

    if not isinstance(user_message, str) or not user_message.strip():
        return JSONResponse({"error": "Message is required."}, status_code=400)

    agent = get_agent(selected_model)

    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", user_message)

    # The thread_id travels INSIDE this config. LangGraph uses it for the chat
    # checkpoint, and passes the same config to every tool (see tools.py), so
    # tools know which chat they work for. This replaces the old global
    # set_current_thread_id(...), which could mix up two users' chats.
    config = {"configurable": {"thread_id": thread_id}}

    def event_generator():
        """Run the agent and yield SSE messages as the answer is produced."""

        final_answer = ""

        # ID of the AI message the last token came from. When the model writes
        # some text, calls a tool, and then writes more text, those are two
        # different messages; without a separator they would be glued together.
        last_message_id = None

        try:
            inputs = {"messages": [HumanMessage(content=user_message)]}

            # stream_mode="messages" gives us the model's output piece by piece
            for chunk, metadata in agent.stream(
                inputs, config=config, stream_mode="messages"
            ):
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)

                if token:
                    if last_message_id and chunk.id != last_message_id:
                        token = "\n\n" + token
                    last_message_id = chunk.id

                    final_answer += token
                    yield sse_data({"token": token})

            # Save the full answer only after it finished streaming
            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)

            yield sse_data({"done": True})

        except Exception as e:
            logger.exception("Chat stream failed (thread_id=%s)", thread_id)
            yield sse_data({"error": str(e)})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",  # never cache a live stream
            "Connection": "keep-alive",  # keep the connection open
            "X-Accel-Buffering": "no",  # tell nginx not to buffer the stream
        },
    )


if __name__ == "__main__":
    # The app now lives inside a package, so uvicorn needs the full import path
    # "admgpt.api.app:app" (the old "app:app" only works when this file is the
    # top-level module). You can also start it from the project root with:
    #     uvicorn admgpt.api.app:app --reload --port 8080
    uvicorn.run("admgpt.api.app:app", host="0.0.0.0", port=8080, reload=True)
