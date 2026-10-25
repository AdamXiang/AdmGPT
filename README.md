# AdmGPT

AdmGPT is a self-hosted, ChatGPT-style web assistant powered by Claude. It is built on LangGraph, so the model can decide to call tools: search your uploaded documents (RAG), search the web, save and recall long-term memories, and do math. Answers stream to the browser token by token.

## Features

- **Streaming chat**: answers arrive over Server-Sent Events as the model writes them.
- **Tool-calling agent**: a LangGraph loop (`chatbot → tools → chatbot`) lets Claude choose when to use a tool.
- **Document Q&A (RAG)**: upload PDF, DOCX, TXT, MD, PY, or CSV files. They are split into chunks, embedded with Gemini, and stored in Chroma. Each chat searches only its own files.
- **Smart re-upload**: a file with the same name and content is skipped. A changed file replaces its old chunks.
- **Web search**: Tavily search for news, prices, releases, and other current information.
- **Long-term memory**: the agent can save facts you ask it to remember and recall them later.
- **Calculator**: evaluates math expressions with a restricted `eval`.
- **Persistent chats**: the sidebar lists past conversations. Messages are stored in SQLite, and LangGraph checkpoints let a chat continue where it stopped.
- **Model picker**: choose from an allow-list of Claude models. Unknown names fall back to the default model.

## Architecture

```
Browser (templates/index.html)
   │  HTTP + SSE
   ▼
api/app.py  (FastAPI)
   ├──▶ agent/   LangGraph agent + Claude (ChatAnthropic), SQLite checkpoints
   │       └──▶ tools/   calculator · document search · memory · Tavily web search
   ├──▶ db/      SQLAlchemy: conversations, messages, long-term memory
   └──▶ rag/     Chroma vector store + Gemini embeddings
```

| Path | Purpose |
| --- | --- |
| `src/admgpt/config.py` | Loads `.env`, defines data paths and the model allow-list |
| `src/admgpt/api/app.py` | FastAPI routes: page, chat list, history, upload, chat stream |
| `src/admgpt/agent/` | Builds and caches one LangGraph agent per model; system prompt |
| `src/admgpt/tools/tools.py` | Tools the agent can call |
| `src/admgpt/rag/rag.py` | File parsing, chunking, indexing, and retrieval |
| `src/admgpt/db/` | Database engine, table models, and query helpers |
| `src/admgpt/templates/index.html` | Single-page chat UI |

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- API keys:
  - `ANTHROPIC_API_KEY` for the Claude chat model
  - `GOOGLE_API_KEY` for Gemini embeddings (document upload and search)
  - `TAVILY_API_KEY` for web search

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/AdamXiang/AdmGPT.git
   cd AdmGPT
   ```

2. Install the dependencies:

   ```bash
   uv sync
   ```

   Without uv, use `pip install -r requirements.txt` and then `pip install -e .`.

3. Create your environment file and fill in the keys:

   ```bash
   cp .env.template .env
   ```

4. Start the server:

   ```bash
   uv run admgpt
   ```

5. Open <http://localhost:8080> in your browser.

For auto-reload during development, run:

```bash
uv run uvicorn admgpt.api.app:app --reload --port 8080
```

## Configuration

Set these variables in `.env`:

| Variable | Required | Description |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `ANTHROPIC_MODEL` | No | Default model. The default is `claude-haiku-4-5`. |
| `GOOGLE_API_KEY` | Yes, for RAG | Used by `gemini-embedding-001` |
| `TAVILY_API_KEY` | Yes, for web search | Tavily API key |
| `LANGSMITH_TRACING` | No | Set to `true` to send traces to LangSmith |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name |

To change which models the UI offers, edit `ALLOWED_MODELS` in `src/admgpt/config.py`.

## Local data

The app creates these folders in the project root. Git ignores all of them.

| Folder | Contents |
| --- | --- |
| `data/` | `chatbot_memory.db` (chats, messages, memories) and `langgraph_checkpoints.sqlite` (agent state) |
| `chroma_db/` | Chroma vector store for uploaded documents |
| `document/` | Uploaded files, one subfolder per chat |

To reset the app, stop the server and delete these folders.

## API

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Chat web page |
| `GET` | `/conversations` | List chats, newest first |
| `GET` | `/history/{thread_id}` | All messages of one chat |
| `POST` | `/upload` | Form fields `file` and `thread_id`. Indexes a document for RAG. |
| `POST` | `/chat/stream` | JSON `{"message", "thread_id", "model"}`. Returns an SSE stream of `{"token"}`, `{"error"}`, and `{"done"}` events. |

## Security notes

AdmGPT is meant for local, single-user use. It has no authentication. The calculator tool uses a restricted `eval`, which is not a full sandbox. Do not expose the server to untrusted users.

## License

[Apache License 2.0](LICENSE)
