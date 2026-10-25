"""
RAG (Retrieval-Augmented Generation) module.

Location: src/admgpt/rag/rag.py

What is RAG? (a mental picture for beginners)
    An AI model does not know what is inside the user's own files. RAG fixes this
    in two stages:

    Stage 1 - Indexing (add_document_to_rag):
        file -> extract text -> cut into small chunks -> turn each chunk into
        numbers ("embedding") -> store them in a vector database (Chroma)

    Stage 2 - Retrieval (retrieve_from_rag):
        user question -> turn it into numbers too -> ask Chroma for the chunks
        whose numbers are the most similar -> give those chunks to the model,
        which uses them to write the answer

Key words:
    - chunk       : a small piece of the document text
    - embedding   : a list of numbers that represents the MEANING of a text,
                    so texts with similar meaning have similar numbers
    - vector store: a database that can search by similarity of embeddings
    - metadata    : extra information saved with each chunk (here: thread_id and
                    source file name)
"""

import hashlib
from pathlib import Path

import docx2txt
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# Importing from config also makes sure .env is loaded BEFORE the embeddings
# object below is created, so API keys are available even if this file is
# tested on its own. CHROMADB_DIR is an absolute path defined in config.py.
from admgpt.config import CHROMADB_DIR

# ---------------------------------------------------------------------------
# Embeddings and vector store (created once, when this module is imported)
# ---------------------------------------------------------------------------

# Embeddings model: converts text into numbers using Google's Gemini model.
# It needs GOOGLE_API_KEY in the environment (normally loaded from .env).
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Chroma is the vector database.
#   collection_name  : like a "table name"; all chunks live in this collection
#   embedding_function: how Chroma turns text into numbers (used for both
#                       adding documents and searching)
#   persist_directory: folder on disk where the data is saved, so uploaded
#                      documents are still there after restarting the app.
#                      We use CHROMADB_DIR from config.py (an absolute path), so
#                      the same folder is used no matter where the app is started.
#                      str(...) is needed because Chroma expects a string, not a Path.
vectorstore = Chroma(
    collection_name="agentic_chatbot_docs",
    embedding_function=embeddings,
    persist_directory=str(CHROMADB_DIR),
)


def read_file_text(file_path: str) -> str:
    """
    Read a file and return its content as plain text.

    Supported file types:
        .pdf                     -> text of every page, joined with new lines
        .docx                    -> text extracted by docx2txt
        .txt .md .py .csv        -> read directly as UTF-8 text

    Args:
        file_path: Path of the file on disk.

    Returns:
        str: The text content of the file.

    Raises:
        ValueError: If the file type is not supported.

    Note:
        Scanned PDFs (pages that are only images) contain no real text, so the
        result will be empty. The caller (add_document_to_rag) checks for this.
    """

    path = Path(file_path)

    # .suffix is the file extension (e.g. ".PDF"); lower() so ".PDF" == ".pdf"
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            # extract_text() can return None for pages without text,
            # so `or ""` turns None into an empty string
            text += page.extract_text() or ""
            text += "\n"  # keep pages separated

        return text

    if suffix == ".docx":
        return docx2txt.process(file_path)

    if suffix in [".txt", ".md", ".py", ".csv"]:
        # errors="ignore": skip characters that cannot be decoded
        # instead of crashing
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV.")


def _file_hash(file_path: str) -> str:
    """
    Calculate a "fingerprint" (SHA-256 hash) of a file's content.

    Two files with exactly the same content always get the same hash, and even
    a one-character change gives a completely different hash. We use it to
    tell whether an uploaded file has really changed.

    The leading underscore means "internal helper, only used in this file".

    Args:
        file_path: Path of the file on disk.

    Returns:
        str: The hash as a hexadecimal string (64 characters).
    """

    sha256 = hashlib.sha256()

    # Read the file in 1 MB pieces, so a big file never has to fit in memory at once
    with open(file_path, "rb") as f:  # "rb" = read raw bytes
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def _find_existing_chunks(thread_id: str, filename: str) -> dict:
    """
    Look up the chunks already stored for one file in one chat.

    Args:
        thread_id: The chat to look in.
        filename: The file name (stored in metadata as "source").

    Returns:
        dict: Chroma's result with "ids" (the chunk IDs) and "metadatas".
            "ids" is an empty list if this file was never uploaded in this chat.
    """

    # Chroma needs "$and" to combine two conditions in one filter
    return vectorstore.get(
        where={"$and": [{"thread_id": thread_id}, {"source": filename}]},
        include=["metadatas"],
    )


def add_document_to_rag(file_path: str, thread_id: str):
    """
    Index a file: read it, cut it into chunks, and store the chunks in Chroma.

    Uploading the same file again is handled like this (a file is "the same"
    when thread_id AND file name match):
        - Same name, same content    -> skipped (no embedding cost at all)
        - Same name, changed content -> the old chunks are deleted, then the
                                        new ones are stored ("replaced")
        - Never uploaded before      -> stored as new ("added")

    Args:
        file_path: Path of the uploaded file on disk.
        thread_id: The chat the file belongs to. It is saved as metadata on every
            chunk so that later searches only see files from the SAME chat.

    Returns:
        dict: {
            "filename": <file name>,
            "chunks": <number of chunks stored for this file>,
            "status": "added" | "replaced" | "skipped",
        }

    Raises:
        ValueError: If the file type is unsupported, or no text could be
            extracted (for example a scanned PDF).
    """

    filename = Path(file_path).name

    # Step 0: check whether this file was uploaded before.
    # We do this FIRST, before reading the text or calling the embeddings model,
    # so a skipped upload costs nothing.
    file_hash = _file_hash(file_path)
    existing = _find_existing_chunks(thread_id, filename)
    old_ids = existing["ids"]

    if old_ids:
        # All chunks of one upload carry the same hash, so checking the first is enough.
        # Chunks stored before this feature existed have no file_hash (None),
        # so they never match and are simply replaced once.
        old_hash = existing["metadatas"][0].get("file_hash")

        if old_hash == file_hash:
            return {"filename": filename, "chunks": len(old_ids), "status": "skipped"}

        status = "replaced"
    else:
        status = "added"

    # Step 1: get the plain text of the file
    text = read_file_text(file_path)

    # strip() removes spaces/new lines; an empty result means nothing useful
    if not text.strip():
        raise ValueError("No text could be extracted from this file.")

    # Step 2: cut the text into chunks.
    # Why? A whole document is too long to search precisely or to send to the
    # model. Small chunks let us find just the relevant parts.
    #   chunk_size=600    : each chunk has at most about 600 characters
    #   chunk_overlap=150 : neighbouring chunks share 150 characters, so a
    #                       sentence cut at a boundary is not lost
    # RecursiveCharacterTextSplitter tries to split at paragraphs first, then
    # lines, then spaces, so chunks stay as natural as possible.
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)

    chunks = splitter.split_text(text)

    # Step 3: wrap each chunk in a Document with metadata.
    # The metadata is what makes per-chat search possible later (see the
    # `filter` in retrieve_from_rag) and lets us show the file name as a source.
    docs: list[Document] = [
        Document(
            page_content=chunk,
            # file_hash lets the next upload of this file be compared with this one
            metadata={
                "thread_id": thread_id,
                "source": filename,
                "file_hash": file_hash,
            },
        )
        for chunk in chunks
    ]

    # Step 4: if this file was uploaded before with different content, delete the
    # old chunks first. We do this only NOW, after reading and splitting the new
    # file succeeded, so a failed upload never destroys the old data.
    # Deleting first (instead of just overwriting) also matters because the new
    # version may have fewer chunks, and leftover old chunks would still show up
    # in searches.
    if old_ids:
        vectorstore.delete(ids=old_ids)

    # Step 5: store the chunks. Chroma calls the embeddings model for us,
    # converts every chunk into numbers, and saves them to persist_directory.
    vectorstore.add_documents(docs)

    return {"filename": filename, "chunks": len(docs), "status": status}


def retrieve_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    """
    Find the document chunks most related to a question.

    Args:
        query: The user's question or search text.
        thread_id: Only chunks uploaded in this chat are searched.
        k: How many chunks to return (default 4). More chunks give the model
            more context but also more text to read (higher cost).

    Returns:
        str: The matching chunks, each labelled with its source file, separated
            by blank lines. If nothing is found, a short message saying so.
            The text is meant to be read by the AI model, which uses it to
            write the final answer.
    """

    # similarity_search: turn `query` into numbers and return the k chunks whose
    # numbers are the closest. The filter keeps only chunks whose metadata
    # thread_id matches, so one chat never sees another chat's files.
    docs = vectorstore.similarity_search(query, k=k, filter={"thread_id": thread_id})

    if not docs:
        return "No relevant uploaded document content found."

    results = []

    # enumerate(..., start=1) gives numbers 1, 2, 3... for the labels
    for i, doc in enumerate(docs, start=1):
        # .get() returns the default value if "source" is missing
        source = doc.metadata.get("source", "uploaded document")
        results.append(f"[Source {i}: {source}]\n{doc.page_content}")

    return "\n\n".join(results)
