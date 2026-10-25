"""
Project-wide configuration.

Location: src/admgpt/config.py

Everything that is shared across the project lives here:
- environment variables (.env) and SSL settings
- important paths (PROJECT_ROOT, DATA_DIR)
- model settings (DEFAULT_MODEL, ALLOWED_MODELS)

Other modules import from here, for example:
    from admgpt.config import DATA_DIR, normalize_model_name
"""

import logging
import os
from pathlib import Path

import certifi
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

# Search parent folders for a .env file and load it.
# A .env file usually holds secrets such as ANTHROPIC_API_KEY. Never commit it to Git!
load_dotenv(find_dotenv())

# Point SSL to certifi's certificate bundle to avoid
# common certificate / path problems on Windows.
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

# This file is at <root>/src/admgpt/config.py, so going up:
#   parents[0] = admgpt, parents[1] = src, parents[2] = project root
# (If you move this file, remember to change this number!)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path of the data folder (e.g. <project root>/data)
DATA_DIR = PROJECT_ROOT / "data"

# Path of the chroma database folder (e.g. <project root>/chroma_db)
CHROMADB_DIR = PROJECT_ROOT / "chroma_db"

# Path of the document folder (e.g. <project root>/document)
DOCUMENT_DIR = PROJECT_ROOT / "document"

# Create the folder if it does not exist.
# parents=True: also create missing parent folders (like `mkdir -p`)
# exist_ok=True: do not raise FileExistsError if the folder already exists
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

logger.info("Data directory ready at: %s", DATA_DIR)


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------

# Default model: read from the ANTHROPIC_MODEL env variable,
# falling back to claude-haiku-4-5 if it is not set.
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")

# Allow-list: only these models may be used. This prevents the frontend
# from sending unexpected or nonexistent model names.
ALLOWED_MODELS = {
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
}

# Newer models reject the `temperature` parameter (the API answers
# "`temperature` is deprecated for this model"), so agent.py leaves it out for these.
NO_TEMPERATURE_MODELS = {
    "claude-opus-4-8",
    "claude-sonnet-5",
}


def normalize_model_name(model_name: str | None) -> str:
    """
    Validate and clean up the model name sent from the frontend.

    Why do we need this?
        The frontend (the user) can pick a model, but we should not blindly
        trust the string it sends, so we check it against the ALLOWED_MODELS allow-list.

    Args:
        model_name: The model name chosen in the frontend. It may be None,
            an empty string, or an invalid value.

    Returns:
        str: A valid model name.
            - Missing (None / empty string)  -> returns DEFAULT_MODEL
            - Not in the allow-list          -> returns DEFAULT_MODEL
            - Valid                          -> returns the name with surrounding spaces removed

    Example:
        >>> normalize_model_name("  claude-sonnet-5 ")
        'claude-sonnet-5'
        >>> normalize_model_name("gpt-4")   # not in the allow-list
        'claude-haiku-4-5'
    """

    # No model name given -> use the default model
    if not model_name:
        return DEFAULT_MODEL

    # Strip surrounding whitespace so input like " claude-haiku-4-5 " still matches
    model_name = model_name.strip()

    # Not in the allow-list -> fall back to the default model (a safe fallback)
    if model_name not in ALLOWED_MODELS:
        return DEFAULT_MODEL

    return model_name
