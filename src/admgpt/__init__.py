def main() -> None:
    """Start the AdmGPT web server (the `admgpt` command in pyproject.toml)."""

    import uvicorn

    uvicorn.run("admgpt.api.app:app", host="0.0.0.0", port=8080)
