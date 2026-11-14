"""Stage 7: the FastAPI back end.

Start it with:  uv run python -m uvicorn backend.api:app --port 8000
"""
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from backend.agent import get_checkpointer
from backend.pipelines import GRAPHS, answer

load_dotenv()
log = logging.getLogger("tenk.api")

# These must be set. LangSmith is optional, but its key is still checked for a line break.
REQUIRED = ["OPENROUTER_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "NEON_DATABASE_URL",
            "SEC_USER_AGENT", "ACCESS_CODE"]
SECRETS = REQUIRED + ["LANGSMITH_API_KEY"]

MAX_QUESTION = 1000
UNAVAILABLE = "The answer service is not available right now. Try again in a minute."


def check_environment() -> None:
    """Stop at start-up if a variable is missing or has a line break. Only the NAME is ever printed.

    A value pasted with an extra line would fail later, and the library's error message would print the key."""
    for name in REQUIRED:
        if not os.environ.get(name):
            raise RuntimeError(f"Missing environment variable: {name}")
    for name in SECRETS:
        if "\n" in os.environ.get(name, "") or "\r" in os.environ.get(name, ""):
            raise RuntimeError(f"The environment variable {name} contains a line break. Paste it on one line.")


def scrub(text: str) -> str:
    """Hide the value of every secret before the text goes into a log."""
    for name in SECRETS:
        value = os.environ.get(name, "")
        if len(value) >= 4:
            text = text.replace(value, "***")
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    check_environment()
    checkpointer = get_checkpointer()   # open the Neon connections once, when the server starts
    yield
    checkpointer.conn.close()           # and close them when it stops
    get_checkpointer.cache_clear()      # so that nothing keeps using the closed pool


app = FastAPI(title="10-K Analyst", lifespan=lifespan)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


class AskRequest(BaseModel):
    question: str = ""
    chat_id: str = ""
    pipeline: str = "agent"
    companies: list[str] = []
    years: list[int] = []


@app.get("/health")
def health() -> dict:
    # Calls no paid service, so a platform can check it as often as it likes.
    return {"status": "ok"}


@app.post("/ask")
async def ask(request: Request):
    # 1. The access code comes first. compare_digest takes the same time for any wrong code.
    code = request.headers.get("X-Access-Code", "")
    if not secrets.compare_digest(code.encode(), os.environ["ACCESS_CODE"].encode()):
        return error(401, "Wrong or missing access code.")

    # 2. The request itself.
    try:
        body = AskRequest.model_validate(await request.json())
    except (ValueError, ValidationError):   # not JSON, or the fields have the wrong types
        return error(400, "The request is not valid.")
    question = body.question.strip()
    if not question:
        return error(400, "The question is empty.")
    if len(question) > MAX_QUESTION:
        return error(400, f"The question is too long. Use at most {MAX_QUESTION:,} characters.")
    if body.pipeline not in GRAPHS:
        return error(400, "Unknown pipeline. Use simple, filters, rerank, decompose, or agent.")

    # 3. Answer. answer() is slow and blocking, so it runs in a thread and the server stays responsive.
    try:
        return await run_in_threadpool(answer, question, body.pipeline, body.companies, body.years,
                                       body.chat_id or str(uuid.uuid4()))
    except Exception as e:  # OpenRouter, Qdrant, Neon, or SEC failed
        # Log the error on the server (without any secret), and show the user only a fixed message.
        log.error("Answer failed: %s", scrub(f"{type(e).__name__}: {e}"))
        return error(503, UNAVAILABLE)
