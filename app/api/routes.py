"""FastAPI routes only — no business logic here."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.schemas import QueryResponse

router = APIRouter()


class AskRequest(BaseModel):
    """
    Request body for POST /ask. Defined HERE, not in schemas.py, since
    (unlike RiskScore/QueryResponse/CodeExplanationResult) it never
    crosses INTO another internal module -- it's consumed only by this
    route function. Same boundary-crossing test applied consistently
    elsewhere in this project (e.g. why UnresolvedCall stayed a dataclass
    in call_graph.py instead of moving to schemas.py).
    """
    question: str = Field(..., min_length=1, description="A question about the httpx codebase.")


def get_graph(request: Request):
    """
    Dependency: returns the ONE compiled graph built once at startup (see
    main.py's lifespan handler) -- never rebuilt per-request. Using a
    FastAPI dependency (not a bare import-time global) means tests can
    swap in a fake graph via app.dependency_overrides, without needing a
    real API key or network access just to test this route's wiring.
    """
    return request.app.state.graph


@router.post("/ask", response_model=QueryResponse)
def ask(body: AskRequest, graph=Depends(get_graph)) -> QueryResponse:
    """
    The single endpoint per the design doc's stated success criteria:
    routes BOTH code_explanation and risk_assessment questions through
    the same compiled graph, which decides the intent internally
    (Step 7's classify_node + route_by_intent). No intent-specific logic
    lives here -- format_response_node already normalized the output
    into QueryResponse, so this function just wires the request in and
    the response out.

    No try/except here: any AppError raised anywhere inside graph.invoke()
    (retrieval, risk scoring, the call graph, the LLM) is caught by
    main.py's global exception handlers, which know how to translate each
    error type into the right HTTP status/response uniformly. Keeping
    that translation logic OUT of this function is what "no business
    logic" actually means in practice, not just in name.

    Declared as a PLAIN `def`, not `async def`: graph.invoke() is
    synchronous and does blocking I/O (the Gemini API call, git log
    subprocess calls for risk scoring) -- it never uses LangGraph's async
    path. An `async def` route calling this directly would block
    FastAPI's entire event loop for the duration of every request. A
    plain `def` route lets FastAPI run it in its background threadpool
    automatically, so one slow request doesn't stall every other
    concurrent one.
    """
    result = graph.invoke({"question": body.question})
    return result["final_output"]