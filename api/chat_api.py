from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.concurrency import run_in_threadpool

# ✅ Import your compiled LangGraph graph
# Your file: backend/rag/generator.py must contain: graph = workflow.compile()
from rag.generator import graph


router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message/question")
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional previous messages (to keep conversation context)",
    )
    debug: bool = Field(default=False, description="Return debug info if true")


class ChatResponse(BaseModel):
    answer: str
    used_history: bool
    debug: Optional[Dict[str, Any]] = None


def _run_graph(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Blocking call to graph.invoke().
    Runs inside threadpool so FastAPI stays responsive.
    """
    return graph.invoke({"messages": messages})


@router.post("/ask", response_model=ChatResponse)
async def ask_chat(request: ChatRequest) -> ChatResponse:
    user_msg = (request.message or "").strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="message is required")

    # Build MessagesState input
    messages: List[Dict[str, str]] = []

    if request.history:
        for m in request.history:
            role = (m.role or "").strip().lower()
            content = (m.content or "").strip()
            if not content:
                continue
            if role not in {"user", "assistant", "system"}:
                role = "user"
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_msg})

    try:
        # ✅ Run in threadpool (each request gets its own thread execution)
        final_state = await run_in_threadpool(_run_graph, messages)

        final_messages = final_state.get("messages", [])
        if not final_messages:
            raise RuntimeError("Graph returned no messages")

        last = final_messages[-1]
        answer = getattr(last, "content", None) or (last.get("content") if isinstance(last, dict) else None)
        if not answer:
            raise RuntimeError("No answer returned by graph")

        debug_payload = None
        if request.debug:
            # Keep debug light: return last few messages only
            debug_payload = {
                "n_messages_in": len(messages),
                "n_messages_out": len(final_messages),
                "last_roles_out": [getattr(x, "type", None) or getattr(x, "role", None) for x in final_messages[-6:]],
            }

        return ChatResponse(
            answer=answer,
            used_history=bool(request.history),
            debug=debug_payload,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat agent error: {str(e)}")
