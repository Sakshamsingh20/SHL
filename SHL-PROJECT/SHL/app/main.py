import logging
import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-agent")

app = FastAPI(title="SHL Assessment Recommender")

_agent: Optional[Agent] = None


def get_agent() -> Agent:
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(default_factory=list)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty list")

    try:
        agent = get_agent()
        result = agent.respond([m.model_dump() for m in req.messages])
        return result
    except Exception as e:
        logger.exception("chat failed")
        # Fail safe rather than fail loud -- still return schema-valid response
        return ChatResponse(
            reply="Sorry, I hit an internal error processing that. Could you rephrase your request?",
            recommendations=[],
            end_of_conversation=False,
        )
