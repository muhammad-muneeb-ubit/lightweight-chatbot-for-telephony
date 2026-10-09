
import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI(
    title="Lightweight AI Chatbot",
    description="A lightweight chatbot built with FastAPI, LangChain and Gemini",
    version="1.0.0",
)

SYSTEM_PROMPT = """
You are a friendly, helpful AI assistant.
Answer clearly and naturally.
Keep answers concise unless the user asks for detail.
Never invent facts when you do not know the answer.
"""


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    reply: str
    model: str


def get_llm():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from .env")

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.3,
        max_retries=2,
        timeout=45,
    )


@app.get("/")
def home():
    return {
        "message": "Lightweight AI Chatbot API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Include recent conversation messages for context.
        for item in request.history[-20:]:
            if item.role == "user":
                messages.append(HumanMessage(content=item.content))
            else:
                messages.append(AIMessage(content=item.content))

        messages.append(HumanMessage(content=request.message))

        response = get_llm().invoke(messages)

        reply = response.content
        if not isinstance(reply, str):
            reply = " ".join(
                str(block.get("text", ""))
                for block in reply
                if isinstance(block, dict)
            ).strip()

        return ChatResponse(
            reply=reply or "Sorry, I couldn't generate a response.",
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        )

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    
    except Exception as exc:
        logging.exception("Chatbot request failed")
        raise HTTPException(
            status_code=502,
            detail=f"AI request failed: {type(exc).__name__}: {exc}",
        )