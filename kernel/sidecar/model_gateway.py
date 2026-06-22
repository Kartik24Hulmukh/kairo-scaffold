from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(title="Model Gateway Stub")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0

class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = 1718582400
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int] = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}

@app.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    return ChatCompletionResponse(
        id="chatcmpl-123",
        model=req.model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content="Stub model response"),
                finish_reason="stop"
            )
        ]
    )
