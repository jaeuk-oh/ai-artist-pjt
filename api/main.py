"""ArtistMind FastAPI 서버.

LLM 백엔드: 로컬 HuggingFace 모델 (EXAONE-3.5-2.4B-Instruct)
LoRA 어댑터: 환경변수 LORA_PATH 로 지정
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from persona.system_prompt import build_system_prompt
from memory.short_term import add_turn, get_history
from safety.input_filter import InputFilter
from safety.output_filter import OutputFilter
import serving.model as model_backend

BLOCKED_REPLY = "앗, 그건 대답하기 어려운 주제야~ 다른 얘기 해볼까? 😅"

input_filter = InputFilter()
output_filter = OutputFilter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_backend.load()  # 서버 시작 시 모델 로드
    yield


app = FastAPI(title="ArtistMind", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str
    session_id: str
    fan_name: str = ""


class ChatResponse(BaseModel):
    response: str
    filtered: bool = False


@app.get("/health")
def health():
    return {"status": "ok", "model": model_backend.MODEL_ID}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not input_filter.is_safe(req.message):
        return ChatResponse(response=BLOCKED_REPLY, filtered=True)

    # 대화 히스토리 + 현재 메시지 조합
    history = get_history(req.session_id)
    messages = [{"role": "system", "content": build_system_prompt(fan_name=req.fan_name)}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": req.message})

    raw = model_backend.generate(messages)
    response, filtered = output_filter.filter(raw)

    add_turn(req.session_id, req.message, response)
    return ChatResponse(response=response, filtered=filtered)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
