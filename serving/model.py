"""로컬 HuggingFace 모델 로더 (싱글턴).

지원 환경:
  - Apple Silicon (MPS): M1/M2/M3 Mac
  - CUDA: NVIDIA GPU
  - CPU: fallback

기본 모델: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
LoRA 어댑터: LORA_PATH 환경변수로 지정 (파인튜닝 후 적용)
"""

import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

load_dotenv()
if hf_token := os.getenv("HF_TOKEN"):
    login(token=hf_token)

MODEL_ID = os.getenv("MODEL_ID", "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct")
MODEL_REVISION = "e949c91dec92095908d34e6b560af77dd0c993f8"  # transformers <5.0 호환 마지막 커밋
LORA_PATH = os.getenv("LORA_PATH", "")  # 파인튜닝된 LoRA 어댑터 경로 (없으면 base 모델 사용)

_model = None
_tokenizer = None


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load() -> None:
    """모델을 메모리에 로드. 이미 로드됐으면 스킵."""
    global _model, _tokenizer
    if _model is not None:
        return

    device = _get_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    logger.info(f"Loading {MODEL_ID} on {device} ({dtype})...")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, revision=MODEL_REVISION)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        trust_remote_code=True,
        revision=MODEL_REVISION,
    ).to(device)

    # LoRA 어댑터 적용 (파인튜닝 결과물)
    if LORA_PATH and os.path.exists(LORA_PATH):
        from peft import PeftModel
        _model = PeftModel.from_pretrained(_model, LORA_PATH)
        logger.info(f"LoRA adapter loaded: {LORA_PATH}")

    _model.eval()
    logger.info("Model ready.")


def generate(
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.8,
) -> str:
    """messages: [{"role": "system"|"user"|"assistant", "content": "..."}]"""
    load()  # lazy load

    input_ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )

    # 입력 제외하고 새로 생성된 토큰만 디코딩
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)
