"""EXAONE LoRA 파인튜닝 스크립트.

실행:
    uv run --extra training python training/train.py

환경별 설정:
    M1 Mac  → config.yaml: load_in_4bit: false, fp16: true
    NVIDIA  → config.yaml: load_in_4bit: true  (bitsandbytes 4bit)
"""

import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
import wandb

load_dotenv()


def load_config(path: str = "training/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_chat(sample: dict, tokenizer) -> dict:
    """JSONL 샘플 → EXAONE chat template 포맷."""
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    cfg = load_config()

    wandb.init(project=cfg["wandb"]["project"], name=cfg["wandb"]["run_name"])

    model_id = cfg["model"]["base_model"]
    dtype = getattr(torch, cfg["model"]["torch_dtype"])

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    load_kwargs = {"torch_dtype": dtype}
    if cfg["model"].get("load_in_4bit"):
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
    else:
        # M1 Mac: MPS 또는 CPU
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

    # LoRA 적용
    lora_cfg = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias=lora_cfg["bias"],
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.print_trainable_parameters()

    # 데이터셋 로드 및 포맷
    dataset = load_dataset(
        "json",
        data_files={"train": cfg["data"]["train_file"], "eval": cfg["data"]["eval_file"]},
    )
    dataset = dataset.map(lambda x: format_chat(x, tokenizer))

    # 학습
    t = cfg["training"]
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        dataset_text_field="text",
        max_seq_length=cfg["model"]["model_max_length"],
        args=TrainingArguments(
            output_dir=t["output_dir"],
            num_train_epochs=t["num_train_epochs"],
            per_device_train_batch_size=t["per_device_train_batch_size"],
            gradient_accumulation_steps=t["gradient_accumulation_steps"],
            learning_rate=t["learning_rate"],
            lr_scheduler_type=t["lr_scheduler_type"],
            warmup_ratio=t["warmup_ratio"],
            weight_decay=t["weight_decay"],
            fp16=t["fp16"],
            logging_steps=t["logging_steps"],
            save_steps=t["save_steps"],
            eval_steps=t["eval_steps"],
            save_total_limit=t["save_total_limit"],
            report_to=t["report_to"],
            evaluation_strategy="steps",
        ),
    )

    trainer.train()
    trainer.save_model()
    wandb.finish()
    print(f"LoRA adapter saved to: {t['output_dir']}")


if __name__ == "__main__":
    main()
