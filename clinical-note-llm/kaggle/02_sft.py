"""QLoRA SFT: 대화 → 4구역 JSON 진료기록.

- 4bit(NF4) 양자화한 베이스 위에 LoRA 어댑터만 학습. T4에는 bfloat16이 없어 fp16 AMP.
- 답(assistant) 부분에만 loss를 건다 — 대화 원문을 외우는 게 아니라 기록 작성을 배우게.
- torchrun으로 띄우면 T4 2장 DDP, python으로 띄우면 1장. 같은 설정으로 학습 시간을 비교한다.
- --merge-only: 학습된 어댑터를 fp16 베이스에 합쳐 저장. T4에서 vLLM LoRA 서빙이 불안정하다는
  보고가 있어 병합된 모델을 vLLM으로 띄운다.

Kaggle (GPU T4 x2) — vLLM을 설치하기 전에 학습한다 (vllm이 torch를 CUDA 13 빌드로 바꾼다):
    !pip install -q trl peft bitsandbytes
    !git clone -q -b clinical-note-llm https://github.com/ejrtn/healthcare.git
    %cd healthcare/clinical-note-llm
    !torchrun --nproc_per_node 2 kaggle/02_sft.py                 # 2장 DDP
    !CUDA_VISIBLE_DEVICES=0 python kaggle/02_sft.py --run-name 1gpu   # (비교용) 1장
    !python kaggle/02_sft.py --merge-only
그다음 vLLM 설치 후 평가:
    !pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio
    !python kaggle/01_baseline.py --model outputs/sft/qwen3b-r16/merged --group sft --tag qwen3b-r16 --mode json
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_split(name):
    """messages 형식 → TRL prompt-completion 형식 (completion에만 loss가 걸린다)."""
    from datasets import Dataset
    rows = []
    with open(ROOT / "data" / "sft" / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            rows.append({"prompt": msgs[:-1], "completion": msgs[-1:]})
    return Dataset.from_list(rows)


def train(args, out_dir):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
        torch_dtype=torch.float16,
        device_map={"": local_rank},  # DDP: 프로세스마다 자기 GPU에 모델 전체를 올린다
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    ))
    if local_rank == 0:
        model.print_trainable_parameters()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=1,
        # GPU 수와 무관하게 유효 배치 8을 유지해야 1장/2장 비교가 공정하다
        gradient_accumulation_steps=max(1, 8 // world_size),
        max_length=args.max_length,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=2,
        report_to="none",
        seed=0,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=load_split("train"),
        eval_dataset=load_split("valid"),
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(out_dir / "adapter")
        tokenizer.save_pretrained(out_dir / "adapter")
        stats = {
            "model": args.model, "lora_r": args.lora_r, "epochs": args.epochs, "lr": args.lr,
            "num_gpus": world_size, "train_time_s": round(elapsed, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "log_history": trainer.state.log_history,
        }
        (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in stats.items() if k != "log_history"}, indent=2))


def merge(args, out_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, out_dir / "adapter").merge_and_unload()
    merged.save_pretrained(out_dir / "merged", safe_serialization=True)
    AutoTokenizer.from_pretrained(out_dir / "adapter").save_pretrained(out_dir / "merged")
    print(f"병합 완료: {out_dir / 'merged'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--run-name", default="qwen3b-r16")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4800)  # 학습 데이터 최대 4,681토큰 — 잘리는 예시 없음
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "sft" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(args, out_dir)
    else:
        train(args, out_dir)


if __name__ == "__main__":
    main()
