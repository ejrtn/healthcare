"""DPO: 2차 SFT 모델 위에 LoRA를 하나 더 얹어 "좋은 답 쪽으로, 나쁜 답에서 멀어지게" 학습한다.

- 출발점과 기준(reference) 모델은 모두 2차 SFT 병합 모델이다. 기준 모델을 따로 올리지 않고 LoRA를 끈
  같은 모델로 계산해(TRL 기본 동작) GPU 메모리를 아낀다.
- 05_dpo_candidates.py → 06_judge.py를 돌리고 나면 vLLM이 설치된 상태라 bitsandbytes(4bit)가 안 맞을 수 있어
  기본은 FP16 가중치 + LoRA다. 3B FP16(6GB)은 T4 16GB에 들어간다. --load-4bit으로 SFT와 같은 QLoRA도 쓸 수 있다.
- 메모리의 대부분은 어휘 15만 개 × 토큰 수 크기의 logits다. 답 부분 logits만 계산하는 설정
  (use_logits_to_keep)으로 대화 부분(평균 약 1,700토큰)의 logits를 만들지 않는다.

    !torchrun --nproc_per_node 2 kaggle/07_dpo.py && python kaggle/07_dpo.py --merge-only

결과: outputs/dpo/<run-name>/{adapter, merged, train_stats.json}
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle"))

_sft = __import__("02_sft")
make_config, ensure_pad_token = _sft.make_config, _sft.ensure_pad_token


def load_pairs(split, data_dir):
    from datasets import Dataset
    rows = []
    with open(ROOT / data_dir / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            rows.append({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]})
    return Dataset.from_list(rows)


def train(args, out_dir):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    quant = None
    if args.load_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.sft_model, quantization_config=quant, dtype=torch.float16, device_map={"": local_rank})
    if args.load_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.enable_input_require_grads()  # 고정된 가중치 + gradient checkpointing에서도 LoRA로 그래디언트가 흐르게
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
    train_ds, valid_ds = load_pairs("train", args.data_dir), load_pairs("valid", args.data_dir)
    grad_accum = max(1, 8 // world_size)  # SFT와 같은 유효 배치 8
    total_steps = math.ceil(len(train_ds) / (world_size * grad_accum)) * math.ceil(args.epochs)
    config = make_config(
        DPOConfig,
        output_dir=str(out_dir),
        beta=args.beta,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.1 * total_steps)),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        max_length=args.max_length,  # 학습 데이터 최대 약 4,700토큰(대화 + 답) — 잘리는 예시 없음
        max_prompt_length=args.max_length - 1500,
        use_logits_to_keep=True,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=1,
        report_to="none",
        seed=0,
    )
    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(args.sft_model))
    trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=train_ds,
                         eval_dataset=valid_ds, processing_class=tokenizer)

    # 02_sft.py와 같은 이유: fp16 AMP의 GradScaler는 fp32 그래디언트만 unscale한다
    for p in trainer.model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - start

    if trainer.is_world_process_zero():
        trainer.model.save_pretrained(out_dir / "adapter")
        tokenizer.save_pretrained(out_dir / "adapter")
        stats = {
            "sft_model": args.sft_model, "pairs_train": len(train_ds), "pairs_valid": len(valid_ds),
            "beta": args.beta, "lr": args.lr, "epochs": args.epochs, "lora_r": args.lora_r,
            "load_4bit": args.load_4bit, "num_gpus": world_size, "train_time_s": round(elapsed, 1),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "log_history": trainer.state.log_history,
        }
        (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in stats.items() if k != "log_history"}, indent=2))


def merge(args, out_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(args.sft_model, dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, str(out_dir / "adapter")).merge_and_unload()
    merged.save_pretrained(str(out_dir / "merged"), safe_serialization=True)
    AutoTokenizer.from_pretrained(args.sft_model).save_pretrained(str(out_dir / "merged"))
    print(f"병합 완료: {out_dir / 'merged'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-model", default="outputs/sft/qwen3b-r16-grounded/merged")
    ap.add_argument("--data-dir", default="data/dpo")
    ap.add_argument("--run-name", default="dpo-b0.1")
    # beta: 기준 모델에서 얼마나 멀어져도 되는지. 작을수록 많이 바뀐다. 짝이 적어 기본값(0.1)에서 시작
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=6144)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "dpo" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge(args, out_dir)
    else:
        train(args, out_dir)


if __name__ == "__main__":
    main()
