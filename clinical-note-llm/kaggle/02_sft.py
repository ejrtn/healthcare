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
    !torchrun --nproc_per_node 2 kaggle/02_sft.py && python kaggle/02_sft.py --merge-only   # 2장 DDP → 병합
    !CUDA_VISIBLE_DEVICES=0 python kaggle/02_sft.py --run-name 1gpu   # (비교용) 1장
그다음 vLLM 설치 후 평가:
    !pip install -q vllm rouge-score==0.1.2 && pip uninstall -y -q torchaudio
    !python kaggle/01_baseline.py --model outputs/sft/qwen3b-r16/merged --group sft --tag qwen3b-r16 --mode json
"""
import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_split(name, data_dir="data/sft"):
    """messages 형식 → TRL prompt-completion 형식 (completion에만 loss가 걸린다)."""
    from datasets import Dataset
    rows = []
    with open(ROOT / data_dir / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            rows.append({"prompt": msgs[:-1], "completion": msgs[-1:]})
    return Dataset.from_list(rows)


def make_config(config_cls, **kwargs):
    """설치된 TRL/transformers 버전이 모르는 설정은 빼고 만든다.

    버전마다 인자 이름이 바뀌어(예: warmup_ratio 삭제) 하나 고치면 다음 인자에서 또 죽는다.
    뺀 항목은 출력해서, 학습 조건이 조용히 바뀌지 않았는지 로그로 확인할 수 있게 한다.
    """
    known = {f.name for f in dataclasses.fields(config_cls)}
    dropped = sorted(set(kwargs) - known)
    if dropped and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[경고] 이 버전의 {config_cls.__name__}가 지원하지 않아 뺀 설정: {dropped}")
    return config_cls(**{k: v for k, v in kwargs.items() if k in known})


def ensure_pad_token(tokenizer):
    """Llama 3.x처럼 패딩 토큰이 없는 모델용. eos(<|eot_id|>)를 패딩으로 쓰면 답 끝의 eos까지 패딩으로
    취급될 수 있어, 학습용으로 비워 둔 전용 토큰이 있으면 그것을 쓴다."""
    if tokenizer.pad_token is None:
        reserved = "<|finetune_right_pad_id|>"
        tokenizer.pad_token = reserved if reserved in tokenizer.get_vocab() else tokenizer.eos_token
    return tokenizer


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
        dtype=torch.float16,
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
    train_ds, valid_ds = load_split("train", args.data_dir), load_split("valid", args.data_dir)
    # GPU 수와 무관하게 유효 배치 8을 유지해야 1장/2장 비교가 공정하다
    grad_accum = max(1, 8 // world_size)
    total_steps = math.ceil(len(train_ds) / (world_size * grad_accum)) * math.ceil(args.epochs)
    config = make_config(
        SFTConfig,
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # transformers 5.x에서 warmup_ratio가 없어져 스텝 수로 준다 (전체의 약 5%)
        warmup_steps=max(1, round(0.05 * total_steps)),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
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
    tokenizer = ensure_pad_token(AutoTokenizer.from_pretrained(args.model))
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )

    # fp16 AMP의 GradScaler는 fp32 그래디언트만 unscale한다. 베이스 체크포인트가 bf16이라
    # LoRA 파라미터가 bf16으로 만들어져 첫 optimizer step에서 죽었다 — 학습 파라미터만 fp32로 둔다.
    # (TRL이 trainer 생성 중에 dtype을 바꿀 수 있어 trainer를 만든 뒤에 맞춘다)
    n_cast = 0
    for p in trainer.model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n_cast += 1
    if local_rank == 0:
        print(f"학습 파라미터 {n_cast}개를 fp32로 변환")

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

    adapter_dir = str(out_dir / "adapter")  # peft는 Path를 받지 않는다
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(str(out_dir / "merged"), safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(str(out_dir / "merged"))
    print(f"병합 완료: {out_dir / 'merged'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--run-name", default="qwen3b-r16")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=4800)  # 학습 데이터 최대 4,681토큰 — 잘리는 예시 없음
    ap.add_argument("--data-dir", default="data/sft", help="data/sft_grounded: 대화에 없는 나이·성을 지운 정답")
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
