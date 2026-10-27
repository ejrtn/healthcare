"""병합된 SFT 모델(safetensors) → GGUF(llama.cpp 형식) → 4bit 양자화(Q4_K_M). CPU만으로 된다.

GPU가 없는 환경(로컬 PC, 저가 서버)에서 llama.cpp로 띄우기 위한 변환이다. vLLM의 양자화(AWQ 등)와 달리
GGUF의 K-quant는 보정 데이터 없이 가중치만 보고 양자화한다.

준비물 (llama.cpp는 커밋/릴리스를 고정해 받는다):
  - llama.cpp 소스: convert_hf_to_gguf.py 와 gguf-py   (git clone --depth 1 -b <tag> https://github.com/ggml-org/llama.cpp)
  - llama-quantize 실행 파일: llama.cpp 릴리스 zip (Windows: llama-<tag>-bin-win-cpu-x64.zip) 안에 있다

    python scripts/export_gguf.py --llama-cpp tools/llama.cpp --quantize-bin tools/llama-bin/llama-quantize.exe
결과: models/note-model-f16.gguf (중간 파일), models/note-model-q4_k_m.gguf
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(ROOT / "outputs" / "sft" / "qwen3b-r16" / "merged"))
    ap.add_argument("--llama-cpp", required=True, help="llama.cpp 소스 폴더 (convert_hf_to_gguf.py가 있는 곳)")
    ap.add_argument("--quantize-bin", required=True, help="llama-quantize 실행 파일 경로")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    ap.add_argument("--keep-f16", action="store_true", help="중간 f16 GGUF(약 6GB)를 남긴다")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / "note-model-f16.gguf"
    quant = out_dir / f"note-model-{args.quant.lower()}.gguf"

    if not f16.exists():
        run([sys.executable, Path(args.llama_cpp) / "convert_hf_to_gguf.py", args.merged,
             "--outfile", f16, "--outtype", "f16"])
    run([args.quantize_bin, f16, quant, args.quant])
    if not args.keep_f16:
        f16.unlink()
    print(f"완료: {quant} ({quant.stat().st_size / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
