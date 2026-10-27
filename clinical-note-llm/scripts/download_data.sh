#!/usr/bin/env bash
# 원본 데이터셋 두 개를 data/raw/ 아래로 받는다. 재현성을 위해 커밋을 고정한다.
#   ACI-BENCH  : 진료 1회 전체 대화 -> 전체 진료기록 (CC BY 4.0)
#   MTS-Dialog : 짧은 대화 -> 진료기록 섹션 하나 (CC BY 4.0)
set -euo pipefail

RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
mkdir -p "$RAW_DIR"

fetch() {
    local url=$1 dir=$2 commit=$3
    if [ ! -d "$RAW_DIR/$dir" ]; then
        git clone -q "$url" "$RAW_DIR/$dir"
    fi
    git -C "$RAW_DIR/$dir" checkout -q "$commit"
    echo "$dir @ $commit"
}

fetch https://github.com/wyim/aci-bench.git   aci-bench  b909b2bb9cf1d19de08df15cddde7bd0179665e4
fetch https://github.com/abachaa/MTS-Dialog.git MTS-Dialog 3ff0801933608d6f570468c13125125fb5cabdea
