#!/usr/bin/env bash
# 원본 데이터(data/raw)가 받은 그대로인지 확인한다: 고정한 커밋인지, 파일이 바뀌거나 추가되지 않았는지.
set -euo pipefail

RAW_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
status=0

check() {
    local dir=$1 commit=$2
    if [ ! -d "$RAW_DIR/$dir" ]; then
        echo "$dir: 없음 (scripts/download_data.sh로 받는다)"; status=1; return
    fi
    local head changed
    head=$(git -C "$RAW_DIR/$dir" rev-parse HEAD)
    changed=$(git -C "$RAW_DIR/$dir" status --porcelain --ignored)
    if [ "$head" != "$commit" ]; then
        echo "$dir: 커밋이 다르다 ($head, 고정값 $commit)"; status=1
    elif [ -n "$changed" ]; then
        echo "$dir: 원본 폴더에 바뀌거나 생긴 파일이 있다:"; echo "$changed"; status=1
    else
        echo "$dir: 원본 그대로 @ ${commit:0:7}"
    fi
}

check aci-bench  b909b2bb9cf1d19de08df15cddde7bd0179665e4
check MTS-Dialog 3ff0801933608d6f570468c13125125fb5cabdea
exit $status
