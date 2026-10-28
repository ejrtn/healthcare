"""서로 다른 법 분야 키워드로 판례를 검색해 골고루 섞은 뒤, 본문까지 받아 저장한다.

1차 시도(이전 세션) 때 발견한 두 가지 문제를 여기서 미리 걸러낸다:
1. 데이터출처명이 "대법원"이 아닌 것(국세법령정보시스템, 조세심판원 등)은
   target=prec 상세조회 자체가 안 된다 (빈 응답).
2. 판결유형이 "판결"이 아니라 "결정"(특별항고 각하 같은 절차적 결정문)인
   것은 본안 판단이 아니라서 판결요지가 비어 있다 — 평가(official summary
   비교) 대상에서 의미가 없으므로 애초에 제외한다.
"""
import json
import time

from law_api import search_cases, get_case_detail

KEYWORDS = [
    "해고", "임대차", "개인정보", "상속", "이혼",
    "사기", "교통사고", "명예훼손", "저작권", "계약해제",
]
PER_KEYWORD = 10
OVERFETCH = 6  # 필터링으로 떨어져 나갈 걸 감안해 넉넉히 받아온다


def is_usable(r):
    return r.get("데이터출처명") == "대법원" and r.get("판결유형") == "판결"


cases = {}  # 판례일련번호 -> 메타정보 (중복 제거용)
for kw in KEYWORDS:
    results = search_cases(kw, display=PER_KEYWORD * OVERFETCH)
    usable = [r for r in results if is_usable(r)][:PER_KEYWORD]
    for r in usable:
        cid = r.get("판례일련번호")
        if cid and cid not in cases:
            cases[cid] = {"keyword": kw, "사건명": r.get("사건명"), "법원명": r.get("법원명")}
    print(f"'{kw}': {len(results)}건 검색 중 대법원·판결 {len(usable)}건 채택, 누적 고유 판례 {len(cases)}건")
    time.sleep(0.5)

print(f"\n총 {len(cases)}건 상세 조회 시작...")
full_cases = []
for i, (cid, meta) in enumerate(cases.items(), 1):
    try:
        detail = get_case_detail(cid)
        if not (detail.get("판례내용") or "").strip() or not (detail.get("판결요지") or "").strip():
            print(f"[{i}/{len(cases)}] 본문/판결요지 없음, 건너뜀: {meta['사건명'][:30]}")
            continue
        detail["_keyword"] = meta["keyword"]
        detail["_판례일련번호"] = cid
        full_cases.append(detail)
        print(f"[{i}/{len(cases)}] {meta['사건명'][:40]}")
    except Exception as e:
        print(f"[{i}/{len(cases)}] 실패 ({cid}): {e}")
    time.sleep(0.5)

with open("data/cases.json", "w", encoding="utf-8") as f:
    json.dump(full_cases, f, ensure_ascii=False, indent=2)

print(f"\n완료: {len(full_cases)}건 -> data/cases.json")
