"""처음 보는 판례 100건(평가 전용)을 받는다. 기존 data/cases.json(62건)은 건드리지 않는다.

프롬프트(13차까지)와 후처리 규칙은 모두 cases.json 62건 점수를 보며 정했다. 그 62건으로 잰 점수(A.X 0.952)는
"이 62건에 맞춘 점수"일 수 있어서, 같은 조건(10개 분야 × 10건, 대법원·판결·판결요지 있음)으로 새 판례를 모으고
프롬프트·규칙을 하나도 바꾸지 않은 채 한 번만 평가한다.

    LAW_OC=<국가법령정보 API ID> python fetch_heldout.py

결과: data/heldout_cases.json (cases.json과 같은 형식)
"""
import json
import time

from fetch_cases import KEYWORDS, is_usable
from law_api import get_case_detail, search_cases

PER_KEYWORD = 10
PAGE_SIZE = 100
MAX_PAGES = 5


def main():
    seen = {str(c["_판례일련번호"]) for c in json.load(open("data/cases.json", encoding="utf-8"))}
    picked = []
    for kw in KEYWORDS:
        got = 0
        for page in range(1, MAX_PAGES + 1):
            results = search_cases(kw, display=PAGE_SIZE, page=page)
            if not results:
                break
            for r in results:
                cid = str(r.get("판례일련번호") or "")
                if got >= PER_KEYWORD or not cid or cid in seen or not is_usable(r):
                    continue
                detail = get_case_detail(cid)
                time.sleep(0.3)
                if not (detail.get("판례내용") or "").strip() or not (detail.get("판결요지") or "").strip():
                    continue
                detail["_keyword"], detail["_판례일련번호"] = kw, cid
                picked.append(detail)
                seen.add(cid)
                got += 1
            if got >= PER_KEYWORD:
                break
            time.sleep(0.5)
        print(f"'{kw}': {got}건")
    with open("data/heldout_cases.json", "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)
    print(f"완료: {len(picked)}건 -> data/heldout_cases.json (기존 62건과 겹치는 판례 0건)")


if __name__ == "__main__":
    main()
