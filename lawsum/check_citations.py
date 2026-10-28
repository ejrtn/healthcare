"""요약 속 날짜·사건번호·조문 번호가 판결문 원문에 그대로 있는지 검사한다.

판결요지 추출은 "원문을 그대로 옮기는" 작업인데, A.X 요약에서 참조판례 날짜 "2017. 11. 9."가 "201. 11. 9."로
바뀐 사례를 찾았다. 임베딩 유사도는 숫자 하나가 틀려도 거의 그대로라 이런 오류를 못 잡는다. 법률 문서에서는
날짜·사건번호·조문 번호 하나가 틀리면 다른 판례·다른 조문을 가리키게 되므로, 요약에 나온 이런 표기를 모두 뽑아
원문에 있는지 확인한다.

    python check_citations.py data/summarized_ax-4.0-light.json data/summarized_llama-3.1-8b.json
"""
import argparse
import difflib
import json
import re

from summarize import strip_html

PATTERNS = {
    # 2017. 11. 9. / 201. 11. 9. (오기도 잡도록 연도 자릿수를 느슨하게)
    "date": re.compile(r"\d{2,4}\.\s*\d{1,2}\.\s*\d{1,2}\."),
    # 2014다49180, 2020도1234, 2019두12345, 2016다227625
    "case_no": re.compile(r"\d{2,4}[가-힣]{1,3}\d{2,7}"),
    # 제2조 제1호, 제750조, 제3조의2 제1항
    "article": re.compile(r"제\d+조(?:의\d+)?(?:\s*제\d+항)?(?:\s*제\d+호)?"),
}


def norm(text):
    return re.sub(r"\s+", "", text)


def check(summary, source):
    src = norm(source)
    out = {}
    for name, pat in PATTERNS.items():
        found = pat.findall(summary)
        out[name] = {"total": len(found), "missing": [f for f in found if norm(f) not in src]}
    return out


def repair(summary, source, min_ratio=0.8):
    """원문에 없는 날짜·사건번호·조문을 원문에서 가장 비슷한 표기로 바꾼다.

    판결요지 추출은 원문을 그대로 옮기는 작업이라, 요약 속 표기는 원문 어딘가에 반드시 있어야 한다. 같은 종류의
    원문 표기 중 가장 비슷한 것이 하나뿐이고 충분히 비슷할 때(한두 자리 차이)만 바꾼다. 후보가 애매하거나 비슷한
    것이 없으면(예: 원문에 없는 사건번호를 지어낸 경우) 그대로 두고 목록으로 돌려줘 사람이 보게 한다.
    """
    fixed, unresolved = summary, []
    for name, pat in PATTERNS.items():
        candidates = sorted(set(pat.findall(source)))
        for wrong in check(summary, source)[name]["missing"]:
            scored = sorted(((difflib.SequenceMatcher(None, norm(wrong), norm(c)).ratio(), c) for c in candidates),
                            reverse=True)
            if scored and scored[0][0] >= min_ratio and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                fixed = fixed.replace(wrong, scored[0][1], 1)
            else:
                unresolved.append((name, wrong))
    return fixed, unresolved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="summarized_*.json")
    ap.add_argument("--fix", action="store_true", help="원문 표기로 바로잡아 summarized_<tag>-fixed.json으로 저장")
    args = ap.parse_args()
    if args.fix:
        for path in args.files:
            cases = json.load(open(path, encoding="utf-8"))
            left = []
            for c in cases:
                c["ai_summary"], unresolved = repair(c["ai_summary"], strip_html(c.get("판례내용")))
                left += [(c.get("_판례일련번호"), *u) for u in unresolved]
            out = path.replace(".json", "-fixed.json")
            json.dump(cases, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
            print(f"{out}: 바로잡지 못한 표기 {len(left)}개 {left}")
        args.files = [p.replace(".json", "-fixed.json") for p in args.files]
    print("| 요약 | 날짜 (원문에 없음/전체) | 사건번호 | 조문 | 오류가 하나라도 있는 요약 |")
    print("|---|---|---|---|---|")
    for path in args.files:
        cases = json.load(open(path, encoding="utf-8"))
        per = [check(c["ai_summary"], strip_html(c.get("판례내용"))) for c in cases]
        cols = []
        for name in PATTERNS:
            total = sum(p[name]["total"] for p in per)
            miss = sum(len(p[name]["missing"]) for p in per)
            cols.append(f"{miss}/{total}")
        bad = sum(any(p[n]["missing"] for n in PATTERNS) for p in per)
        print(f"| {path} | {' | '.join(cols)} | {bad}/{len(per)} |")
        for c, p in zip(cases, per):
            for name in PATTERNS:
                for m in p[name]["missing"]:
                    print(f"    {c.get('_판례일련번호')} {name}: {m}")


if __name__ == "__main__":
    main()
