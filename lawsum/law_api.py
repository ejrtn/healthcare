"""국가법령정보 OpenAPI(판례) 호출 모듈. 표준 라이브러리만 사용(별도 패키지 불필요)."""
import json
import os
import urllib.parse
import urllib.request

# 국가법령정보 OpenAPI 신청 ID. 저장소에 올리지 않도록 환경변수로 받는다
OC = os.environ.get("LAW_OC", "")
SEARCH_URL = "http://www.law.go.kr/DRF/lawSearch.do"
DETAIL_URL = "http://www.law.go.kr/DRF/lawService.do"


def _get_json(url, params):
    qs = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    with urllib.request.urlopen(f"{url}?{qs}", timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def search_cases(query, display=10, page=1, sort="ddes"):
    """query로 판례를 검색해 판례 목록(사건명, 판례일련번호 등)을 반환한다."""
    data = _get_json(SEARCH_URL, {
        "OC": OC, "target": "prec", "type": "JSON",
        "query": query, "display": display, "page": page, "sort": sort,
    })
    return data.get("PrecSearch", {}).get("prec", [])


def get_case_detail(case_id):
    """판례일련번호로 본문(판례내용, 판시사항, 판결요지 등)을 가져온다."""
    data = _get_json(DETAIL_URL, {
        "OC": OC, "target": "prec", "ID": case_id, "type": "JSON",
    })
    return data.get("PrecService", {})
