"""판례 요약 데모 API. 판례일련번호를 넣으면 국가법령정보에서 본문을 가져와
로컬 Llama로 요약해 공식 판결요지와 나란히 보여준다.

실행: uvicorn app:app --reload --port 8003
"""
import html
import json
import os
import re
import threading

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from law_api import get_case_detail
from summarize import get_llm, summarize_case, strip_html

app = FastAPI(title="판례 요약 데모")

_llm = None
_llm_lock = threading.Lock()
_embed_model = None
_embed_lock = threading.Lock()


def llm_instance():
    # llm/main.py와 같은 이유로 지연 로딩한다 — 8B 모델 로딩에 100초 이상
    # 걸려서, 서버가 뜨자마자 그 시간을 기다리게 만들지 않기 위함
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = get_llm()
    return _llm


def embed_model():
    # evaluate.py와 같은 모델(jhgan/ko-sroberta-multitask) — AI 요약이 사람이
    # 쓴 공식 판결요지와 얼마나 가까운지 그 자리에서 수치로 보여주기 위함
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            _embed_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    return _embed_model


class SummarizeRequest(BaseModel):
    case_id: str


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><head><meta charset="utf-8"><title>판례 요약 데모</title></head>
    <body style="font-family:sans-serif;max-width:1200px;margin:40px auto;padding:0 16px;">
      <h2>판례 요약 데모</h2>
      <p>국가법령정보의 판례일련번호를 입력하면 본문을 가져와, AI가 공식
      판결요지와 얼마나 가깝게 재현하는지 보여줍니다.
      (예: 618229 — 새마을금고 징계처분 사건)</p>
      <form onsubmit="run(event)">
        <input id="cid" placeholder="판례일련번호 (예: 618229)" style="width:300px;padding:6px;">
        <button type="submit">요약하기</button>
      </form>
      <p id="status"></p>
      <div id="result"></div>
      <script>
        function esc(s) {
          const d = document.createElement('div');
          d.innerText = s;
          return d.innerHTML;
        }
        function collapsible(title, text, open) {
          return '<details' + (open ? ' open' : '') + ' style="margin:16px 0;border:1px solid #ddd;border-radius:6px;padding:10px;">' +
                 '<summary style="cursor:pointer;font-weight:bold;">' + title + '</summary>' +
                 '<div style="white-space:pre-wrap;margin-top:10px;line-height:1.6;">' + esc(text) + '</div></details>';
        }
        function column(title, subtitle, text) {
          return '<div style="flex:1;min-width:0;border:1px solid #ddd;border-radius:6px;padding:14px;">' +
                 '<h4 style="margin:0 0 4px 0;">' + title + '</h4>' +
                 '<p style="margin:0 0 10px 0;color:#666;font-size:13px;">' + subtitle + '</p>' +
                 '<div style="white-space:pre-wrap;line-height:1.7;">' + esc(text) + '</div></div>';
        }
        async function run(e) {
          e.preventDefault();
          const cid = document.getElementById('cid').value;
          document.getElementById('status').innerText = '불러오는 중... (첫 호출은 모델 로딩 때문에 2분 이상 걸릴 수 있습니다)';
          document.getElementById('result').innerHTML = '';
          const res = await fetch('/summarize', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({case_id: cid})
          });
          const data = await res.json();
          document.getElementById('status').innerText = '';
          if (data.detail) {
            document.getElementById('result').innerText = '오류: ' + data.detail;
            return;
          }
          document.getElementById('result').innerHTML =
            '<h3>' + esc(data.case_name) + '</h3>' +
            collapsible('원본 전문 (' + data.original_length + '자)', data.original_text, false) +
            '<p style="text-align:center;font-size:18px;font-weight:bold;margin:20px 0 8px;">AI 요약 ↔ 공식 판결요지 유사도: ' +
              (data.similarity * 100).toFixed(1) + '%</p>' +
            '<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;">' +
              column('사람(법제처)이 쓴 공식 판결요지', 'API 데이터, ' + data.official_summary.length + '자', data.official_summary) +
              column('AI 요약 (LLM 결과)', data.summary_length + '자', data.ai_summary) +
            '</div>';
        }
      </script>
    </body></html>
    """


_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _br_to_newline(text):
    # data/evaluated.json의 "판례내용"은 strip_html을 거치지 않은 원본이라
    # <br/> 태그가 그대로 남아있다 — html.escape만 하면 화면에 "<br/>"
    # 문자가 그대로 보이므로, 실제 줄바꿈으로 바꾼 뒤 escape한다
    # (표시 영역이 white-space:pre-wrap이라 \n이 줄바꿈으로 렌더링된다).
    return _BR_RE.sub("\n", text or "")


def _case_accordion(c):
    name = html.escape(c.get("사건명", ""))
    kw = html.escape(c.get("_keyword", ""))
    sim = c.get("similarity_ai_vs_official", 0) * 100
    original = html.escape(_br_to_newline(c.get("판례내용", "")))
    official = html.escape(_br_to_newline(c.get("판결요지_clean", "")))
    ai_summary = html.escape(_br_to_newline(c.get("ai_summary", "")))

    return f"""
    <details style="margin:12px 0;border:1px solid #ccc;border-radius:8px;padding:0;">
      <summary style="cursor:pointer;font-weight:bold;padding:14px;background:#f7f7f7;border-radius:8px;">
        [{kw}] 유사도 {sim:.1f}% &mdash; {name}
      </summary>
      <div style="padding:16px;">
        <details style="margin:12px 0;border:1px solid #ddd;border-radius:6px;padding:10px;">
          <summary style="cursor:pointer;font-weight:bold;">원본 전문 ({len(original)}자)</summary>
          <div style="white-space:pre-wrap;margin-top:10px;line-height:1.6;">{original}</div>
        </details>
        <p style="text-align:center;font-size:16px;font-weight:bold;margin:16px 0 8px;">
          AI 요약 &harr; 공식 판결요지 유사도: {sim:.1f}%
        </p>
        <div style="display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start;">
          <div style="flex:1;min-width:280px;border:1px solid #ddd;border-radius:6px;padding:14px;">
            <h4 style="margin:0 0 4px 0;">사람(법제처)이 쓴 공식 판결요지</h4>
            <p style="margin:0 0 10px 0;color:#666;font-size:13px;">API 데이터, {len(official)}자</p>
            <div style="white-space:pre-wrap;line-height:1.7;">{official}</div>
          </div>
          <div style="flex:1;min-width:280px;border:1px solid #ddd;border-radius:6px;padding:14px;">
            <h4 style="margin:0 0 4px 0;">AI 요약 (LLM 결과)</h4>
            <p style="margin:0 0 10px 0;color:#666;font-size:13px;">{len(ai_summary)}자</p>
            <div style="white-space:pre-wrap;line-height:1.7;">{ai_summary}</div>
          </div>
        </div>
      </div>
    </details>
    """


@app.get("/review", response_class=HTMLResponse)
def review():
    with open("data/evaluated.json", encoding="utf-8") as f:
        cases = json.load(f)

    worst_by_kw = {}
    for c in cases:
        kw = c.get("_keyword", "?")
        if kw not in worst_by_kw or c["similarity_ai_vs_official"] < worst_by_kw[kw]["similarity_ai_vs_official"]:
            worst_by_kw[kw] = c

    accordions = "".join(
        _case_accordion(c) for _, c in sorted(worst_by_kw.items(), key=lambda kv: kv[1]["similarity_ai_vs_official"])
    )

    return f"""
    <html><head><meta charset="utf-8"><title>분야별 저점수 사건 검토</title></head>
    <body style="font-family:sans-serif;max-width:1200px;margin:40px auto;padding:0 16px;">
      <h2>분야별 저점수 사건 검토</h2>
      <p>분야(키워드)마다 AI 요약 유사도가 가장 낮았던 사건 1건씩, 유사도 낮은
      순으로 나열했습니다. 제목을 클릭하면 펼쳐집니다.</p>
      {accordions}
    </body></html>
    """


@app.get("/review-low", response_class=HTMLResponse)
def review_low():
    # fallback(SEMANTIC_PROMPT_WITH_APPLICATION)이 걸린 건 1단계 결과가
    # 20자 미만일 때뿐이다. 여기서는 반대로, AI가 20자 이상 "뭔가 내용
    # 있는" 요약을 냈는데도 공식 판결요지와 유사도가 0.5 미만인 사건만
    # 모은다 — 빈 출력 실패와는 다른 종류의 실패(내용은 있는데 핵심을
    # 못 짚은 경우)를 사람이 직접 읽고 판단하기 위한 화면.
    with open("data/evaluated.json", encoding="utf-8") as f:
        cases = json.load(f)

    low = [
        c for c in cases
        if c["similarity_ai_vs_official"] < 0.5 and len(c.get("ai_summary", "")) >= 20
    ]
    low.sort(key=lambda c: c["similarity_ai_vs_official"])

    accordions = "".join(_case_accordion(c) for c in low)

    return f"""
    <html><head><meta charset="utf-8"><title>저점수(50% 미만) 비(非)빈출력 사건 검토</title></head>
    <body style="font-family:sans-serif;max-width:1200px;margin:40px auto;padding:0 16px;">
      <h2>저점수(50% 미만) 비(非)빈출력 사건 검토</h2>
      <p>AI 요약 길이가 20자 이상(= 1단계 결과가 비정상적으로 짧아 fallback이
      걸린 경우가 아님)인데도, 공식 판결요지와의 유사도가 50% 미만인
      사건 {len(low)}건입니다. 유사도 낮은 순으로 나열했습니다.</p>
      {accordions}
    </body></html>
    """


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    detail = get_case_detail(req.case_id)
    content = strip_html(detail.get("판례내용"))
    if not content:
        raise HTTPException(404, "해당 판례일련번호의 본문을 찾을 수 없습니다.")

    summary = summarize_case(llm_instance(), content)
    official = strip_html(detail.get("판결요지")) or "(공식 요약 없음)"

    emb = embed_model().encode([summary, official], convert_to_tensor=True)
    similarity = cos_sim(emb[0], emb[1]).item()

    return {
        "case_name": detail.get("사건명", ""),
        "original_text": content,
        "original_length": len(content),
        "ai_summary": summary,
        "summary_length": len(summary),
        "official_summary": official,
        "similarity": round(similarity, 4),
    }
