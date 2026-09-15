"""
RAG 생성 품질 평가 (RAGAS). run_eval.py가 "검색이 맞는 카테고리를 찾아오는가"를
채점한다면, 여기서는 "생성된 답변이 근거에 충실한가(faithfulness), 질문에
맞는가(answer_relevancy)"를 LLM judge로 채점한다.

judge는 로컬(Llama-3.1-8B-Q4, 항상 포함) + Gemini/ChatGPT(API 키가
있을 때만, 선택) 중 여러 개를 동시에 시도한다. 프로덕션(main.py/mcp_server.py)의
"100% 로컬" 철학은 그대로 유지하되, 평가 도구에서만 로컬 judge의 판정을 더
큰 모델과 교차 검증한다 — 로컬 8B가 디제너레이션으로 크래시난 적이 있어서
(RESULTS.md 실험 5) 필요해졌다. 키가 없거나 호출이 실패하면 그 judge만
건너뛰고 나머지로 계속 진행하며, 로컬만 있어도 항상 돌아간다.

embedding은 judge와 무관하게 항상 로컬(ko-sroberta-multitask)을 쓴다 — judge와
embedding을 동시에 바꾸면 "어느 쪽 차이 때문인지" 비교가 불공정해지기 때문.

설치: pip install -r eval/requirements-ragas.txt (버전 고정 이유는 그 파일 안에
있음 — ragas==0.4.3 + 최신 langchain-community 조합은 import 자체가 깨지는
상위 버그가 있다). Gemini는 추가로 `google-genai` 패키지가 필요(없어도 로컬
judge는 정상 동작). ChatGPT는 ragas가 이미 의존하는 `langchain-openai`를
재사용한다.

구현 메모 (ragas 0.4.3 기준): `ragas.metrics.collections`의 최신
Faithfulness/ResponseRelevancy는 instructor의 tool-calling 구조화 출력을
강제해서 로컬 8B Q4에는 불안정하다. 대신 순수 텍스트 생성 + JSON 파싱 방식인
구(舊) `ragas.metrics`(PydanticPrompt.generate) 쪽을 썼다 — deprecated
경고는 뜨지만 이 버전에서 동작하고, judge를 langchain `LLM` 서브클래스로만
감싸면 로컬/Gemini/ChatGPT 어느 것이든 동일한 인터페이스로 바꿔 끼울 수 있다.

주의(실행 시간·비용): judge 하나당 질문 하나에 LLM을 여러 번 호출한다
(faithfulness는 주장 단위 분해 후 대조, answer_relevancy는 역질문 생성 후
임베딩 비교). judge 3개를 다 켜면 3배가 되고, 로컬은 CPU라 느리며 클라우드
judge는 무료 티어 한도에 걸릴 수 있다 — 기본은 일부만 샘플링하고 --full로
전체를 돌린다.

실행:
    cd project
    python eval/ragas_eval.py                 # GROUND_TRUTH --n개(기본 10)
    python eval/ragas_eval.py --full           # GROUND_TRUTH 33문항 전체
    python eval/ragas_eval.py --n 5            # 문항 수 직접 지정
    python eval/ragas_eval.py --judges local   # 로컬만 (API 호출 없음)

환경변수 (전부 선택, 회사명이 아니라 서비스명으로 통일):
    GEMINI_API_KEY,   GEMINI_MODEL(기본 gemini-3.6-flash)
    CHATGPT_API_KEY,  CHATGPT_MODEL(기본 gpt-4o-mini)

run_eval.py와 마찬가지로 main.py를 임포트하면 8B LLM과 리랭커까지 로드되어
첫 실행에 수십 초가 걸린다.
"""
import os
import sys
import json
import asyncio
import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv  # noqa: E402

# llm/.env가 있으면 GEMINI_API_KEY 등을 여기서 읽어 os.environ에 채운다 — 없어도
# 에러 없이 그냥 넘어간다(로컬 judge만 쓰는 경우). llm/.env.example이 템플릿이다.
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from pydantic import ConfigDict  # noqa: E402
from langchain_core.language_models.llms import LLM  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import Faithfulness, ResponseRelevancy  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402

import main  # noqa: E402 — embed_model, llm, search_disease, rerank, THRESHOLD 등을 그대로 재사용
from ground_truth import GROUND_TRUTH  # noqa: E402


# =============================================================================
# judge 어댑터 — 전부 langchain `LLM`(텍스트 in, 텍스트 out) 인터페이스로 통일
# =============================================================================

class LocalLlamaJudge(LLM):
    """main.py가 이미 로드한 llama-cpp-python 인스턴스를 RAGAS judge로 감싼다."""

    @property
    def _llm_type(self) -> str:
        return "local-llama-cpp-judge"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        try:
            response = main.llm.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=kwargs.get("temperature") or 0.01,
                repeat_penalty=1.3,
            )
            return response["choices"][0]["message"]["content"]
        except ValueError as e:
            # 로컬 8B(Q4)가 디제너레이션(토큰 무한 반복)에 빠지면 RAGAS의 재프롬프트로
            # 프롬프트+응답이 n_ctx(4096)를 넘겨 llama-cpp-python이 여기서 죽는다
            # (실측 33문항 중 1건). 예외를 전파해 배치 전체를 죽이는 대신 빈 문자열을
            # 반환해 RAGAS 자체의 파싱-재시도 로직에 맡긴다.
            print(f"    [judge 경고] 컨텍스트 윈도우 초과로 이번 judge 호출을 건너뜀: {e}")
            return ""

    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        return await asyncio.to_thread(self._call, prompt, stop, None, **kwargs)


class GeminiJudge(LLM):
    """Google Gemini API(google-genai SDK)를 RAGAS judge로 감싼다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any
    model_name: str = "gemini-3.6-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini-judge"

    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        response = self.client.models.generate_content(model=self.model_name, contents=prompt)
        return response.text or ""

    async def _acall(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        return await asyncio.to_thread(self._call, prompt, stop, None, **kwargs)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """main.py가 이미 로드한 ko-sroberta-multitask 인스턴스를 RAGAS 임베딩으로 감싼다.
    judge와 무관하게 항상 이걸 쓴다(이유는 모듈 docstring 참고)."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return main.embed_model.encode(texts).tolist()

    def embed_query(self, text: str) -> List[float]:
        return main.embed_model.encode([text]).tolist()[0]


# =============================================================================
# judge 가용성 확인 — 키가 없거나 호출이 실패하면 그 judge만 건너뛴다
# =============================================================================

def _smoke_test(llm: LLM) -> bool:
    """judge가 실제로 응답하는지 아주 짧은 호출로 확인한다."""
    try:
        text = llm.invoke("OK라고만 답해.")
        return bool(text)
    except Exception as e:
        print(f"    [judge 건너뜀] 연결 테스트 실패: {type(e).__name__}: {e}")
        return False


def build_judges(requested: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    사용 가능한 judge들을 {이름: LangchainLLMWrapper} 형태로 반환한다.
    requested가 주어지면 그 이름들만 시도하고, None이면 3개(local/gemini/
    chatgpt) 전부 시도한다. 로컬은 항상 성공한다(이미 main.py에
    로드돼 있음). 나머지는 API 키가 없거나 호출이 실패하면 조용히 건너뛴다.
    """
    order = requested if requested else ["local", "gemini", "chatgpt"]
    judges: Dict[str, Any] = {}

    for name in order:
        print(f"[judge 준비] {name}...")

        if name == "local":
            judges["local"] = LangchainLLMWrapper(LocalLlamaJudge())
            print("    OK (main.py가 이미 로드한 Llama-3.1-8B-Q4 재사용)")
            continue

        if name == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("    [judge 건너뜀] GEMINI_API_KEY 환경변수 없음")
                continue
            try:
                from google import genai
                client = genai.Client(api_key=api_key)
                model_name = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
                candidate = LangchainLLMWrapper(GeminiJudge(client=client, model_name=model_name))
                if _smoke_test(candidate.langchain_llm):
                    judges["gemini"] = candidate
                    print(f"    OK ({model_name})")
            except ImportError:
                print("    [judge 건너뜀] google-genai 패키지 미설치 (pip install google-genai)")
            except Exception as e:
                print(f"    [judge 건너뜀] {type(e).__name__}: {e}")
            continue

        if name == "chatgpt":
            api_key = os.environ.get("CHATGPT_API_KEY")
            if not api_key:
                print("    [judge 건너뜀] CHATGPT_API_KEY 환경변수 없음")
                continue
            try:
                from langchain_openai import ChatOpenAI
                model_name = os.environ.get("CHATGPT_MODEL", "gpt-4o-mini")
                chat_model = ChatOpenAI(model=model_name, api_key=api_key)
                candidate = LangchainLLMWrapper(chat_model)
                if _smoke_test(candidate.langchain_llm):
                    judges["chatgpt"] = candidate
                    print(f"    OK ({model_name})")
            except ImportError:
                print("    [judge 건너뜀] langchain-openai 패키지 미설치")
            except Exception as e:
                print(f"    [judge 건너뜀] {type(e).__name__}: {e}")
            continue

        print(f"    [judge 건너뜀] 알 수 없는 judge 이름: {name}")

    return judges


def answer_with_contexts(question: str, disease: str) -> Tuple[str, List[str]]:
    """
    main.chat()의 "병명이 감지된 질문" 분기를 그대로 재현해 (답변, 실제 생성에
    쓰인 컨텍스트 청크 목록)을 반환한다. GROUND_TRUTH는 전부 이 분기(단일 병명
    언급)에 해당하므로 그 경로만 재현한다 — 증상 기반(병명 미언급) 분기는
    run_eval.py의 SYMPTOM_CASES 채점 대상이라 여기서는 다루지 않는다.

    judge를 몇 개를 돌리든 답변 자체(로컬 8B가 생성)는 딱 한 번만 만든다 —
    "여러 judge가 같은 답변을 다르게 채점하는지" 비교하는 게 목적이라, 매
    judge마다 답변을 새로 생성하면 그 변수까지 섞여서 비교가 무의미해진다.
    """
    query_embedding = main.embed_model.encode([question]).tolist()
    category_keywords = main.detect_category_keywords(question)

    docs, distances, metadatas = main.search_disease(query_embedding, disease, category_keywords, 3)
    if not docs or distances[0] > main.THRESHOLD:
        return "관련 정보를 찾지 못했습니다.", []

    docs, distances, metadatas = main.rerank(question, docs, distances, metadatas, 3)
    context_list = [f"[{m.get('disease', '')} - {m.get('category', '')}]\n{d}" for d, m in zip(docs, metadatas)]
    context = "\n\n".join(context_list)

    user_message = f"[의학 지식]:\n{context}\n\n질문: {question}"
    response = main.llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "당신은 자가 진단 보조 AI입니다. 아래 제공된 [의학 지식]만을 바탕으로 "
                            "사용자의 질문에 친절하고 명확하게 답변하세요. 지식에 없는 내용은 절대로 "
                            "지어내지 마세요. 필요하면 병원 방문을 권유하세요.",
            },
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
    )
    answer = response["choices"][0]["message"]["content"]
    return answer, context_list


async def run_one_judge(judge_name: str, judge_llm, samples: List[Dict[str, Any]], out_dir: str):
    """samples(question/answer/contexts 캐시)를 judge 하나로 전부 채점하고 결과를 저장한다."""
    judge_embeddings = LangchainEmbeddingsWrapper(LocalSentenceTransformerEmbeddings())
    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings)

    out_path = os.path.join(out_dir, f"ragas_run_{judge_name}.json")
    rows: List[Dict[str, Any]] = []

    def save_progress():
        avg_faith = sum(r["faithfulness"] for r in rows) / len(rows) if rows else None
        avg_rel = sum(r["answer_relevancy"] for r in rows) / len(rows) if rows else None
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "judge": judge_name,
            "n_questions": len(rows),
            "avg_faithfulness": avg_faith,
            "avg_answer_relevancy": avg_rel,
            "rows": rows,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return avg_faith, avg_rel

    print()
    print("=" * 100)
    print(f"judge = {judge_name} ({len(samples)}문항)")
    print("=" * 100)

    for s in samples:
        sample = SingleTurnSample(
            user_input=s["question"], response=s["answer"], retrieved_contexts=s["contexts"]
        )
        try:
            faith_score = await faithfulness.single_turn_ascore(sample)
            relevancy_score = await answer_relevancy.single_turn_ascore(sample)
        except Exception as e:
            # 이 judge가 이 문항에서 끝내 회복 못 해도(RAGAS 자체 재시도까지 다
            # 소진한 뒤), 이 한 문항만 건너뛰고 나머지는 계속 진행한다 — 문항
            # 하나 때문에 이 judge의 --full 전체가 죽는 걸 방지.
            print(f"  {s['question']:<32} -> judge 실패로 채점 제외 ({type(e).__name__}: {e})")
            continue

        rows.append({
            "question": s["question"],
            "disease": s["disease"],
            "answer": s["answer"],
            "contexts": s["contexts"],
            "faithfulness": faith_score,
            "answer_relevancy": relevancy_score,
        })
        print(f"  {s['question']:<32} faithfulness={faith_score:.3f}  answer_relevancy={relevancy_score:.3f}")
        save_progress()  # 문항마다 즉시 저장 — 중간에 죽어도 결과 유실 방지

    avg_faith, avg_rel = save_progress()
    print("-" * 100)
    if rows:
        print(f"[{judge_name}] 평균 faithfulness: {avg_faith:.3f} / answer_relevancy: {avg_rel:.3f}")
    else:
        print(f"[{judge_name}] 채점된 문항이 없습니다.")
    print(f"결과가 {out_path}에 저장되었습니다.")
    return avg_faith, avg_rel


async def run(n: int, requested_judges: Optional[List[str]]):
    questions = GROUND_TRUTH[:n]

    # 1) 답변은 judge와 무관하게 한 번만 생성해서 재사용한다
    print("=" * 100)
    print(f"1단계: 로컬 8B로 답변 생성 ({len(questions)}문항)")
    print("=" * 100)
    samples = []
    for question, disease, _expected in questions:
        answer, contexts = answer_with_contexts(question, disease)
        if not contexts:
            print(f"  {question:<32} -> 검색 실패(THRESHOLD 미달), 전체 judge에서 채점 제외")
            continue
        samples.append({"question": question, "disease": disease, "answer": answer, "contexts": contexts})
    print(f"-> {len(samples)}/{len(questions)}문항 채점 대상 확보")

    # 2) 사용 가능한 judge 확인
    print()
    print("=" * 100)
    print("2단계: judge 가용성 확인")
    print("=" * 100)
    judges = build_judges(requested_judges)
    if not judges:
        print("\n사용 가능한 judge가 하나도 없습니다 (로컬조차 실패) — 종료합니다.")
        return

    # 3) judge별로 순서대로 채점
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    summary = {}
    for judge_name, judge_llm in judges.items():
        avg_faith, avg_rel = await run_one_judge(judge_name, judge_llm, samples, out_dir)
        summary[judge_name] = (avg_faith, avg_rel)

    print()
    print("=" * 100)
    print("전체 요약 (judge별 비교)")
    print("=" * 100)
    for judge_name, (avg_faith, avg_rel) in summary.items():
        faith_str = f"{avg_faith:.3f}" if avg_faith is not None else "N/A"
        rel_str = f"{avg_rel:.3f}" if avg_rel is not None else "N/A"
        print(f"  {judge_name:<10} faithfulness={faith_str}  answer_relevancy={rel_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="GROUND_TRUTH 33문항 전체 실행 (오래 걸림)")
    parser.add_argument("--n", type=int, default=10, help="샘플링할 문항 수 (기본 10, --full이면 무시됨)")
    parser.add_argument(
        "--judges", type=str, default=None,
        help="쉼표로 구분한 judge 목록 (local,gemini,chatgpt). 기본값은 3개 전부 시도 "
             "(API 키 없는 건 자동 건너뜀). 예: --judges local,gemini",
    )
    args = parser.parse_args()

    n_questions = len(GROUND_TRUTH) if args.full else args.n
    requested = [j.strip() for j in args.judges.split(",")] if args.judges else None
    asyncio.run(run(n_questions, requested))
