# Hugging Face Spaces(Docker SDK)용 배포 이미지.
#
# 중요: 이 Dockerfile은 llm/ 폴더(이 파일의 부모 디렉터리)를 빌드 컨텍스트로
# 삼아야 한다 — main.py, static/, data/, vector_db/가 전부 llm/ 바로 아래
# 있기 때문이다(models/는 빌드 중에 Hugging Face Hub에서 받으므로 컨텍스트에
# 없어도 된다). 배포 방법은 이 폴더의 README.md 참고.
#
# 앱 로직(main.py)은 전혀 복제하지 않고 그대로 COPY만 한다 — RAG 파이프라인이
# 두 군데로 나뉘어서 한쪽만 고치는 실수를 막기 위함.

FROM python:3.12-slim

WORKDIR /app

# Hugging Face Hub 캐시 경로를 고정한다. 기본 경로(~/.cache/huggingface)는
# 빌드 중엔 root, 실행 중엔 appuser로 유저가 바뀌면서 서로 다른 홈 디렉터리를
# 가리키게 되어, 빌드 시점에 미리 받아둔 모델을 실행 시점에 못 찾는 문제가
# 있었다(실측: main.py가 HF_HUB_OFFLINE=1로 뜨는데 캐시를 못 찾아
# LocalEntryNotFoundError로 컨테이너가 기동조차 못 함). /app 아래로 고정해두면
# 마지막의 chown 한 번으로 빌드·실행 양쪽이 같은 캐시를 쓴다.
ENV HF_HOME=/app/hf_cache

# llama-cpp-python은 소스 빌드 시 C++ 컴파일러가 필요한데, 이 이미지엔 없다.
# 로컬에서도 겪었던 문제(nmake/CMake 컴파일러 없음)와 같은 이유로, CPU 전용
# 프리빌트 휠 인덱스를 추가로 지정해서 컴파일 없이 바로 설치되게 한다.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    -r requirements.txt

# main.py는 이미 로컬에 캐시된 모델을 Hugging Face Hub에 매번 재검증하지 않도록
# HF_HUB_OFFLINE=1로 뜬다(로컬 개발 환경 기준 최적화). 그런데 이 이미지는 매번
# 새로 빌드되는 컨테이너라 임베딩·리랭커 모델이 캐시에 전혀 없는 상태로
# 시작하는데, 오프라인 모드라 다운로드도 못 해서 그대로 크래시한다(실측 확인 —
# 8B LLM은 아래에서 미리 받아두면서 이 둘은 빠뜨려서 발생한 문제). 여기서
# 미리 받아 캐시를 채워둔다.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('jhgan/ko-sroberta-multitask'); CrossEncoder('Dongjin-kr/ko-reranker', max_length=512)"

# 앱 코드
COPY main.py .
COPY static static

# 검색에 필요한 최소 데이터만 포함 (원본 XML·txt는 런타임에 안 쓰여서 제외 —
# vector_db가 이미 만들어져 있으면 load_data()가 이 파일들도 사실 안 읽지만,
# 혹시 vector_db가 비어있는 상태로 배포되는 경우를 대비한 폴백용으로 넣어둔다)
COPY data/diseases.json data/disease_categories.json data/medical_knowledge.jsonl data/

# 이미 구축해둔 ChromaDB(~378MB) — 콜드스타트마다 처음부터 임베딩을 다시 만들지
# 않도록 미리 빌드된 걸 그대로 포함시킨다. (Space git/LFS 저장소 용량 한도가
# 무료 티어 1GB라 이 정도는 괜찮지만, 모델 파일까지 더하면 넘긴다 — 아래 참고)
COPY vector_db vector_db

# 8B 모델(gguf, ~4.6GB)은 git 저장소에 커밋하지 않는다 — Space git/LFS
# 저장소 용량 한도(무료 티어 1GB)를 모델 파일 혼자서 넘겨버리기 때문이다
# (실측: push 시도 시 "Repository storage limit reached (Max: 1 GB)" 발생).
# 대신 로컬과 똑같은 양자화 파일(Meta-Llama-3.1-8B-Instruct-Q4_K_M, bartowski
# 업로드)을 Hugging Face Hub에서 빌드 시점에 받아 이미지 레이어에 굽는다 —
# 이건 git/LFS 저장소 용량이 아니라 Docker 이미지 크기로 잡혀서 한도에 안 걸린다.
# 파일명이 대소문자까지 다르므로(main.py는 소문자 경로를 기대) 받은 뒤 이름을 맞춘다.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', filename='Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', local_dir='models')" \
    && mv models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf models/meta-llama-3.1-8b-instruct-q4_k_m.gguf

ENV HOST=0.0.0.0
ENV PORT=7860
EXPOSE 7860

# Hugging Face Spaces는 컨테이너를 root가 아닌 사용자로 돌리는 걸 권장한다.
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]
