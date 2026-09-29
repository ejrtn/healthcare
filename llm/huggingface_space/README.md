---
title: 자가 진단 보조 RAG 챗봇
emoji: 🩺
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
dockerfile: huggingface_space/Dockerfile
pinned: false
---

# 자가 진단 보조 RAG 챗봇 (Hugging Face Spaces 데모)

질병관리청(KDCA) 공개 건강정보 713건을 벡터DB(ChromaDB)로 구축하고, Llama-3.1-8B(Q4
양자화)를 **100% 로컬로 서빙**하는 RAG 챗봇입니다. 외부 LLM API를 전혀 호출하지 않습니다.

- 청크 크기, 리랭킹 적용 여부, 안전 임계값(THRESHOLD) 같은 설계 선택을 전부 실측으로
  결정했습니다 — 자세한 실험 기록: [`eval/RESULTS.md`](https://github.com/ejrtn/healthcare/blob/main/llm/eval/RESULTS.md)
- 소스: [github.com/ejrtn/healthcare](https://github.com/ejrtn/healthcare)

## 참고

무료 CPU 티어에서 8B 모델을 로컬로 돌리기 때문에 **답변 생성에 수십 초 걸릴 수 있습니다**
(화면에 "답변을 생성하는 중..." 표시가 뜨면 정상 작동 중이니 기다려주세요). 한동안 방문자가
없으면 Space가 잠들었다가 첫 요청에서 다시 깨어나는 데 추가로 시간이 걸릴 수 있습니다.

MCP 서버(`mcp_server.py`)는 stdio 기반 프로토콜이라 이 웹 데모에는 포함되어 있지 않습니다 —
로컬에서 `python test_mcp_client.py`로만 확인 가능합니다.

---

## 배포 방법 (이 Space를 직접 올리는 경우)

**주의**: `llm/.gitignore`가 `models/`와 `vector_db/`를 제외하고 있어서(메인 프로젝트
저장소에 5GB짜리 바이너리가 들어가지 않게 하려는 의도적 설정), 지금 프로젝트 git 저장소
안에서 그대로 `git add .`를 하면 정작 필요한 모델·DB 파일이 안 올라갑니다. 또 메인
포트폴리오 저장소(`github.com/ejrtn/healthcare`) 역사에 큰 바이너리를 섞고 싶지 않으므로,
**Space 배포는 완전히 별도의 임시 폴더에서** 진행하는 걸 권장합니다 (메인 저장소는 전혀
건드리지 않음).

```bash
# 1. 프로젝트 밖 임시 폴더에 Space용 저장소를 새로 만든다
mkdir ~/hf-space-deploy && cd ~/hf-space-deploy
git init
git lfs install
git lfs track "vector_db/**"

# 2. 필요한 파일만 복사해온다 (메인 프로젝트 git 이력과 무관하게, models/는 제외 —
#    모델은 git에 안 올리고 Dockerfile이 빌드 중에 직접 받는다. 아래 "왜 모델
#    파일을 git에 안 올리나" 참고)
cp /path/to/llm/main.py .
cp -r /path/to/llm/static .
mkdir data && cp /path/to/llm/data/diseases.json /path/to/llm/data/disease_categories.json \
    /path/to/llm/data/medical_knowledge.jsonl data/
cp -r /path/to/llm/vector_db .
cp /path/to/llm/requirements.txt .
cp /path/to/llm/huggingface_space/Dockerfile .
cp /path/to/llm/huggingface_space/README.md .

# 3. 커밋 후 Space 원격 저장소로 push
git add .
git commit -m "Deploy RAG chatbot to HF Spaces"
git remote add space https://huggingface.co/spaces/<사용자명>/<Space이름>
git push space main
```

이렇게 하면 Dockerfile이 이 임시 폴더 루트에 있으므로 위 README 상단의 `dockerfile:` 필드는
필요 없어집니다(참고용으로 남겨둠 — 지원 안 돼도 무해함). 빌드 컨텍스트도 자동으로 이
임시 폴더가 되므로 Dockerfile의 `COPY main.py .` 같은 상대 경로가 그대로 맞습니다.

**vector_db(378MB)는 반드시 Git LFS로 추적해야** push가 됩니다(위 `git lfs track` 명령을
커밋 전에 먼저 실행) — 빠뜨리면 push가 실패하거나 파일이 잘려서 올라갑니다.

### 왜 모델 파일을 git에 안 올리나

Hugging Face Spaces 무료 티어는 **git/LFS 저장소 용량이 1GB로 제한**됩니다(실측: 4.6GB
모델 + 378MB DB를 push하다가 `Repository storage limit reached (Max: 1 GB)`로 거절당함).
그래서 모델은 git에 커밋하지 않고, `Dockerfile`이 빌드되는 시점에 Hugging Face Hub에 공개된
**같은 양자화 파일**(`bartowski/Meta-Llama-3.1-8B-Instruct-GGUF`의
`Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`)을 직접 받아 이미지 레이어에 구워 넣습니다. 이건
git/LFS 저장소 용량이 아니라 Docker 이미지 크기로 집계되어 1GB 한도에 걸리지 않습니다.
