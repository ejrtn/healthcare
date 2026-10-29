# API 서버 이미지. 모델은 들어 있지 않다 — LLM_BASE_URL로 vLLM(GPU) 또는 llama.cpp(CPU) 서버를 가리킨다.
# 빌드 (저장소 루트 clinical-note-llm/에서): docker build -f service/Dockerfile -t clinical-note-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY service/requirements.txt service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# 프롬프트·스키마는 학습·평가와 같은 코드를 쓴다 (src/)
COPY src/prompts.py src/schema.py src/sectiontagger.py src/faithfulness.py src/correct.py src/
COPY service/app.py service/

RUN useradd --create-home app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/metrics')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
