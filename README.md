# 포트폴리오

의료 AI 관련 작업을 두 갈래로 분리해서 관리한다.

## [`llm/`](llm/README.md) — 자가 건강 체크 RAG 챗봇

질병관리청(KDCA) 공개 건강정보로 구축한 텍스트 기반 RAG 의료 정보 챗봇.
청크 크기·라우팅·리랭킹·안전 임계값 등 설계 결정을 전부 측정해서 검증했고,
임베딩 파인튜닝·에이전틱 tool calling 실험까지 포함한다.

## [`image/`](image/README.md) — 의료영상 AI

복부 CT 외상 진단(RSNA 2023 Abdominal Trauma Detection 데이터·채점 기준 활용),
CT segmentation·3D reconstruction·denoising(LiTS).

핵심 설계 철학은 **"확진이 아니라 의심"** — 정확한 병명보다 "이상 소견이
있으니 정밀 검사가 필요하다"는 1차 스크리닝 신호를 우선한다(CADe/CADt).
정제된 벤치마크보다 여러 기관에서 모은 지저분한 실전 데이터를 의도적으로
선택했다.

두 프로젝트는 각자 독립된 포트폴리오이며, 세부 내용은 각 폴더의 README를
참고한다.
