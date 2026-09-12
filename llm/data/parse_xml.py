import os
import glob
import json
import html
import xml.etree.ElementTree as ET
import re

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60

# "겨울철 한파대비 건강수칙"처럼 질병이 아니라 공공 안내문/캠페인성 문서인 항목은
# 걸러낸다. 이런 문서가 병명 목록에 섞이면 증상 기반 되묻기(main.py)에서
# "이 두 안내문의 차이가 뭔가요?" 같은 엉뚱한 질문을 만들어내는 원인이 된다.
# 실제 관련 질환(예: 동상, 저체온증)은 별도 항목으로 이미 등록되어 있어서
# 걸러내도 정보 손실이 없다.
NON_DISEASE_SUBJECT_PATTERNS = ["수칙", "요령", "대비"]


def is_non_disease_subject(subject: str) -> bool:
    return any(p in subject for p in NON_DISEASE_SUBJECT_PATTERNS)

def clean_text(text):
    if not text:
        return ""
    # URL만 있는 줄 제거
    if re.match(r"^https?://\S+$", text.strip()):
        return ""
    # base64 인코딩된 이미지 데이터 제거 (예: data:image/png;base64,...)
    text = re.sub(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+", "", text)
    # 통계표 등에 섞여 들어온 HTML 태그 제거 및 엔티티 디코딩
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    # 개행 문자 및 연속된 공백 제거
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    긴 본문을 문장 단위로 묶어 chunk_size 이하로 분할.
    문장 경계를 최대한 지키고, 이전 청크의 끝부분(overlap)을 다음 청크 앞에
    이어붙여 청크 경계에서 문맥이 끊기는 것을 줄인다.
    """
    if len(text) <= chunk_size:
        return [text]

    sentences = re.split(r'(?<=[.!?다요])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        # 문장 하나가 그 자체로 chunk_size를 넘는 경우(표/목록이 이어붙은 경우 등)
        # 문장 경계로는 더 못 쪼개므로 고정 길이로 강제 분할한다.
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(sentence), chunk_size - overlap):
                chunks.append(sentence[start:start + chunk_size])
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            tail = chunks[-1][-overlap:] if chunks and overlap > 0 else ""
            current = f"{tail} {sentence}".strip() if tail else sentence

    if current:
        chunks.append(current)

    return chunks

def parse_all_xml(data_dir, output_file, diseases_file, categories_file, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    xml_files = glob.glob(os.path.join(data_dir, "**/*.xml"), recursive=True)
    print(f"총 {len(xml_files)}개의 XML 파일을 찾았습니다.")

    records = []
    diseases = set()
    disease_categories = {}  # 병명 -> 실제 존재하는 카테고리 이름 목록 (등장 순서 유지)

    for file_path in xml_files:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 질환명/콘텐츠 제목 추출
            subject_elem = root.find(".//CNTNTSSJ")
            if subject_elem is None or not subject_elem.text:
                # CDATA 파싱에서 텍스트가 안 보일 수도 있으니 직접 처리할 수도 있음
                # ET는 기본적으로 CDATA 내용을 .text로 반환함
                continue
                
            subject = subject_elem.text.strip()

            if is_non_disease_subject(subject):
                continue

            # 세부 분류 리스트 추출
            cl_elements = root.findall(".//cntntsCl")
            for cl in cl_elements:
                name_elem = cl.find("CNTNTS_CL_NM")
                content_elem = cl.find("CNTNTS_CL_CN")
                
                if name_elem is not None and content_elem is not None:
                    name = name_elem.text
                    content = content_elem.text
                    
                    if name and content:
                        name = name.strip()
                        cleaned_content = clean_text(content)
                        
                        # 내용이 비었거나 너무 짧으면 스킵 (URL만 있는 경우 등)
                        if not cleaned_content or len(cleaned_content) < 10:
                            continue
                            
                        # '참고문헌'이나 이미지 링크 등의 섹션 스킵
                        if name in ["참고문헌", "첨부파일"]:
                            continue
                            
                        chunks = split_into_chunks(cleaned_content, chunk_size, overlap)
                        for idx, chunk in enumerate(chunks):
                            records.append({
                                "disease": subject,
                                "category": name,
                                "content": chunk,
                                "chunk_index": idx,
                                "total_chunks": len(chunks),
                            })
                        diseases.add(subject)
                        cat_list = disease_categories.setdefault(subject, [])
                        if name not in cat_list:
                            cat_list.append(name)
        except Exception as e:
            print(f"파일 파싱 오류 ({os.path.basename(file_path)}): {e}")

    print(f"총 {len(records)}개의 청크를 추출했습니다.")

    # 결과를 output_file에 JSONL로 저장 (레코드당 한 줄)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"결과가 {output_file}에 성공적으로 저장되었습니다.")

    # 질문에서 병명을 감지할 때 쓸 병명 목록. 부분 문자열이 겹치는 병명끼리
    # (예: "간염"/"만성 간염") 짧은 쪽이 먼저 매칭되지 않도록 긴 이름부터 정렬해둔다.
    sorted_diseases = sorted(diseases, key=len, reverse=True)
    with open(diseases_file, "w", encoding="utf-8") as f:
        json.dump(sorted_diseases, f, ensure_ascii=False, indent=2)

    print(f"병명 목록 {len(sorted_diseases)}건이 {diseases_file}에 저장되었습니다.")

    # 병명별 실제 카테고리 목록 (질문 의도 -> 카테고리 필터링에 사용)
    with open(categories_file, "w", encoding="utf-8") as f:
        json.dump(disease_categories, f, ensure_ascii=False, indent=2)

    print(f"병명별 카테고리 목록이 {categories_file}에 저장되었습니다.")

if __name__ == "__main__":
    # 실행 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "kdca_health_info")
    output_file = os.path.join(base_dir, "medical_knowledge.jsonl")
    diseases_file = os.path.join(base_dir, "diseases.json")
    categories_file = os.path.join(base_dir, "disease_categories.json")

    parse_all_xml(data_dir, output_file, diseases_file, categories_file)
