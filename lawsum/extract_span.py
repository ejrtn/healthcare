"""판례내용에서 "법리 설명 구간"(이유 시작 ~ 사실관계 서술 시작 전)을
코드로 결정적으로 잘라낸다. LLM에게 "어디까지 가져올지"를 맡기지 않는
이유: 8차까지 프롬프트로 "끝까지 가져오라"고 시켰지만 모델이 일관되게
일찍 멈추는 걸 실측으로 확인했다 — 경계 찾기는 코드가, 다듬기는 LLM이
맡는 쪽이 더 안정적이다."""
import re


def extract_legal_reasoning_span(content):
    """"이유" 시작부터, 사실관계 서술이 시작되는 지점 직전까지를 반환한다.

    618229로 실측하다가 두 가지 경계 오류를 찾았다:
    - 시작쪽: "이유" 바로 뒤에 "상고이유(...범위에서)를 판단한다." 같은
      정형 절차 문구가 끼어 있는데(62건 중 58건, 94%), 이것도 법리가
      아니니 건너뛰어야 한다.
    - 끝쪽: "2. 원심판결 이유와 기록에 따르면" 처럼 앞에 번호와
      "원심판결 이유와"가 붙어 나오는데, "기록에 따르면" 글자 앞에서만
      끊으면 그 앞부분("2. 원심판결 이유와")이 꼬리로 남는다.

    617171(사기, 유사도 0.316)로 추가 실측하다가 세 번째 오류를 찾았다:
    "기록에 따르면"만 찾고 있었는데, "원심판결 이유와 **이 사건** 기록에
    **의하면**, 다음과 같은 사실을 알 수 있다"처럼 "따르면" 대신
    "의하면"을 쓰는 사건이 전체 62건 중 19건(31%)이나 됐다 — 매칭이
    안 되면 end=len(content)로 폴백해서 사실관계·결론까지 전부
    LLM에게 넘기게 되고, 그러면 소제목(가/나/1/2) 제거까지 LLM이
    전부 떠안아야 해서 다듬기 품질이 눈에 띄게 나빠졌다. "따르면"과
    "의하면"을 합쳐도 62건 중 37건(60%)만 커버되니, 나머지는 여전히
    폴백에 의존한다 — 100% 해결은 아니고 개선이다.
    """
    iyu_match = re.search(r"이\s{0,4}유", content)
    start = iyu_match.end() if iyu_match else 0

    # "상고이유(...)를 판단한다." 같은 정형 절차 문구가 있으면 그 뒤부터 시작
    boilerplate = re.search(r"판단한다\.", content[start:start + 400])
    if boilerplate:
        start = start + boilerplate.end()

    # "사안의 개요"/"공소사실의 요지" 등은 본문 서술 중에 우연히 등장할
    # 일이 없는 고유한 제목 표현이라 window 어디서 찾든 안전하다. 반면
    # "원심의 판단"은 "원심의 판단에는 잘못이 없다" 식으로 본문 문장
    # 안에서도 매우 흔히 쓰여서(241863 iter2에서 실측: "...관련 법리와
    # 적법하게 채택된 증거에 비추어 살펴보면, 원심의 판단에 필요한 심리를
    # 다하지 않은 채..."라는 본문 중간 문구를 제목으로 오인해 "4. 결론"
    # (주문)까지 건너뛰어버렸다) 거짓 양성 위험이 커서 따로 분리하고,
    # window 시작 바로 근처(20자 이내)에 있을 때만 "제목"으로 인정한다.
    OVERVIEW_RE = re.compile(
        r"사안의\s*개요|공소사실의\s*요지|사건\s*(?:의\s*)?경위(?:와\s*원심\s*판단)?|"
        r"사실관계의?\s*개요"
    )
    PRIOR_JUDGMENT_RE = re.compile(r"원심의?\s*판단")
    FACT_MARKER_RE = re.compile(
        r"(?:\d+\.\s*)?(?:원심(?:판결)?\s*이유(?:와|과|및)?\s*)?(?:이\s*사건\s*)?"
        r"기록에?\s*(?:따르면|의하면)\s*,?\s*(?:다음과?\s*같은|다음)\s*사실(?:을|들을|은)\s*알\s*수\s*있"
    )
    NEXT_SECTION_RE = re.compile(r"\s{2,}\d+\.\s+\S")

    # 법리보다 사실관계가 먼저 나오는 구조를 건너뛴다. 두 가지 변형이 있다:
    # (a) "1. 사안의 개요"/"공소사실의 요지" 같은 제목이 먼저 나오는 경우
    #     (62건 중 25건, 40% — 617171로 실측 확인)
    # (b) 제목 없이 바로 "1. 기록에 의하면 다음과 같은 사실을 알 수 있다"로
    #     시작하는 경우(619451로 실측 확인 — 이땐 바로 뒤의 fact_marker가
    #     시작 지점에서 매치되어 구간 길이가 0이 돼버리는 문제가 있었다)
    # 두 경우 다 "시작 지점 바로 근처(150자 이내)"에서 감지되면, 다음
    # 최상위 번호 섹션까지 건너뛴다. "1. 사안의 개요 -> 2. 원심의 판단 ->
    # 3. [진짜 쟁점, 관련 법리]"처럼 3단계로 이어지는 사건도 있어서
    # (624241, 241863로 실측) 3회까지 반복한다. "원심의 판단"도
    # 법리가 아니라 판단 경위 서술이라 OVERVIEW_RE에 포함시켰다.
    for _ in range(3):
        window = content[start:start + 150]
        overview_m = OVERVIEW_RE.search(window)
        fact_m = FACT_MARKER_RE.search(window)

        prior_judgment_m = PRIOR_JUDGMENT_RE.search(window)

        skip_to = None
        if overview_m:
            skip_to = start + overview_m.end()
        elif prior_judgment_m and prior_judgment_m.start() < 20:
            skip_to = start + prior_judgment_m.end()
        elif fact_m and fact_m.start() < 50:
            skip_to = start + fact_m.end()

        if skip_to is None:
            break
        next_section = NEXT_SECTION_RE.search(content[skip_to:])
        next_section_pos = skip_to + next_section.start() if next_section else len(content)

        # 618195로 실측: 사실관계를 건너뛰고 바로 다음 최상위 섹션으로
        # 가면 너무 멀리 가는 경우가 있다 — "1. [쟁점] 가. 사실관계 ...
        # 나. 원심판단 다. 대법원판단 1) 관련 법리"처럼, 법리가 "다음
        # 최상위 섹션"이 아니라 "같은 섹션 안, 조금 더 뒤"에 있는 구조다.
        # "관련 법리"라는 제목이 같은 섹션 안(다음 최상위 섹션 이전)에
        #있으면 그 지점에서 멈추고, 없으면 기존처럼 다음 섹션으로 간다.
        legal_principle = re.search(r"관련\s*법리", content[skip_to:next_section_pos])
        if legal_principle:
            start = skip_to + legal_principle.end()
            continue
        if not next_section:
            break
        start = next_section_pos

    # "(숫자.) (원심판결 이유와) (이 사건) 기록에 따르면/의하면
    # 다음(과 같은) 사실을 알 수 있다" 패턴을 통째로 찾아서, 그 패턴이
    # 시작하는 지점에서 끊는다(꼬리 안 남게)
    fact_marker = FACT_MARKER_RE.search(content[start:])
    if fact_marker:
        end = start + fact_marker.start()
    else:
        end = len(content)

    span = content[start:end].strip()
    # 당사자 표시 구간이 남아있으면 생기는 "】" 같은 닫는 괄호 찌꺼기 제거
    span = span.lstrip("】 \t")
    return span
