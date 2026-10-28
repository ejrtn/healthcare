"""차트의 이름·나이·성별을 규칙(필터)만으로 채운다 — LLM 판단을 쓰지 않는다.

정답 기록은 "Martha Collins is a pleasant 50-year-old female who presents…"처럼 이름·나이·성별로 시작하는 경우가 많고,
실제 EHR에는 이 세 칸이 항상 있다. 같은 데이터면 항상 같은 값이 나오도록 정해진 순서로 찾고, 값마다 어느 규칙으로
찾았는지(source)를 남긴다. 어디에도 없으면 사례 id로 시드를 고정한 무작위 값을 넣고 source="random"으로 표시한다.

  이름: 정답 기록 소개 문장 → 기록 속 호칭(Ms. Collins) → 대화 인사말("hi, martha", 기록에도 나오거나 대화에 2번 이상) → 무작위
  나이: 정답 기록("50-year-old", "50 y/o", "age 50") → 대화("fifty years old") → 기록 내용으로 범위를 정한 무작위
  성별: 정답 기록 성별 단어 → 기록 대명사 개수 → 대화 호칭(mrs./sir) → 무작위

여러 곳에서 서로 다른 값이 나오면 conflicts에 남겨 사람이 검토하게 한다(값은 앞 순위를 쓴다).
"""
import random
import re

from faithfulness import spoken_numbers

# 인사말 뒤에 오지만 이름이 아닌 말 ("hi, how are you", "hey there", DAX 녹음 안내의 "story")
NOT_NAMES = {
    "how", "there", "story", "doctor", "doc", "everyone", "everybody", "again", "guys", "dax", "sir", "maam", "ma",
    "folks", "buddy", "dear", "you", "so", "okay", "ok", "well", "um", "uh", "thanks", "thank", "welcome", "good",
    "morning", "afternoon", "evening", "nice", "dragon", "i", "im", "it", "its", "my", "the", "this", "that", "yes",
    "yeah", "hello", "hi", "hey", "mister", "miss", "missus", "mr", "mrs", "ms", "dr", "all", "both", "come", "have",
    "pleasure", "great", "glad", "long", "today", "back", "in", "and", "to", "please", "sorry", "alright", "right",
    "are", "dad", "mom", "also", "what", "who", "can", "let", "just", "hows", "nurse", "honey", "man", "too", "did",
}
FIRST_NAMES = {
    "Female": ["Mary", "Linda", "Susan", "Karen", "Lisa", "Nancy", "Sandra", "Donna", "Carol", "Michelle", "Laura",
               "Sarah", "Kimberly", "Deborah", "Jessica", "Angela", "Melissa", "Rebecca", "Emily", "Amanda"],
    "Male": ["James", "Robert", "John", "Michael", "David", "William", "Richard", "Joseph", "Thomas", "Charles",
             "Daniel", "Matthew", "Anthony", "Mark", "Steven", "Paul", "Andrew", "Kevin", "Brian", "George"],
}
SURNAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
            "Harris", "Clark", "Lewis", "Walker", "Young", "Allen", "King", "Wright", "Scott", "Hill"]

INTRO = re.compile(r"(?:^|\n)\s*([A-Z][a-z]+(?: [A-Z][a-z]+)?) is an? (?:pleasant |very pleasant )?\d+[- ]year[- ]old")
INTRO_NOAGE = re.compile(r"(?:^|\n)\s*([A-Z][a-z]+(?: [A-Z][a-z]+)?) is an? (?:pleasant |very pleasant )?(?:\w+[- ])*?"
                         r"(?:male|female|man|woman|gentleman|lady)\b")
TITLED = re.compile(r"\b(Mr|Mrs|Ms|Miss)\. ([A-Z][a-z]+)\b")
# 인사말과 이름 사이에 쉼표나 공백이 꼭 있어야 한다 — 없으면 "hirsutism", "history", "himself"의 "hi"를 인사로 읽었다
GREETING = re.compile(r"\b(?:hi|hey|hello|good (?:morning|afternoon|evening)|nice to (?:see|meet) you)(?:\s*,\s*|\s+)"
                      r"(?:(?:mr|mrs|ms|miss|mister)\.?\s+)?([a-z]{3,})\b")
NOTE_AGE = re.compile(r"\b(\d{1,3})(?:[- ]year[- ]old|\s?y/?o\b|[- ]yo\b)|\bage (\d{1,3})\b|\baged (\d{1,3})\b", re.I)
SEX_WORD = re.compile(r"\b(female|male|woman|man|gentleman|lady|girl|boy)\b", re.I)
FEMALE_WORDS = {"female", "woman", "lady", "girl"}
AGE_RANGES = [
    (re.compile(r"\b(?:is|currently|she's) pregnant\b"), (24, 38)),
    (re.compile(r"\b(?:student|college|university)\b"), (18, 24)),
    (re.compile(r"\b(?:retired|grandchild\w*|grandkid\w*|grandson|granddaughter)\b"), (65, 82)),
    (re.compile(r"\b(?:coronary|myocardial|stemi|stent|heart failure|chf|copd|stage (?:3|iii|4|iv) chronic|"
                r"arthroplasty|osteoarthritis|macular degeneration)\b"), (52, 80)),
    (re.compile(r"\b(?:diabet\w*|hypertension|high blood pressure|gout|hyperlipidemia)\b"), (40, 75)),
]


def dialogue_ages(dialogue):
    """대화에서 '<숫자> years old / year-old' 형태로 말한 나이들."""
    d = dialogue.lower()
    ages = {int(m.group(1)) for m in re.finditer(r"\b(\d{1,3})[- ](?:years?[- ]old|year-old)", d)}
    for m in re.finditer(r"((?:[a-z]+[ -]){1,3})years?[- ]old", d):
        ages |= {n for n in spoken_numbers(m.group(1)) if 1 <= n <= 110}
    return ages


def find_sex(note, dialogue):
    """(성별, 출처) — 못 찾으면 ("", "")."""
    m = SEX_WORD.search(note[:600])  # 소개 문장 근처의 성별 단어
    if m:
        return ("Female" if m.group(1).lower() in FEMALE_WORDS else "Male"), "note:word"
    she = len(re.findall(r"\b(?:she|her|hers|herself)\b", note, re.I))
    he = len(re.findall(r"\b(?:he|him|his|himself)\b", note, re.I))
    if she != he and max(she, he) >= 2:
        return ("Female" if she > he else "Male"), "note:pronoun"
    d = dialogue.lower()
    fem = len(re.findall(r"\b(?:mrs|ms|miss|ma'?am)\b", d))
    mal = len(re.findall(r"\b(?:mr|sir|mister)\b", d))
    if fem != mal:
        return ("Female" if fem > mal else "Male"), "dialogue:title"
    return "", ""


def find_name(note, dialogue):
    """(이름, 출처, 후보들) — 이름만 찾고 성이 없으면 이름만 돌려준다."""
    candidates = []
    for pat, src in [(INTRO, "note:intro"), (INTRO_NOAGE, "note:intro")]:
        m = pat.search(note)
        if m and m.group(1) not in ("The patient", "This patient", "Patient", "The"):
            candidates.append((m.group(1), src))
    for title, last in TITLED.findall(note):
        candidates.append((last, "note:title"))
    d = dialogue.lower()
    for g in GREETING.findall(d):
        if g not in NOT_NAMES and (g in note.lower() or len(re.findall(rf"\b{g}\b", d)) >= 2):
            candidates.append((g.capitalize(), "dialogue:greeting"))
    if not candidates:
        return "", "", []
    full = next((c for c, _ in candidates if " " in c), None)
    first = next((c for c, s in candidates if " " not in c and s != "note:title"), None)
    if full:
        return full, next(s for c, s in candidates if c == full), candidates
    if first:
        return first, next(s for c, s in candidates if c == first), candidates
    return "", "", candidates  # 성(Ms. Collins)만 있으면 이름은 무작위로 붙인다(fill에서)


def fill(cid, note, dialogue, current=None):
    """{"name", "age", "sex", "source": {...}, "conflicts": [...]}. current: 지금 차트 값(대조용, 값은 쓰지 않는다)."""
    rng = random.Random(cid)
    out, source, conflicts = {}, {}, []

    sex, source["sex"] = find_sex(note, dialogue)
    if not sex:
        sex, source["sex"] = rng.choice(["Female", "Male"]), "random"
    out["sex"] = sex

    # 소개 문장의 나이가 우선이다. 기록에는 "her 7-year-old son"처럼 가족의 나이도 나온다
    intro_age = re.search(r"\bis an? (?:pleasant |very pleasant )?(\d{1,3})[- ]year[- ]old", note[:800])
    note_ages = [int(next(g for g in m.groups() if g)) for m in NOTE_AGE.finditer(note)]
    if intro_age:
        note_ages = [int(intro_age.group(1))] + note_ages
    d_ages = dialogue_ages(dialogue)
    if note_ages:
        out["age"], source["age"] = str(note_ages[0]), "note:intro" if intro_age else "note"
        if len(set(note_ages)) > 1:
            conflicts.append(f"기록에 나이가 여럿(가족 나이일 수 있음): {sorted(set(note_ages))}")
        if d_ages and note_ages[0] not in d_ages:
            conflicts.append(f"기록 나이 {note_ages[0]} ≠ 대화 나이 {sorted(d_ages)}")
    elif d_ages:
        out["age"], source["age"] = str(min(d_ages)), "dialogue"
    else:
        low, high = next((r for pat, r in AGE_RANGES if pat.search(note.lower())), (28, 75))
        out["age"], source["age"] = str(rng.randint(low, high)), "random"

    name, source["name"], candidates = find_name(note, dialogue)
    firsts = {c.split()[0].lower() for c, s in candidates if s != "note:title"}
    if len(firsts) > 1:
        conflicts.append(f"이름 후보가 여럿: {sorted(firsts)}")
    titled_last = next((c for c, s in candidates if s == "note:title"), None)
    if name and " " not in name:  # 이름만 있음 → 기록 속 호칭의 성, 없으면 목록에서
        name = f"{name} {titled_last or rng.choice(SURNAMES)}"
        source["name"] += "+surname:" + ("note:title" if titled_last else "random")
    elif not name:
        name = f"{rng.choice(FIRST_NAMES[sex])} {titled_last or rng.choice(SURNAMES)}"
        source["name"] = "random" + ("+surname:note:title" if titled_last else "")
    out["name"] = name

    if current:  # LLM으로 복원한 차트 값과 다르면 표시만 한다
        for k in ["age", "sex"]:
            if current.get(k) and current[k] != out[k]:
                conflicts.append(f"복원 차트 {k} {current[k]} ≠ 규칙 {out[k]}")
    return out | {"source": source, "conflicts": conflicts}
