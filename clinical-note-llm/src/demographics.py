"""차트의 이름·나이·성별을 규칙(필터)만으로 채운다 — LLM 판단을 쓰지 않는다.

정답 기록은 "Martha Collins is a pleasant 50-year-old female who presents…"처럼 이름·나이·성별로 시작하는 경우가 많고,
실제 EHR에는 이 세 칸이 항상 있다. 같은 데이터면 항상 같은 값이 나오도록 정해진 순서로 찾고, 값마다 어느 규칙으로
찾았는지(source)를 남긴다. 어디에도 없으면 사례 id로 시드를 고정한 무작위 값을 넣고 source="random"으로 표시한다.

  이름: 정답 기록 소개 문장 → 대화 소개("karen nelson is a 3 -year-old", "next patient is grace ross")
        → 대화 인사말("hi, martha") → 무작위. 호칭이 붙은 이름(Ms. Collins, "hey miss miller")은 성으로만 쓴다
  나이: 기록 소개 문장("50-year-old", "75 y.o.", "22-month-old") → 대화 소개 → 기록의 다른 나이 표기 → 대화에서 말한 나이
        → 기록 내용으로 범위를 정한 무작위. 가족 나이("her 7-year-old son")와 때의 나이("since age 12")는 뺀다
  성별: non-binary 표기 → 기록 성별 단어(대명사와 크게 어긋나면 대명사) → 기록 대명사 → 대화 소개·호칭 → 이름 → 무작위

여러 곳에서 서로 다른 값이 나오면 conflicts에 남긴다(값은 앞 순위를 쓴다). 219건을 모두 돌려 경고·LLM 복원 차트와 다른
곳을 하나씩 읽고 규칙을 고쳤다 — 남은 경고는 정답 기록의 성별 오타 1건, 이름 철자 2가지 1건이다.
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
    "document",  # 음성 인식이 "doctor"를 "document"로 적었다
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

# 나이 표기: "50-year-old", "75 y.o.", "50 y/o", "50yo", 영유아는 "22-month-old"
AGE = r"\b(\d{1,3})\s?(?:-\s?)?(?:(years?|months?|weeks?)[- ]old\b|y\.\s?o\b\.?|y/o\b|yo\b)"
NAME = r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)"
# "Jerry Nguyen a 54-year-old"처럼 is가 빠지거나 "Next patient Ashley James is ..."로 시작하는 기록도 있다
INTRO = re.compile(rf"(?:^|\n|\.\s+)\s*(?:Next patient,? )?{NAME},? (?:is )?an? (?:pleasant |very pleasant )?{AGE}")
INTRO_NOAGE = re.compile(rf"(?:^|\n)\s*{NAME} is an? (?:pleasant |very pleasant )?(?:\w+[- ])*?"
                         r"(?:male|female|man|woman|gentleman|lady)\b")
# 의사가 녹음 첫머리에 환자를 소개하는 말: "[doctor] karen nelson is a 3 -year-old female ..."
DIALOGUE_INTRO = re.compile(r"\b([a-z]{2,}) ([a-z]{2,}) is an? " + AGE + r"((?:[^.?\n]){0,40})")
# 이름만 부르는 소개("carolyn is a 34 -year-old female", "so elizabeth is a 53 -year-old female")와 "next patient is grace ross"
DIALOGUE_INTRO1 = re.compile(r"(?:^|\]\s*|[.,?!]\s+)(?:(?:so|okay|ok|alright|and|now)\s*,?\s+)?([a-z]{3,}) is an? " + AGE)
# 첫 발화에서 이름 부르기: "[doctor] so sophia i see that you...", "[doctor] alright okay theresa so i understand..."
# 뒤에 대명사·접속사가 바로 와야 이름으로 본다 ("so sorry to keep you waiting"을 이름으로 읽지 않게)
OPENING = re.compile(r"^\[doctor\]\s*(?:(?:so|okay|ok|alright|well)\s*,?\s+)+([a-z]{3,})\s*,?\s+"
                     r"(?=(?:i|so|how|what|tell|you|it|we|let|thanks|nice|good)\b)")
NEXT_PATIENT = re.compile(r"\bnext (?:patient )?is ([a-z]{2,}) ([a-z]{2,})\b")
NOT_PERSON = {"she", "he", "patient", "who", "which", "that", "this", "it", "there", "today"}
TITLED = re.compile(r"\b(Mr|Mrs|Ms|Miss)\. ([A-Z][a-z]+)\b")
# 인사말과 이름 사이에 쉼표나 공백이 꼭 있어야 한다 — 없으면 "hirsutism", "history", "himself"의 "hi"를 인사로 읽었다.
# 호칭이 붙은 이름("hey , miss miller")은 성이다
GREETING = re.compile(r"\b(?:hi|hey|hello|good (?:morning|afternoon|evening)|nice to (?:see|meet) you)(?:\s*,\s*|\s+)"
                      r"((?:mr|mrs|ms|miss|mister)\.?\s+)?([a-z]{3,})\b")
SEX_WORD = re.compile(r"\b(female|male|woman|man|gentleman|lady|girl|boy)\b", re.I)
NONBINARY = re.compile(r"\bnon-?\s?binary\b", re.I)
FEMALE_WORDS = {"female", "woman", "lady", "girl"}
# 환자가 아닌 사람의 나이("her 7-year-old son", "Brother: 9 years old")나 어떤 일이 있던 때의 나이("since age 12")를 거른다
FAMILY = re.compile(r"\b(?:son|sons|daughter|daughters|brother|sister|twins?|child|children|kids?|baby|grand\w+|niece|"
                    r"nephew|mother|father|mom|dad|husband|wife|cousin|sibling\w*|parents?|friend)\b", re.I)
AT_EVENT = re.compile(r"\b(?:since|at|by|until|before|after|when|from|was|were|in (?:his|her|their))\s+(?:approximately |about |around )?$",
                      re.I)
AGE_RANGES = [
    (re.compile(r"\b(?:is|currently|she's) pregnant\b"), (24, 38)),
    (re.compile(r"\b(?:student|college|university)\b"), (18, 24)),
    (re.compile(r"\b(?:retired|grandchild\w*|grandkid\w*|grandson|granddaughter)\b"), (65, 82)),
    (re.compile(r"\b(?:coronary|myocardial|stemi|stent|heart failure|chf|copd|stage (?:3|iii|4|iv) chronic|"
                r"arthroplasty|osteoarthritis|macular degeneration)\b"), (52, 80)),
    (re.compile(r"\b(?:diabet\w*|hypertension|high blood pressure|gout|hyperlipidemia)\b"), (40, 75)),
]


def age_value(num, unit):
    """"22", "months" → "22 months". 해 단위는 숫자만 (차트의 다른 나이와 같은 모양)."""
    unit = (unit or "year").lower()
    return num if unit.startswith("year") or not unit[0].isalpha() else f"{num} {unit.rstrip('s')}s"


def patient_ages(text):
    """기록·대화에서 환자 나이로 볼 수 있는 표기들 [(값, 위치)]. 가족 나이·'since age 12' 같은 때의 나이는 뺀다."""
    out = []
    for m in re.finditer(AGE, text, re.I):
        around = text[max(0, m.start() - 25):m.end() + 25]
        if FAMILY.search(around) or AT_EVENT.search(text[max(0, m.start() - 30):m.start()]):
            continue
        out.append((age_value(m.group(1), m.group(2) or "year"), m.start()))
    return out


def dialogue_ages(dialogue):
    """대화에서 말로 한 나이("i'm fifty years old")들. 가족 이야기는 뺀다."""
    d = dialogue.lower()
    ages = {v for v, _ in patient_ages(d) if v.isdigit()}
    for m in re.finditer(r"((?:[a-z]+[ -]){1,3})years?[- ]old", d):
        # 가족 이야기, 지난 일("when i was around eight years old")은 뺀다
        if FAMILY.search(d[max(0, m.start() - 25):m.end() + 25]) or \
                re.search(r"\b(?:was|were|since|when|until)\b[^.?]{0,25}$", d[max(0, m.start() - 40):m.start()]):
            continue
        if "half" in m.group(1):  # "two and a half years old": 근거 검사용 소수 조합(25)이 나이로 읽히지 않게
            continue
        ages |= {str(n) for n in spoken_numbers(m.group(1)) if 1 <= n <= 110}
    return ages


def pronoun_counts(note):
    she = len(re.findall(r"\b(?:she|her|hers|herself)\b", note, re.I))
    he = len(re.findall(r"\b(?:he|him|his|himself)\b", note, re.I))
    return she, he


def find_sex(note, dialogue):
    """(성별, 출처, 경고) — 못 찾으면 ("", "", None)."""
    if NONBINARY.search(note) or NONBINARY.search(dialogue):
        return "Non-binary", "note:word" if NONBINARY.search(note) else "dialogue:word", None
    she, he = pronoun_counts(note)
    by_pronoun = ("Female" if she > he else "Male") if she != he and max(she, he) >= 2 else ""
    m = SEX_WORD.search(note[:600])  # 소개 문장 근처의 성별 단어
    if m:
        word = "Female" if m.group(1).lower() in FEMALE_WORDS else "Male"
        # 정답 기록의 오타: "44-year-old male"인데 she/her가 29번 → 대명사를 따른다
        if by_pronoun and by_pronoun != word and max(she, he) >= 5 and max(she, he) >= 5 * min(she, he):
            return by_pronoun, "note:pronoun", f"기록 성별 단어 '{m.group(1)}'와 대명사(she {she}·he {he})가 달라 대명사를 따름"
        return word, "note:word", None
    if by_pronoun:
        return by_pronoun, "note:pronoun", None
    d = dialogue.lower()
    intro = DIALOGUE_INTRO.search(d)
    if intro and SEX_WORD.search(intro.group(5)):
        return ("Female" if SEX_WORD.search(intro.group(5)).group(1) in FEMALE_WORDS else "Male"), "dialogue:intro", None
    fem = len(re.findall(r"\b(?:mrs|ms|miss|ma'?am)\b", d))
    mal = len(re.findall(r"\b(?:mr|sir|mister)\b", d))
    if fem != mal:
        return ("Female" if fem > mal else "Male"), "dialogue:title", None
    return "", "", None


def is_name(word):
    return word not in NOT_NAMES and word not in NOT_PERSON and not FAMILY.fullmatch(word)


def find_name(note, dialogue):
    """(이름, 출처, 이름 후보들, 성 후보들). 이름 후보는 (값, 출처), 성 후보는 (성, 출처)."""
    names, surnames = [], []
    for pat in [INTRO, INTRO_NOAGE]:  # 한 기록에 "Mr. Fisher is a 59-year-old"와 "Joseph Fisher is a ..."가 함께 있기도 하다
        for m in pat.finditer(note):
            if m.group(1).split()[0] not in ("The", "This", "Patient", "He", "She", "They", "Next"):
                names.append((m.group(1), "note:intro"))
    surnames += [(last, "note:title") for _, last in TITLED.findall(note)]
    d = dialogue.lower()
    for m in list(DIALOGUE_INTRO.finditer(d)) + list(NEXT_PATIENT.finditer(d)):
        first, last = m.group(1), m.group(2)
        if is_name(first) and is_name(last):
            names.append((f"{first.capitalize()} {last.capitalize()}", "dialogue:intro"))
    for m in DIALOGUE_INTRO1.finditer(d):
        if is_name(m.group(1)):
            names.append((m.group(1).capitalize(), "dialogue:intro"))
    opening = OPENING.match(d)
    if opening and is_name(opening.group(1)):
        names.append((opening.group(1).capitalize(), "dialogue:opening"))
    for title, g in GREETING.findall(d):
        if g in NOT_NAMES:
            continue
        if title:
            surnames.append((g.capitalize(), "dialogue:title"))
        # 기록에도 나오거나, 대화에 두 번 이상 나오거나, 진료 첫인사("[doctor] well good morning emily")이면 이름으로 본다
        elif g in note.lower() or len(re.findall(rf"\b{g}\b", d)) >= 2 or d.find(g) < 120:
            names.append((g.capitalize(), "dialogue:greeting"))
    last_names = {s.lower() for s, _ in surnames}
    # 성을 이름으로 쓰지 않는다 ("Ms. James" → "James James"가 되었다)
    names = [(n, s) for n, s in names if " " in n or n.lower() not in last_names]
    full = next(((n, s) for n, s in names if " " in n), None)
    if full:
        return full[0], full[1], names, surnames
    if names:
        return names[0][0], names[0][1], names, surnames
    return "", "", names, surnames


def fill(cid, note, dialogue, current=None):
    """{"name", "age", "sex", "source": {...}, "conflicts": [...]}. current: 지금 차트 값(대조용, 값은 쓰지 않는다)."""
    rng = random.Random(cid)
    out, source, conflicts = {}, {}, []

    name, source["name"], names, surnames = find_name(note, dialogue)

    sex, source["sex"], warn = find_sex(note, dialogue)
    if warn:
        conflicts.append(warn)
    if not sex:
        first = name.split()[0] if name else ""
        known = next((k for k, v in FIRST_NAMES.items() if first in v), None)
        sex, source["sex"] = (known, "name") if known else (rng.choice(["Female", "Male"]), "random")
    out["sex"] = sex

    # 나이: 기록 소개 문장 → 대화 소개 문장 → 기록의 다른 환자 나이 표기 → 대화에서 말한 나이 → 무작위
    intro = INTRO.search(note[:1500])
    d_intro = DIALOGUE_INTRO.search(dialogue.lower())
    note_ages = patient_ages(note)
    d_ages = dialogue_ages(dialogue)
    if intro:
        out["age"], source["age"] = age_value(intro.group(2), intro.group(3)), "note:intro"
    elif d_intro:
        out["age"], source["age"] = age_value(d_intro.group(3), d_intro.group(4)), "dialogue:intro"
    elif note_ages:
        out["age"], source["age"] = note_ages[0][0], "note"
        if len({v for v, _ in note_ages}) > 1:
            conflicts.append(f"기록에 나이 표기가 여럿: {sorted({v for v, _ in note_ages})}")
    elif d_ages:
        out["age"], source["age"] = min(d_ages, key=int), "dialogue"
    else:
        low, high = next((r for pat, r in AGE_RANGES if pat.search(note.lower())), (28, 75))
        out["age"], source["age"] = str(rng.randint(low, high)), "random"
    if source["age"] != "random" and out["age"].isdigit() and d_ages and out["age"] not in d_ages:
        conflicts.append(f"나이 {out['age']} ≠ 대화에서 말한 나이 {sorted(d_ages, key=int)}")

    firsts = {n.split()[0].lower() for n, _ in names}
    if len(firsts) > 1:
        conflicts.append(f"이름 후보가 여럿: {sorted(firsts)}")
    last = surnames[0][0] if surnames else None
    if name and " " not in name:  # 이름만 있음 → 호칭의 성, 없으면 목록에서
        name = f"{name} {last or rng.choice(SURNAMES)}"
        source["name"] += "+surname:" + (surnames[0][1] if last else "random")
    elif not name:
        name = f"{rng.choice(FIRST_NAMES.get(sex, FIRST_NAMES['Female'] + FIRST_NAMES['Male']))} {last or rng.choice(SURNAMES)}"
        source["name"] = "random" + (f"+surname:{surnames[0][1]}" if last else "")
    out["name"] = name

    if current:  # LLM으로 복원한 차트 값과 다르면 표시만 한다
        for k in ["name", "age", "sex"]:
            cur = re.sub(r"^(?:mr|mrs|ms|miss)\.? ", "", (current.get(k) or "").lower())
            mine = out[k].lower()
            # "Ms. Hill" ↔ "Mary Hill", "22" ↔ "22 months"는 같은 값으로 본다
            if cur and cur != mine and not mine.endswith(" " + cur) and not mine.startswith(cur + " "):
                conflicts.append(f"복원 차트 {k} {current[k]} ≠ 규칙 {out[k]}")
    return out | {"source": source, "conflicts": conflicts}
