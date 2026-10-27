"""진료기록 구역 분리기 — ACI-BENCH 공식 평가 코드에서 가져옴.

출처: https://github.com/wyim/aci-bench/blob/b909b2bb9cf1d19de08df15cddde7bd0179665e4/baselines/sectiontagger.py
(Yim et al., "ACI-BENCH", Nature Scientific Data 2023)

원본 파일에는 같은 클래스가 두 번 정의돼 있고 뒤쪽 정의(IMPRESSION 처리 포함)가
실제로 쓰인다. 그 활성 정의만 옮겼고, 정규식 문자열을 raw string으로 바꾼 것
외에는 동작을 바꾸지 않았다. 공식 채점과 같은 규칙으로 구역을 나누려고 그대로 쓴다.
"""
import re
from collections import OrderedDict

subjective_subsections = OrderedDict()
subjective_subsections['cc'] = ['cc :', 'chief complaint :', 'reason for visit :', "CHIEF COMPLAINT"]
subjective_subsections['hpi'] = ['history :', 'history of present illness :',
                                 'history of present illness', 'hpi :', 'hpi', 'hpi notes :',
                                 'interval history :', 'interval hx :', 'subjective :',
                                 "HISTORY OF PRESENT ILLNESS"]
subjective_subsections['ros'] = ['ros :', 'review of system :', 'review of systems :']
subjective_subsections['other_histories'] = ['SOCIAL HISTORY', "PAST HISTORY"]

objectiveexam_subsections = OrderedDict()
objectiveexam_subsections['pe'] = ['physical exam :', 'physical examination :', 'pe :',
                                   'physical findings :', 'examination :', 'exam :',
                                   "PHYSICAL EXAMINATION", "PHYSICAL EXAM"]
objectiveexam_subsections['vitals'] = ['VITALS REVIEWED']

objectiveresults_subsections = OrderedDict()
objectiveresults_subsections['findings'] = ['results :', 'findings :', "RESULTS"]

ap_subsections = OrderedDict()
ap_subsections['assessment'] = ['assessment :', 'a:']
ap_subsections['plan'] = ['plan :', 'plan of care :', 'p:', 'medical decision-making plan :', 'summary plan']
ap_subsections['ap'] = ['ap :', 'a / p :', 'assessment and plan :', 'assessment & plan :',
                        'disposition / plan :', "ASSESSMENT AND PLAN"]

sectcat2subsections = OrderedDict()
sectcat2subsections['subjective'] = subjective_subsections
sectcat2subsections['objective_exam'] = objectiveexam_subsections
sectcat2subsections['objective_results'] = objectiveresults_subsections
sectcat2subsections['assessment_and_plan'] = ap_subsections

NOSECTIONHEADER = 'default'
subsectionheader2section = {}
for sh, sshdicts in sectcat2subsections.items():
    for ssh in sshdicts:
        subsectionheader2section[ssh] = sh
subsectionheader2section[NOSECTIONHEADER] = NOSECTIONHEADER


class SectionTagger:

    def __init__(self, sectcat2subsections=sectcat2subsections):
        self.sectcat2subsections = sectcat2subsections
        self.compileregexes()

    def compileregexes(self):
        self.subsect2regex = {}
        for _, subsections in self.sectcat2subsections.items():
            for subsect, vlst in subsections.items():
                self.subsect2regex[subsect] = self._compile_regexexpression(vlst)

    def _compile_regexexpression(self, vlst):
        expressions = []
        otherexps = []
        for exp in vlst:
            expressions.append('(' + re.escape(exp).replace(r'\ ', r'\s*') + ')')
            if exp[-1] == ':':
                # 줄이 비어 있으면 콜론 없는 형태도 허용
                otherexps.append('(' + re.escape(exp[:-1]).replace(r'\ ', r'\s*') + ')')

        patt = r'\s*(?P<sectionheader1>' + '|'.join(expressions) + ').*'
        if otherexps:
            pattott = r'\s*(?P<sectionheader2>' + '|'.join(otherexps) + r')\s*$'
            return '(' + patt + '|' + pattott + ')'
        return patt

    def tag_sectionheaders(self, text):
        """Return: list of (subsectionheader, linenum, char_start, char_end)"""
        offset = 0
        subsects = []
        for linenum, line in enumerate(text.split('\n')):
            for subsect, rgx in self.subsect2regex.items():
                m = re.match(rgx, line, re.IGNORECASE)
                if m:
                    secthpattname = 'sectionheader1' if m.groupdict()['sectionheader1'] is not None else 'sectionheader2'
                    subsects.append((subsect, linenum, offset, offset + m.end(secthpattname)))

            # 대문자 IMPRESSION만 assessment로 본다 (소문자는 이 데이터에서 results의 일부)
            m = re.match(r'\s*IMPRESSION', line)
            if m:
                subsects.append(('impression', linenum, offset, offset + m.end()))

            offset += len(line) + 1
        return subsects

    def tag_sections(self, text):
        """Return: list of
        [section, subsectionheader, subsection_line, subsectheader_start, subsectheader_end, subsection_end]
        한 줄에 여러 헤더가 걸리면 정의 순서상 먼저인 것을 쓴다. 구간의 끝은 다음 헤더의 시작이다.
        """
        subsects = self.tag_sectionheaders(text)

        linenum2tuple = {}
        for subsect in subsects:
            linenum2tuple.setdefault(subsect[1], subsect)

        sectionlist = []
        prevsectionheadertuple = linenum2tuple.get(0, (NOSECTIONHEADER, 0, 0, 0))

        lines = text.split('\n')
        offset = len(lines[0]) + 1
        for linenum, line in enumerate(lines[1:], 1):
            if linenum in linenum2tuple:
                if prevsectionheadertuple[0] == 'impression':
                    prevsection = 'assessment_and_plan'
                else:
                    prevsection = subsectionheader2section[prevsectionheadertuple[0]]
                sectionlist.append([prevsection] + list(prevsectionheadertuple) + [offset])
                prevsectionheadertuple = linenum2tuple[linenum]
            offset += len(line) + 1

        if prevsectionheadertuple[0] == 'impression':
            prevsection = 'assessment_and_plan'
        else:
            prevsection = subsectionheader2section[prevsectionheadertuple[0]]
        sectionlist.append([prevsection] + list(prevsectionheadertuple) + [len(text)])
        return sectionlist

    def divide_note_by_metasections(self, text):
        detected_sections = self.tag_sections(text)

        # 헤더 없이 시작하면 subjective로 본다
        if detected_sections and detected_sections[0][0] == NOSECTIONHEADER:
            detected_sections[0][0] = 'subjective'

        # 구역마다 처음 나온 것 하나만 남긴다
        order = ['subjective', 'objective_exam', 'objective_results', 'assessment_and_plan']
        meta_sections = [None] * 4
        for section in detected_sections:
            if section[0] in order:
                i = order.index(section[0])
                if meta_sections[i] is None:
                    meta_sections[i] = section

        meta_sections = sorted((x for x in meta_sections if x is not None), key=lambda x: x[3])

        # 각 구역의 끝을 다음 구역 헤더의 시작으로 맞춘다
        for ind, section in enumerate(meta_sections[1:], start=1):
            meta_sections[ind - 1][-1] = section[-3]
        if meta_sections:
            meta_sections[-1][-1] = len(text)

        return meta_sections
