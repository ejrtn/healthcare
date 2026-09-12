---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:25
- loss:TripletLoss
base_model: jhgan/ko-sroberta-multitask
widget:
- source_sentence: 편두통약은 어떤 게 있나요?
  sentences:
  - '편두통 진단 및 검사: 짐편두통으로 진단하고, 한 달에 15일 이상 지속되는 두통이 3개월 이상 계속되면 만성편두통으로 진단합니다. 뇌출혈
    등의 뇌 질환과 편두통을 구분하기 위해 필요에 따라 뇌 MRI(또는 CT), MRA(또는 CTA), MR 정맥조영 검사를 시행합니다. 뇌출혈
    및 뇌수막염과 구분하기 위해 뇌척수액검사가 필요한 경우도 있습니다. 뇌전증으로 인한 두통을 의심한다면 뇌파검사를 시행하며, 뇌혈류검사는 가역뇌혈관수축증후군
    환자에서 추적검사로 이용합니다.'
  - '감기 위험요인 및 예방: 감기를 일으키는 바이러스는 다양하므로 이를 예방하는 백신의 개발에는 한계가 있어 간염 백신이나 독감 백신과 달리
    아직까지 감기를 예방하는 백신은 없습니다. 비타민 C를 매일 복용하는 것은 정상 성인에서는 감기를 예방하는 효과가 없습니다. 하지만 군인이나
    마라톤 선수처럼 힘든 훈련을 하는 사람이 비타민 C를 복용하는 경우에는 감기 발생이 50%정도 감소되었으므로 단기간 육체적 훈련을 하거나 추운
    환경에 노출되는 사람에게는 비타민 C 보충이 감기 예방에 도움이 될 수 있습니다. 규칙적이고 중등도 강도의 운동이 감기 예방에 효과가 있었다는
    연구가 있습니다. 규칙적인 유산균제(probiotics) 복용이 도움이 된다는 연구도 있지만 효과는 미미합니다.'
  - '편두통 약물 치료: 뇌기저형편두통, 허혈뇌졸중과 심혈관질환, 말초혈관질환, 악성고혈압 환자에서는 사용을 피하는 것이 좋습니다. (2) 에르고트
    에르고타민은 복용 후 구역이 심한 단점이 있습니다. 단순 진통제나 카페인, 항구토제 등을 복합한 약물이 많은데 크레밍정과 카펠규정이 시판되고
    있습니다. 에르고트는 구역과 구토를 악화시키고 혈관폐색의 부작용이 많아서 절대 과용하지 않아야 합니다. 또한 혈관질환이나 고혈압, 간질환,
    신장질환이 있는 환자는 사용을 피해야 합니다. (3) 게판트와 디탄 기존에 편두통 특이 약물로 사용하던 트립탄 계열의 혈관 수축 부작용을 피할
    수 있는 약물로 게판트(gepant) 계열의 리메게판트, 우브로게판트, 자베게판트와 디탄(ditan) 계열의 라스미디탄이 출시되었습니다.'
- source_sentence: 고혈압 증상 알려줘
  sentences:
  - '고혈압 증상: 고혈압은 심장·뇌·망막·말초혈관 등에서 표적 장기 손상이 발생하기 전까지 대부분 특별한 증상이 없습니다. 일부 환자에서는 두통,
    두근거림, 어지러움, 호흡곤란 같은 비특이적 증상이 나타날 수 있습니다. 이러한 증상만으로 고혈압을 판단할 수 없기 때문에 정기적인 혈압 측정이
    중요합니다.'
  - '고혈압 원인: 대부분의 고혈압은 여러 유전적 요인과 생활습관이 함께 작용해 발생하는 본태성(일차성) 고혈압으로, 가족력이 흔하고 다인성 유전
    형태를 보입니다. 환경적 요인으로는 과도한 음주, 신체활동 부족, 체중 증가, 고염식 등이 고혈압 발생 위험을 높이며, 대시(DASH) 식단과
    같은 건강한 식습관은 위험을 낮춥니다. 흡연과 스트레스는 일시적으로 혈압을 올릴 수 있으나, 만성 고혈압의 직접 원인이라는 근거는 제한적입니다.
    한편, 이차성 고혈압의 주요 원인으로는 콩팥실질질환, 콩팥동맥협착, 원발성 알도스테론증, 크롬친화세포종, 갑상선 기능 이상, 쿠싱증후군, 수면무호흡증
    등이 있습니다.'
  - '간흡충증 약물 치료: 체중 kg당 25 mg의 프라지콴텔(Praziquantel)을 하루에 3번, 하루 또는 이틀간 복용합니다. 하루 복용으로
    치료가 안 될 경우 같은 용법으로 반복 투여합니다. 많은 수의 주민들을 대상으로 하는 집단치료일 경우에는 체중 kg당 40 mg을 1회 투여하는
    용법이 권장됩니다. 약제 복용 시 메스꺼움, 설사와 같은 소화기장애, 어지럼증, 두통 등의 부작용이 나타날 수 있기 때문에 취침 전에 투약하는
    것이 좋습니다. 임산부에게는 투약하지 않아야 합니다.'
- source_sentence: 아토피피부염 관리는 어떻게 하나요?
  sentences:
  - '아토피피부염 생활습관 관리: 톡톡 두드려 닦은 후 물기가 마르기 전 3분 이내에 보습제를 바릅니다. 2. 의복 의복은 모직이나 나일론보다
    부드러운 질감의 면이나 견이 좋습니다. 새 옷은 세탁한 뒤에 입고, 세탁 시에는 세제가 남지 않도록 물로 여러 번 헹구는 것이 좋습니다. 3.
    손톱 긁으면 더 가려워지므로 손톱을 짧게 유지합니다. 유아는 손에 장갑을 끼워 재우는 것도 도움이 됩니다. 4. 운동 스트레스 해소를 위한
    가벼운 운동은 좋으나, 지나친 발열이나 발한을 일으키는 격렬한 운동은 피하는 것이 좋습니다. 운동 후 땀은 바로 닦아줍니다. 5. 정서적 안정
    아토피피부염 환자는 피부 상태로 인해 분노, 불안, 좌절감 등을 느낄 수 있습니다.'
  - '아토피피부염 약물 치료: 1. 국소 치료 1) 국소 스테로이드 국소 스테로이드는 아토피피부염의 기본 치료제로 많은 경우 효과적입니다. 국소
    스테로이드는 피부 면역 세포에 작용해 이상 면역 반응을 억제합니다. 또한 가려움증을 줄이고, 염증을 가라앉혀 아토피피부염 악화와 관련이 있는
    황색포도알균 수를 감소시키는 효과가 있습니다. 국소 스테로이드는 강도에 따라 가장 강한 1등급에서 제일 약한 7등급으로 구분되며, 작용 원리에
    따라 연고, 크림, 로션, 용액, 겔 등 다양한 제제가 개발되어 있습니다. 치료 부위나 상태에 따라 약물의 강도, 제제를 달리해 적절히 사용해야
    효과적인 치료가 가능합니다. 피부가 얇아 흡수가 잘 되는 부위에는 약한 연고를 사용하고, 두꺼운 부위에는 강도를 높여 사용합니다.'
  - '당뇨병 위험요인 및 예방: 인슐린 분비 촉진제를 사용해 저혈당 발생 위험이 높은 당뇨병 환자는 저혈당 예방과 치료법을 숙지해야 합니다. 1.
    당뇨병 환자의 비만 관리 • 비만 관리는 당뇨병 전단계에서 당뇨병으로의 진행을 막을 수 있고, 2형당뇨병 치료에 도움이 됩니다. 과체중이거나
    비만한 2형당뇨병 환자가 체중을 줄이면 혈당이 개선되고, 경구 약제 사용이 줄어들 수 있습니다.• 비만한 당뇨병 환자는 식사요법, 운동요법
    및 행동치료로 치료 전 체중의 5% 이상을 감량하고 유지해야합니다.• 당뇨병 약제를 선택할 때는 약제가 체중에 미치는 영향을 고려합니다. •
    체질량지수 25 kg/m2 이상(1단계 비만)인 제2형 당뇨병환자가 체중감량에 실패한 경우 항비만제를 고려할 수 있습니다.'
- source_sentence: 골다공증 원인이 뭐야?
  sentences:
  - '골다공증 자가 진단: 골다공증의 위험 요인을 가진 사람이 갑자기 등 쪽에 통증이 발생하거나 키가 줄어든다면, 골다공증과 척추 골절의 가능성을
    고려해야 합니다.'
  - '중이염 증상: 1. 귀통증 중이강 내의 삼출액이 고막을 밀어 팽창하면 귀에 통증이 발생하게 됩니다. 소아는 귀의 통증을 직접 호소할 수도
    있으나, 영아는 귀를 잡아당기거나 단순히 보채고 평소보다 많이 울 수도 있고, 눕거나, 씹거나, 빨 때 귀의 통증이 심해질 수 있기 때문에
    잘 먹지 않거나 자지 않을 수 있습니다. 2. 귓물(이루) 삼출액은 점차 고름으로 바뀌고 압력도 높아져서 이에 의한 압력이 어느 수준을 넘으면,
    고막을 터뜨리고 외이도로 흘러나오게 됩니다. 일단 고막에 구멍이 생겨 고름이 흘러나오면 고막에 대한 압력이 소실되어 통증은 사라지게 됩니다.
    3. 난청 중이강 내에 고인 삼출액은 소리의 전달을 방해하므로 소아는 일시적인 난청이 생기게 됩니다.'
  - '골다공증 원인: 질이 뼈의 양을 감소시키고, 조기 폐경과 여성 호르몬의 감소를 일으켜 골다공증을 발생시킬 수 있습니다. 8. 과음 과다한
    음주는 뼈의 형성을 줄이며 칼슘 흡수도 떨어뜨립니다.'
- source_sentence: 당뇨병 무슨 약 먹어?
  sentences:
  - '당뇨병 치료: 몸에서 포도당이 빠져나가므로 체중이 줄어들 뿐만 아니라, 이뇨 작용에 의해 혈압을 낮추는 효과도 있습니다. 최근 임상 연구에서
    동맥경화 심장질환이 있는 환자나 단백뇨가 있는 당뇨병 환자에서 SGLT2 억제제가 질환의 진행을 예방함이 증명되었습니다. 따라서 이런 질환에서는
    SGLT2 억제제를 우선 고려합니다. 그러나 SGLT2 억제제는 요로감염이나 케톤산증, 탈수 등의 위험이 있어 주의해야 합니다.③ 다이펩타이드(펩타이드)
    분해효소-4(dipeptidyl peptidase-4, DPP-4) 억제제는 우리 몸에서 분비되는 인크레틴 호르몬의 분해를 억제해 인슐린 분비를
    증가시키는 한편, 혈당을 높이는 호르몬인 글루카곤 분비를 억제해 혈당을 낮춥니다.'
  - '당뇨병 요약문: ''이것만은 꼭 기억하세요'' • 당뇨병은 혈액 속 포도당이 세포로 들어가지 못해 혈당이 높아지는 질환으로, 1형, 2형,
    기타, 임신당뇨병으로 나눠집니다. • 당뇨병은 혈당만 상승시키는 것이 아니라, 잘 관리하지 않으면 합병증을 초래할 수 있습니다. • 당뇨병의
    주요 증상은 다음, 다식, 다뇨이며, 증상이 없을 수도 있어 정기적인 건강검진이 중요합니다. • 합병증으로는 망막병증, 신경병증, 신장병증
    등 다양한 문제가 발생할 수 있습니다. • 비만한 당뇨병 환자는 체중을 5% 이상 줄이고, 혈압, 이상지질혈증 및 심혈관질환 관리, 금연,
    저혈당 예방에 주의해야 하며, 식사요법과 운동요법으로 혈당과 건강을 적극적으로 관리해야 합니다.'
  - '고혈압 원인: 대부분의 고혈압은 여러 유전적 요인과 생활습관이 함께 작용해 발생하는 본태성(일차성) 고혈압으로, 가족력이 흔하고 다인성 유전
    형태를 보입니다. 환경적 요인으로는 과도한 음주, 신체활동 부족, 체중 증가, 고염식 등이 고혈압 발생 위험을 높이며, 대시(DASH) 식단과
    같은 건강한 식습관은 위험을 낮춥니다. 흡연과 스트레스는 일시적으로 혈압을 올릴 수 있으나, 만성 고혈압의 직접 원인이라는 근거는 제한적입니다.
    한편, 이차성 고혈압의 주요 원인으로는 콩팥실질질환, 콩팥동맥협착, 원발성 알도스테론증, 크롬친화세포종, 갑상선 기능 이상, 쿠싱증후군, 수면무호흡증
    등이 있습니다.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on jhgan/ko-sroberta-multitask

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for retrieval.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [jhgan/ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask) <!-- at revision 8fca7c9c98c26599be0e14b9916b11a756a26f19 -->
- **Maximum Sequence Length:** 128 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'RobertaModel'})
  (1): Pooling({'embedding_dimension': 768, 'pooling_mode': 'mean', 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '당뇨병 무슨 약 먹어?',
    '당뇨병 치료: 몸에서 포도당이 빠져나가므로 체중이 줄어들 뿐만 아니라, 이뇨 작용에 의해 혈압을 낮추는 효과도 있습니다. 최근 임상 연구에서 동맥경화 심장질환이 있는 환자나 단백뇨가 있는 당뇨병 환자에서 SGLT2 억제제가 질환의 진행을 예방함이 증명되었습니다. 따라서 이런 질환에서는 SGLT2 억제제를 우선 고려합니다. 그러나 SGLT2 억제제는 요로감염이나 케톤산증, 탈수 등의 위험이 있어 주의해야 합니다.③ 다이펩타이드(펩타이드) 분해효소-4(dipeptidyl peptidase-4, DPP-4) 억제제는 우리 몸에서 분비되는 인크레틴 호르몬의 분해를 억제해 인슐린 분비를 증가시키는 한편, 혈당을 높이는 호르몬인 글루카곤 분비를 억제해 혈당을 낮춥니다.',
    "당뇨병 요약문: '이것만은 꼭 기억하세요' • 당뇨병은 혈액 속 포도당이 세포로 들어가지 못해 혈당이 높아지는 질환으로, 1형, 2형, 기타, 임신당뇨병으로 나눠집니다. • 당뇨병은 혈당만 상승시키는 것이 아니라, 잘 관리하지 않으면 합병증을 초래할 수 있습니다. • 당뇨병의 주요 증상은 다음, 다식, 다뇨이며, 증상이 없을 수도 있어 정기적인 건강검진이 중요합니다. • 합병증으로는 망막병증, 신경병증, 신장병증 등 다양한 문제가 발생할 수 있습니다. • 비만한 당뇨병 환자는 체중을 5% 이상 줄이고, 혈압, 이상지질혈증 및 심혈관질환 관리, 금연, 저혈당 예방에 주의해야 하며, 식사요법과 운동요법으로 혈당과 건강을 적극적으로 관리해야 합니다.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7955, 0.2284],
#         [0.7955, 1.0000, 0.3738],
#         [0.2284, 0.3738, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 25 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 25 samples:
  |          | sentence_0                                                                      | sentence_1                                                                           | sentence_2                                                                           |
  |:---------|:--------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                          | string                                                                               | string                                                                               |
  | modality | text                                                                            | text                                                                                 | text                                                                                 |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 9.6 tokens</li><li>max: 13 tokens</li></ul> | <ul><li>min: 43 tokens</li><li>mean: 112.04 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 40 tokens</li><li>mean: 117.12 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                        | sentence_2                                                                                                                                                                                                                                                                                                                                                                                |
  |:--------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>고혈압 원인이 뭐야?</code>  | <code>고혈압 원인: 대부분의 고혈압은 여러 유전적 요인과 생활습관이 함께 작용해 발생하는 본태성(일차성) 고혈압으로, 가족력이 흔하고 다인성 유전 형태를 보입니다. 환경적 요인으로는 과도한 음주, 신체활동 부족, 체중 증가, 고염식 등이 고혈압 발생 위험을 높이며, 대시(DASH) 식단과 같은 건강한 식습관은 위험을 낮춥니다. 흡연과 스트레스는 일시적으로 혈압을 올릴 수 있으나, 만성 고혈압의 직접 원인이라는 근거는 제한적입니다. 한편, 이차성 고혈압의 주요 원인으로는 콩팥실질질환, 콩팥동맥협착, 원발성 알도스테론증, 크롬친화세포종, 갑상선 기능 이상, 쿠싱증후군, 수면무호흡증 등이 있습니다.</code>                                         | <code>고혈압 치료: 치료 목표 고혈압 치료의 가장 중요한 목표는 심장·뇌·신장 등 표적 장기 손상을 예방하고, 심뇌혈관질환 발생을 줄여 건강한 삶을 유지하는 것입니다. 이를 위해 임상 진료지침에서는 연령과 동반질환에 따라 적정 혈압 목표치를 설정하고 있습니다. 표적 장기 손상이 없거나 알부민뇨가 없는 만성콩팥병 환자, 그리고 노인 고혈압 환자의 치료 목표는 수축기혈압 140 mmHg 미만, 이완기혈압 90 mmHg 미만입니다. 반면 고위험 당뇨병 환자를 포함하여 표적 장기 손상이 있는 환자에서는 보다 엄격한 조절이 필요하며, 수축기혈압 130 mmHg 미만, 이완기혈압 80 mmHg 미만을 목표로 합니다.</code>                          |
  | <code>중이염 증상이 뭔가요?</code> | <code>중이염 증상: 1. 귀통증 중이강 내의 삼출액이 고막을 밀어 팽창하면 귀에 통증이 발생하게 됩니다. 소아는 귀의 통증을 직접 호소할 수도 있으나, 영아는 귀를 잡아당기거나 단순히 보채고 평소보다 많이 울 수도 있고, 눕거나, 씹거나, 빨 때 귀의 통증이 심해질 수 있기 때문에 잘 먹지 않거나 자지 않을 수 있습니다. 2. 귓물(이루) 삼출액은 점차 고름으로 바뀌고 압력도 높아져서 이에 의한 압력이 어느 수준을 넘으면, 고막을 터뜨리고 외이도로 흘러나오게 됩니다. 일단 고막에 구멍이 생겨 고름이 흘러나오면 고막에 대한 압력이 소실되어 통증은 사라지게 됩니다. 3. 난청 중이강 내에 고인 삼출액은 소리의 전달을 방해하므로 소아는 일시적인 난청이 생기게 됩니다.</code>  | <code>중이염 진단 및 검사: 서 약간의 압력의 변화나 몇 번의 짧은 소리를 들을 수 있을 뿐 다른 어려움은 없는 간단한 검사입니다. 3. 청력검사 청력검사는 검사자가 피검사자에게 특정 주파수의 소리를 들려주면, 피검자가 소리를 들었는지 버튼을 누르는 방식으로 청력을 측정하는 방법입니다. 따라서 말을 잘 이해하지 못하거나 집중하기 어려운 유소아의 경우는 전문적인 청력 검사자가 측정하여야만 결과를 신뢰할 수 있습니다. 항상 청력검사가 필요한 것은 아니지만 중이강에 지속적으로 삼출액이 고여 있거나 소아가 난청의 증상을 보이면 청력검사를 시행하여 환자의 청력상태를 파악하는 것이 좋습니다.</code>                                        |
  | <code>당뇨병 무슨 약 먹어?</code> | <code>당뇨병 치료: 몸에서 포도당이 빠져나가므로 체중이 줄어들 뿐만 아니라, 이뇨 작용에 의해 혈압을 낮추는 효과도 있습니다. 최근 임상 연구에서 동맥경화 심장질환이 있는 환자나 단백뇨가 있는 당뇨병 환자에서 SGLT2 억제제가 질환의 진행을 예방함이 증명되었습니다. 따라서 이런 질환에서는 SGLT2 억제제를 우선 고려합니다. 그러나 SGLT2 억제제는 요로감염이나 케톤산증, 탈수 등의 위험이 있어 주의해야 합니다.③ 다이펩타이드(펩타이드) 분해효소-4(dipeptidyl peptidase-4, DPP-4) 억제제는 우리 몸에서 분비되는 인크레틴 호르몬의 분해를 억제해 인슐린 분비를 증가시키는 한편, 혈당을 높이는 호르몬인 글루카곤 분비를 억제해 혈당을 낮춥니다.</code> | <code>당뇨병 요약문: '이것만은 꼭 기억하세요' • 당뇨병은 혈액 속 포도당이 세포로 들어가지 못해 혈당이 높아지는 질환으로, 1형, 2형, 기타, 임신당뇨병으로 나눠집니다. • 당뇨병은 혈당만 상승시키는 것이 아니라, 잘 관리하지 않으면 합병증을 초래할 수 있습니다. • 당뇨병의 주요 증상은 다음, 다식, 다뇨이며, 증상이 없을 수도 있어 정기적인 건강검진이 중요합니다. • 합병증으로는 망막병증, 신경병증, 신장병증 등 다양한 문제가 발생할 수 있습니다. • 비만한 당뇨병 환자는 체중을 5% 이상 줄이고, 혈압, 이상지질혈증 및 심혈관질환 관리, 금연, 저혈당 예방에 주의해야 하며, 식사요법과 운동요법으로 혈당과 건강을 적극적으로 관리해야 합니다.</code> |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 4
- `per_device_eval_batch_size`: 4
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 4
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 4
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Time
- **Training**: 1.4 minutes

### Framework Versions
- Python: 3.12.10
- Sentence Transformers: 5.6.0
- Transformers: 5.13.1
- PyTorch: 2.13.0+cpu
- Accelerate: 1.14.0
- Datasets: 5.0.1
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->