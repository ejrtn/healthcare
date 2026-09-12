---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:30
- loss:TripletLoss
base_model: jhgan/ko-sroberta-multitask
widget:
- source_sentence: 중이염은 왜 생기나요?
  sentences:
  - '중이염 원인: 중이염의 발병원인은 다양하며, 바이러스나 세균 감염, 귀인두관(이관)의 기능장애, 알레르기, 그 외 환경적 유전적 요소가 복합적으로
    작용하여 중이염이 발병합니다.'
  - '중이염 생활습관 관리: <식이, 운동, 생활 시 주의사항> 보존적인 치료(수술을 안하는 방법)로 일반적인 건강 상태를 좋게 하고, 저항력을
    높여주기 위해 생활 양식의 개선, 영양식 투여 등에 주의하며, 당뇨병, 간장, 위장 질환, 갑상선 기능저하 등의 만성 질병의 관리에 주의를
    기울여야 합니다. 또한 코(비강 및 부비동)와 목(인두 및 편도)의 질환 등을 치료하여 중이염의 원인을 없애주어야 합니다.'
  - '편두통 약물 치료: 뇌기저형편두통, 허혈뇌졸중과 심혈관질환, 말초혈관질환, 악성고혈압 환자에서는 사용을 피하는 것이 좋습니다. (2) 에르고트
    에르고타민은 복용 후 구역이 심한 단점이 있습니다. 단순 진통제나 카페인, 항구토제 등을 복합한 약물이 많은데 크레밍정과 카펠규정이 시판되고
    있습니다. 에르고트는 구역과 구토를 악화시키고 혈관폐색의 부작용이 많아서 절대 과용하지 않아야 합니다. 또한 혈관질환이나 고혈압, 간질환,
    신장질환이 있는 환자는 사용을 피해야 합니다. (3) 게판트와 디탄 기존에 편두통 특이 약물로 사용하던 트립탄 계열의 혈관 수축 부작용을 피할
    수 있는 약물로 게판트(gepant) 계열의 리메게판트, 우브로게판트, 자베게판트와 디탄(ditan) 계열의 라스미디탄이 출시되었습니다.'
- source_sentence: 대상포진 합병증이 있나요?
  sentences:
  - '고혈압 증상: 고혈압은 심장·뇌·망막·말초혈관 등에서 표적 장기 손상이 발생하기 전까지 대부분 특별한 증상이 없습니다. 일부 환자에서는 두통,
    두근거림, 어지러움, 호흡곤란 같은 비특이적 증상이 나타날 수 있습니다. 이러한 증상만으로 고혈압을 판단할 수 없기 때문에 정기적인 혈압 측정이
    중요합니다.'
  - '대상포진 생활습관 관리: 찜질이 도움이 됩니다. 대상포진의 통증은 필요 시 전문의의 진통제 처방이나 신경차단술 등 시술을 고려합니다. 4)
    열이 날 때는 아세트아미노펜 사용 수두 환아는 아스피린을 피하고 아세트아미노펜을 사용해야 하며, 아스피린은 드물지만 라이 증후군과 연관될 수
    있습니다. 5) 전염 방지를 위한 격리 수두는 모든 발진이 딱지로 아물 때까지 외출을 삼가야 합니다. 대상포진은 수포 부위와의 직접 접촉을
    피해야 합니다. 수두 환아는 학교나 어린이집에 진단 사실을 알리는 것이 중요합니다.'
  - '대상포진 합병증: 1. 수두 수두는 대부분 경미하게 지나가며 자연 치유되지만, 일부에서는 합병증이 발생할 수 있습니다. 가장 흔한 합병증은
    세균에 의한 2차 피부 감염입니다. 드물게 신경계 합병증도 나타날 수 있으며, 무균성 수막염, 뇌염, 소뇌 실조증(cerebellar ataxia)
    등이 있습니다. 뇌염은 10,000명 중 약 1.8명꼴로 드물게 발생하지만, 경련이나 혼수로 이어질 수 있으며 성인에서 소아보다 더 심하게
    나타나는 경향이 있습니다. 임신부가 수두에 걸리면 태아와 신생아에게 영향을 줄 수 있습니다. 임신 20주 이전에 처음 수두에 감염될 경우,
    태아는 선천 수두 증후군 위험이 있으며, 저체중, 사지 기형, 피부 흉터, 근육 위축, 소두증, 뇌피질 위축 등 다양한 이상이 나타날 수 있습니다.'
- source_sentence: 고혈압 생활습관 관리 어떻게 해?
  sentences:
  - '췌장염 증상: 소할 수도 있고 통증이 있다 없다를 반복하기도 합니다. 즉 통증과 통증 사이에 통증이 없는 시기가 있습니다. 대부분 음식을
    먹으면 통증이 악화되기 때문에 체중 감소가 심합니다. 1) 만성 췌장염의 주요 증상 ① 상복부 통증 • 통증의 양상은 미약한 통증에서 매우
    심한 통증, 지속적인 통증 또는 주기적인 통증으로 개인에 따라 차이가 납니다. • 주로 식사를 하면 통증이 악화됩니다. • 급성 췌장염과는
    달리 통증이 오래 지속되는 경우가 많아 진통제 의존성이 생기기도 합니다. • 영상 검사로는 매우 진행된 만성 췌장염이라도 통증이 없을 수 있습니다.
    ② 만성 설사, 지방변, 무기력증: 췌장의 외분비 기능 부전에 의해 나타납니다. ③ 체중 감소: 췌장의 외분비 및 내분비 기능 부전 모두가
    원인입니다.'
  - '고혈압 경과 및 예후: 어질 수 있습니다. 뇌에서는 열공경색, 미세출혈, 뇌혈관 협착 등이 발생하여 뇌졸중 위험이 크게 증가합니다. 콩팥에서는
    콩팥 혈관과 실질이 손상되어 사구체여과율이 감소하고 단백뇨가 나타나며, 만성콩팥병으로 진행할 수 있습니다. 눈에서는 고혈압성 망막병증이 발생하여
    시력 저하나 망막 출혈 등이 나타날 수 있습니다. 반대로 혈압을 잘 관리하면 이러한 장기 손상을 예방하고, 심뇌혈관질환 위험을 크게 줄일 수
    있습니다.'
  - '고혈압 자가 관리: 고혈압은 싱겁게 먹기·규칙적 운동·금연·절주 같은 생활습관을 꾸준히 실천하고 정기적인 혈압 확인을 통해 스스로 건강을
    지킬 수 있는 질환입니다. 약물치료를 받는 환자뿐 아니라, 고혈압 전단계나 주의혈압 단계의 사람에게도 자가 관리는 매우 중요합니다.'
- source_sentence: 갑상선기능항진증 진단은 어떻게 하나요?
  sentences:
  - '갑상선기능항진증 개요: 갑상선호르몬을 분비하는 갑상선은 목 앞부분에 위치하고 있으며, 나비모양을 하고 있습니다. 갑상선호르몬은 우리 몸의
    에너지대사에 관여하여 체온을 유지하고 신진대사를 조절합니다. 갑상선에서 갑상선호르몬이 과잉 생산되어 에너지를 필요 이상으로 만들어내면 몸이
    더워지고 땀이 많이 나며 체중이 줄고, 자율신경기능이 흥분되어 심장박동수가 빨라집니다. 반대로 갑상선호르몬이 너무 적게 나오거나 기능이 비정상적으로
    저하될 경우 동작이 느려지고 추위를 많이 타며 체중이 늘고, 심장박동수도 느려지게 됩니다. 갑상선항진증은 갑상선에서 과잉 생산된 갑상선호르몬이
    혈액 내에서 증가되어 갑상선의 생리적 작용이 과도하게 나타나는 임상증후군입니다. 갑상선항진증의 가장 중요하며 흔한 원인은 자가 면역 질환인
    그레이브스병입니다.'
  - '갑상선기능항진증 진단 및 검사: 1. 진단 신체검사에서 갑상선항진증을 나타내는 전형적인 증상 및 징후를 보이고, 혈액검사에서 갑상선호르몬이
    증가되어 있는 경우 쉽게 갑상선항진증으로 진단할 수 있습니다. 특히, 안구 증상을 보이는 갑상선 안병증, 갑상선질환의 가족력이 있는 경우,
    목이 커져 있는 갑상선종 등이 동반되어 있는 경우 쉽게 진단할 수 있습니다. 그러나, 초기 갑상선항진증으로 갑상선 기능 변화가 크지 않은 경우
    증상 및 징후만으로 알아내기 어렵습니다.갑상선기능검사는 혈액에서 갑상선호르몬의 농도를 측정합니다. 갑상선기능검사에서 갑상선호르몬이 증가된 것이
    확인될 경우 갑상선항진증을 진단할 수 있습니다. 특히, 혈액검사에서 갑상선호르몬 증가와 함께 갑상선을 자극하는 항체 수치가 높을 경우 그레이브스병에
    의한 갑상선항진증으로 진단합니다.'
  - '아토피피부염 합병증: 드물게 피부의 발적, 비늘, 진물, 딱지가 광범위하게 나타나면서 발열이나 림프절이 커지는 증상이 나타납니다. 심한 아토피피부염
    치료를 위해 장기간 사용한 전신 스테로이드를 갑자기 중단했을 때나 피부 감염 시에 발생할 수 있습니다.'
- source_sentence: 대상포진 증상이 뭔가요?
  sentences:
  - '당뇨병 위험요인 및 예방: 인슐린 분비 촉진제를 사용해 저혈당 발생 위험이 높은 당뇨병 환자는 저혈당 예방과 치료법을 숙지해야 합니다. 1.
    당뇨병 환자의 비만 관리 • 비만 관리는 당뇨병 전단계에서 당뇨병으로의 진행을 막을 수 있고, 2형당뇨병 치료에 도움이 됩니다. 과체중이거나
    비만한 2형당뇨병 환자가 체중을 줄이면 혈당이 개선되고, 경구 약제 사용이 줄어들 수 있습니다.• 비만한 당뇨병 환자는 식사요법, 운동요법
    및 행동치료로 치료 전 체중의 5% 이상을 감량하고 유지해야합니다.• 당뇨병 약제를 선택할 때는 약제가 체중에 미치는 영향을 고려합니다. •
    체질량지수 25 kg/m2 이상(1단계 비만)인 제2형 당뇨병환자가 체중감량에 실패한 경우 항비만제를 고려할 수 있습니다.'
  - '대상포진 요약문: ''이것만은 꼭 기억하세요'' • 수두와 대상포진은 같은 바이러스(VZV)에 의해 발생하며, 수두는 처음 감염될 때, 대상포진은
    수두 후 바이러스가 신경절에 숨어 있다가 면역력이 떨어질 때 재활성화되어 나타납니다. • 수두는 전염력이 매우 강하며, 발진이 딱지로 변할
    때까지 전파될 수 있고, 대부분 어린이에서는 경미하게 지나가지만 성인이나 면역저하자는 합병증 위험이 높습니다. • 대상포진은 주로 몸 한쪽
    신경 분포를 따라 발생하며, 통증과 수포가 나타나고, 면역이 약한 경우 전신으로 퍼지거나 심각한 합병증을 일으킬 수 있습니다. • 수두와 대상포진
    모두 항바이러스제를 통해 증상 완화와 합병증 예방이 가능하며, 특히 발진 초기에 치료하는 것이 효과적입니다.'
  - '대상포진 증상: 면역이 결핍된 사람은 대상포진이 피부에 국한되지 않고 전신으로 퍼지거나, 신경계, 폐, 간 등 여러 장기를 침범하는 중증
    형태로 진행될 수 있어 특별한 주의가 필요합니다. 대상포진은 소아에서는 비교적 드물지만, 자궁 내에서 수두 바이러스에 감염된 경우(선천성 수두
    증후군)나 2세 이전에 수두를 앓은 경우 발생할 수 있습니다. 이 경우에는 증상이 비교적 경미한 경향을 보입니다.'
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
    '대상포진 증상이 뭔가요?',
    '대상포진 증상: 면역이 결핍된 사람은 대상포진이 피부에 국한되지 않고 전신으로 퍼지거나, 신경계, 폐, 간 등 여러 장기를 침범하는 중증 형태로 진행될 수 있어 특별한 주의가 필요합니다. 대상포진은 소아에서는 비교적 드물지만, 자궁 내에서 수두 바이러스에 감염된 경우(선천성 수두 증후군)나 2세 이전에 수두를 앓은 경우 발생할 수 있습니다. 이 경우에는 증상이 비교적 경미한 경향을 보입니다.',
    "대상포진 요약문: '이것만은 꼭 기억하세요' • 수두와 대상포진은 같은 바이러스(VZV)에 의해 발생하며, 수두는 처음 감염될 때, 대상포진은 수두 후 바이러스가 신경절에 숨어 있다가 면역력이 떨어질 때 재활성화되어 나타납니다. • 수두는 전염력이 매우 강하며, 발진이 딱지로 변할 때까지 전파될 수 있고, 대부분 어린이에서는 경미하게 지나가지만 성인이나 면역저하자는 합병증 위험이 높습니다. • 대상포진은 주로 몸 한쪽 신경 분포를 따라 발생하며, 통증과 수포가 나타나고, 면역이 약한 경우 전신으로 퍼지거나 심각한 합병증을 일으킬 수 있습니다. • 수두와 대상포진 모두 항바이러스제를 통해 증상 완화와 합병증 예방이 가능하며, 특히 발진 초기에 치료하는 것이 효과적입니다.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.8257, 0.4374],
#         [0.8257, 1.0000, 0.7470],
#         [0.4374, 0.7470, 1.0000]])
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

* Size: 30 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 30 samples:
  |          | sentence_0                                                                      | sentence_1                                                                          | sentence_2                                                                           |
  |:---------|:--------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|
  | type     | string                                                                          | string                                                                              | string                                                                               |
  | modality | text                                                                            | text                                                                                | text                                                                                 |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 9.9 tokens</li><li>max: 14 tokens</li></ul> | <ul><li>min: 43 tokens</li><li>mean: 109.9 tokens</li><li>max: 128 tokens</li></ul> | <ul><li>min: 40 tokens</li><li>mean: 117.23 tokens</li><li>max: 128 tokens</li></ul> |
* Samples:
  | sentence_0                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                   | sentence_2                                                                                                                                                                                                                                                                                                                                                                                                                            |
  |:--------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>편두통 치료법이 뭐야?</code> | <code>편두통 비약물 치료: 고압산소요법, 생체 되먹임, 시각상상요법, 침, 유발점 마사지, 관자동맥 누르기, 경피전기신경자극, 얼음주머니, 이완요법, 명상, 신경차단 등이 있으나 아직 근거가 부족합니다.</code>                                                                                                                                                                                                                                                                                                  | <code>편두통 종류: 조짐이나 실어증 형태의 언어 조짐도 있습니다. 조짐편두통은 일과성허혈발작 등의 질환들과 감별이 필요합니다. 만성편두통은 3개월 이상의 기간 동안 한 달에 15일 이상 두통이 있을 때 진단합니다. 대개 무조짐편두통으로, 편두통 빈도가 증가하면서 만성편두통으로 진행되어 진단되는 경우가 많습니다.</code>                                                                                                                                                                                                                                              |
  | <code>머리가 지끈지끈 아파요</code> | <code>두통 약물 치료: 통의 횟수와 강도를 조절할 수 있으며, 전구기나 전조기까지만 겪고 두통기가 오지 않도록 예방할 수 있습니다. 편두통의 전조증상이 시작되었다면, 우선 아세트아미노펜, 소염진통제가 도움 될 수 있고 여기에 반응이 없다면 트립탄 등의 편두통 치료 약물을 단독 혹은 아세트아미노펜이나 소염진통제와 함께 사용할 수 있습니다. 그러나 트립탄은 혈관 수축 작용이 있어 협심증 등의 혈관 질환이 있는 사람은 사용하지 않아야 하며 그렇지 않더라도 의사와 사용의 필요성, 간격을 잘 상의하여야 합니다. 투약 횟수가 주 3회 이상이 된다면 도리어 약물 과용 두통으로 옮겨갈 수 있어, 예방 약물을 규칙적으로 먹는 예방요법으로 바꾸는 게 나은데, 이때 사용되는 예방 약물로는 베타차단제, 토피라메이트가 있습니다. 3.</code> | <code>식이영양(소아/청소년) 종류: 저장량과 필요량이 크게 증가해 철 결핍이 가장 흔히 생기는 시기입니다. 이 시기에는 흡수율 또한 증가합니다. ③ 비타민 D 급속한 성장과 함께 골격도 커지므로 비타민 D를 적절히 섭취하는 것이 중요합니다. 비타민 D는 햇빛을 충분히 쬐면 피부에서 합성되지만, 야외 활동이 부족하면 햇빛에 노출되는 시간이 충분하지 않으므로 음식을 통해 섭취하도록 합니다. 2. 청소년기에 필요한 영양소 1) 에너지와 다량 영양소 ① 에너지 에너지 섭취 기준은 기초대사량과 활동 수준, 사춘기 성장과 발달에 필요한 에너지를 고려해 설정합니다. 청소년은 일과의 대부분이 공부와 관련되어 활동량이 낮다고 가정합니다. 아주 활발한 청소년은 활동량을 고려해야 합니다. ② 단백질 청소년기의 급격한 성장에 필수적인 영양소입니다.</code> |
  | <code>중이염은 왜 생기나요?</code> | <code>중이염 원인: 중이염의 발병원인은 다양하며, 바이러스나 세균 감염, 귀인두관(이관)의 기능장애, 알레르기, 그 외 환경적 유전적 요소가 복합적으로 작용하여 중이염이 발병합니다.</code>                                                                                                                                                                                                                                                                                                              | <code>중이염 생활습관 관리: <식이, 운동, 생활 시 주의사항> 보존적인 치료(수술을 안하는 방법)로 일반적인 건강 상태를 좋게 하고, 저항력을 높여주기 위해 생활 양식의 개선, 영양식 투여 등에 주의하며, 당뇨병, 간장, 위장 질환, 갑상선 기능저하 등의 만성 질병의 관리에 주의를 기울여야 합니다. 또한 코(비강 및 부비동)와 목(인두 및 편도)의 질환 등을 치료하여 중이염의 원인을 없애주어야 합니다.</code>                                                                                                                                                                                         |
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
- **Training**: 1.7 minutes

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