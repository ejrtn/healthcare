# 흉부 X-ray 판독 효율화를 위한 AI
- train and test data
    - chest x-ray : https://www.kaggle.com/datasets/ashery/chexpert
    - chest x-ray : https://www.kaggle.com/datasets/nih-chest-xrays/data
    - ct : https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection
    - ct 전처리 : https://www.kaggle.com/datasets/yoodeoksu/rsna-2023-atd-preprocessed-s224

- 코드
- x-ray : denseNet-121.py
    - 파일 이름과 동일하게 denseNet-121 전이학습
    - 2 에포크까지 동결 이후 학습
    - 같은 코드 다른 데이터를 했을 때 결과가 확실히 달라지는 것을 확인
    - 가중치 데이터 개수에 맞춰서 매기고 성능을 올려보기 위해 ai도움을 받아 루트를 했지만 이 또한 결과가 생각보다 원하는 값이 나오지 않아 log를 씌워서 처리
    - 데이터 변경을 통해 성능을 올리고자 시도


- ct : ct-convnext-tiny-s128.ipynb(전처리 포함)
    - convnext_base를 사용하려 했지만 메모리 용량으로 인해 convnext_tiny 변경
    - 캐글에서 output 용량에 한계가 있어 전처리 따로 해서 dataset에 저장 (224, 128 크기)
    - 전처리 과정에서 monai만 사용해서 할 수 있지만 직접 코드 작성을 통해 처리 하는 방식으로 어떻게 성능을 비슷 또는 더 좋은 성능을 하는지 확인
    - 진행중, gpu 부족으로 전처리만 진행
    - 224 크기(대회에서 이렇게 했다고 함)를 진행해서 학습을 하려 했지만 gpu 성능이 딸리고 시간이 30h 제한이 있어 128로 줄임
    