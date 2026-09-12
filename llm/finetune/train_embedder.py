"""
(anchor, positive, negative) triplet으로 임베딩 모델을 파인튜닝한다.

TripletLoss: anchor-positive 코사인 거리는 줄이고, anchor-negative 코사인 거리는
늘리도록 모델 가중치를 조정한다. 데이터가 적으면 에폭을 많이 돌릴수록 그대로
암기(overfit)해버리므로, 적은 에폭(4)과 낮은 학습률로 살짝만 조정하는 정도로 돌린다.

실행:
    cd project
    python finetune/train_embedder.py            # train.json (병명+카테고리 33건)
    python finetune/train_embedder.py --v2        # train_v2.json (+증상 기반 6건 합친 버전)
"""
import os
import sys
import json

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

BASE_MODEL = "jhgan/ko-sroberta-multitask"
EPOCHS = 4
BATCH_SIZE = 4


def load_train_examples(train_file):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), train_file)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return [InputExample(texts=[r["anchor"], r["positive"], r["negative"]]) for r in rows]


def main():
    use_v2 = "--v2" in sys.argv
    train_file = "train_v2.json" if use_v2 else "train.json"
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ko-sroberta-finetuned-v2" if use_v2 else "ko-sroberta-finetuned",
    )

    examples = load_train_examples(train_file)
    print(f"학습 데이터 {len(examples)}건 로드 ({train_file})")

    model = SentenceTransformer(BASE_MODEL)
    dataloader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)
    train_loss = losses.TripletLoss(model)

    print(f"파인튜닝 시작 (epochs={EPOCHS}, batch_size={BATCH_SIZE})...")
    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=5,
        show_progress_bar=True,
    )

    model.save(out_dir)
    print(f"파인튜닝된 모델을 {out_dir}에 저장했습니다.")


if __name__ == "__main__":
    main()
