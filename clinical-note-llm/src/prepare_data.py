"""원본 CSV → data/processed/*.jsonl.

산출물
  aci_{train,valid,test1,test2,test3}.jsonl  ACI-BENCH 공식 분할 (MEDIQA-CHAT/SUM 2023)
  aci_asr_test.jsonl / aci_asrcorr_test.jsonl 같은 대화의 ASR 원본 / 교정본 (test1~3의 aci 부분, 66쌍)
  mts_{train,valid}.jsonl                    MTS-Dialog (짧은 대화 → 기록 섹션 하나)

ACI 레코드: {id, dataset, dialogue, note, target}  — target은 note를 4구역으로 나눈 JSON
"""
import csv
import json
from pathlib import Path

from schema import DIVISIONS, render_note, split_note

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

ACI_CHALLENGE = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test1": "clinicalnlp_taskB_test1.csv",
    "test2": "clinicalnlp_taskC_test2.csv",
    "test3": "clef_taskC_test3.csv",
}


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(name, records):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(records)}")


def aci_record(row, id_key):
    note = row["note"].strip()
    target = split_note(note)
    # 4구역으로 나눴다가 다시 합쳐도 같은 구역이 나와야 채점과 학습 정답이 일치한다
    assert split_note(render_note(target)) == target, row[id_key]
    return {
        "id": row[id_key],
        "dataset": row["dataset"],
        "dialogue": row["dialogue"].strip(),
        "note": note,
        "target": target,
    }


def main():
    for split, fname in ACI_CHALLENGE.items():
        rows = read_csv(RAW / "aci-bench" / "data" / "challenge_data" / fname)
        write_jsonl(f"aci_{split}.jsonl", [aci_record(r, "encounter_id") for r in rows])

    src = RAW / "aci-bench" / "data" / "src_experiment_data"
    for variant in ["asr", "asrcorr"]:
        records = []
        for t in ["test1", "test2", "test3"]:
            records += [aci_record(r, "id") for r in read_csv(src / f"{t}_aci_{variant}.csv")]
        write_jsonl(f"aci_{variant}_test.jsonl", records)

    mts = RAW / "MTS-Dialog" / "Main-Dataset"
    for split, fname in [("train", "MTS-Dialog-TrainingSet.csv"), ("valid", "MTS-Dialog-ValidationSet.csv")]:
        write_jsonl(f"mts_{split}.jsonl", [
            {"id": r["ID"], "section_header": r["section_header"],
             "dialogue": r["dialogue"].strip(), "section_text": r["section_text"].strip()}
            for r in read_csv(mts / fname)
        ])

    empty = {d: 0 for d in DIVISIONS}
    for line in open(OUT / "aci_train.jsonl", encoding="utf-8"):
        for d, text in json.loads(line)["target"].items():
            empty[d] += not text
    print("aci_train 빈 구역 수:", empty)


if __name__ == "__main__":
    main()
