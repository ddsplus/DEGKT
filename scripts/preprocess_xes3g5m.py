#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from collections import defaultdict

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocess_common import preprocess_interactions, print_summary  # noqa: E402


def parse_list(value) -> list:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_xes(train_path: str, test_path: str) -> pd.DataFrame:
    rows = []
    user_offsets = defaultdict(int)
    for path in (train_path, test_path):
        raw = pd.read_csv(path, dtype={"uid": str})
        for row in raw.itertuples(index=False):
            user_id = str(getattr(row, "uid")).strip()
            questions = parse_list(getattr(row, "questions", ""))
            concepts = parse_list(getattr(row, "concepts", ""))
            responses = parse_list(getattr(row, "responses", ""))
            seq_len = min(len(questions), len(concepts), len(responses))
            base_offset = user_offsets[user_id]
            for idx in range(seq_len):
                qid = int(questions[idx])
                sid = int(concepts[idx])
                ans = int(responses[idx])
                if qid <= 0 or sid <= 0 or ans not in (0, 1):
                    continue
                rows.append(
                    {
                        "user_id": user_id,
                        "question_id": str(qid),
                        "skill_id": str(sid),
                        "correct": ans,
                        "order_id": base_offset + idx,
                    }
                )
            user_offsets[user_id] += seq_len
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess XES3G5M for graph-enhanced KT training.")
    parser.add_argument("--train-input", default=os.path.join(PROJECT_ROOT, "Dataset", "xes3g5m", "train.csv"))
    parser.add_argument("--test-input", default=os.path.join(PROJECT_ROOT, "Dataset", "xes3g5m", "test.csv"))
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "Dataset", "xes3g5m")
    summary = preprocess_interactions(
        dataset_name="xes3g5m",
        normalized_df=load_xes(args.train_input, args.test_input),
        train_pid_path=os.path.join(output_dir, "xes3g5m_pid_train.csv"),
        test_pid_path=os.path.join(output_dir, "xes3g5m_pid_test.csv"),
        h_path=os.path.join(PROJECT_ROOT, "Dataset", "H", "xes3g5m.csv"),
        metadata_path=os.path.join(output_dir, "xes3g5m_meta.json"),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
