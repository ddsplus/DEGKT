#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.preprocess_common import (  # noqa: E402
    normalize_interaction_frame,
    pick_first_existing,
    preprocess_interactions,
    primary_skill,
    print_summary,
)


def default_input_path() -> str:
    candidates = [
        os.path.join(PROJECT_ROOT, "Dataset", "assist2017", "anonymized_full_release_competition_dataset.csv"),
        os.path.join(PROJECT_ROOT, "Data", "ASSIST2017", "anonymized_full_release_competition_dataset.csv"),
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "Data", "ASSIST2017", "anonymized_full_release_competition_dataset.csv")),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def load_assist2017(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    user_col = pick_first_existing(df, ["studentId", "student_id", "user_id"], "user")
    item_col = pick_first_existing(df, ["problemId", "problem_id"], "problem")
    skill_col = pick_first_existing(df, ["skill", "skill_id", "skillId", "KC"], "skill")
    correct_col = pick_first_existing(df, ["correct", "isCorrect"], "correct")

    order_candidates = [
        "startTime",
        "start_time",
        "timestamp",
        "ms_first_response",
        "problemLogId",
        "attemptCount",
    ]
    order_col = next((name for name in order_candidates if name in df.columns), None)
    if order_col is None:
        df = df.copy()
        df["_row_order"] = range(len(df))
        order_col = "_row_order"

    df = df.dropna(subset=[user_col, item_col, skill_col, correct_col]).copy()
    df[skill_col] = df[skill_col].apply(primary_skill)
    df = df[df[skill_col] != ""].copy()

    if order_col in ("startTime", "start_time", "timestamp"):
        parsed = pd.to_datetime(df[order_col], errors="coerce")
        if parsed.notna().any():
            df["_parsed_order"] = parsed.astype("int64")
            order_col = "_parsed_order"

    return normalize_interaction_frame(df, user_col, item_col, skill_col, correct_col, order_col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ASSIST2017 for graph-enhanced KT training.")
    parser.add_argument("--input", default=default_input_path(), help="Path to ASSIST2017 raw CSV")
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "Dataset", "assist2017")
    summary = preprocess_interactions(
        dataset_name="assist2017",
        normalized_df=load_assist2017(args.input),
        train_pid_path=os.path.join(output_dir, "assist2017_pid_train.csv"),
        test_pid_path=os.path.join(output_dir, "assist2017_pid_test.csv"),
        h_path=os.path.join(PROJECT_ROOT, "Dataset", "H", "2017.csv"),
        metadata_path=os.path.join(output_dir, "assist2017_meta.json"),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
