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
    preprocess_interactions,
    primary_skill,
    print_summary,
)


def load_statics2011(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"Anon Student Id": str})
    df = df.dropna(
        subset=[
            "Anon Student Id",
            "Problem Name",
            "Step Name",
            "First Transaction Time",
            "KC (F2011)",
            "First Attempt",
        ]
    ).copy()
    df["question_text"] = df["Problem Name"].astype(str).str.strip() + "::" + df["Step Name"].astype(str).str.strip()
    df["skill_text"] = df["KC (F2011)"].apply(primary_skill)
    df["correct_num"] = (
        df["First Attempt"].astype(str).str.strip().str.lower().isin(["correct", "1", "true"]).astype(int)
    )
    df = df[df["skill_text"] != ""].copy()
    parsed = pd.to_datetime(df["First Transaction Time"], errors="coerce")
    if parsed.notna().any():
        df["_order"] = parsed.astype("int64")
        order_col = "_order"
    else:
        df["_order"] = range(len(df))
        order_col = "_order"
    return normalize_interaction_frame(df, "Anon Student Id", "question_text", "skill_text", "correct_num", order_col)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Statics2011 for graph-enhanced KT training.")
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "Dataset", "statics2011", "AllData_student_step_2011F.csv"),
        help="Path to Statics2011 raw CSV",
    )
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "Dataset", "statics2011")
    summary = preprocess_interactions(
        dataset_name="statics2011",
        normalized_df=load_statics2011(args.input),
        train_pid_path=os.path.join(output_dir, "Statics2011_pid_train.csv"),
        test_pid_path=os.path.join(output_dir, "Statics2011_pid_test.csv"),
        h_path=os.path.join(PROJECT_ROOT, "Dataset", "H", "statics2011.csv"),
        metadata_path=os.path.join(output_dir, "statics2011_meta.json"),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
