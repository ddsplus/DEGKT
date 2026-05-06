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


def load_assist2009(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", low_memory=False)
    skill_col = "skill_id" if "skill_id" in df.columns else "skill_name"
    df = df.dropna(subset=["user_id", "order_id", "problem_id", skill_col, "correct"]).copy()
    df[skill_col] = df[skill_col].apply(primary_skill)
    df = df[df[skill_col] != ""].copy()
    return normalize_interaction_frame(df, "user_id", "problem_id", skill_col, "correct", "order_id")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess ASSIST2009 for graph-enhanced KT training.")
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "Dataset", "assist2009", "skill_builder_data.csv"),
        help="Path to ASSIST2009 raw CSV",
    )
    args = parser.parse_args()

    output_dir = os.path.join(PROJECT_ROOT, "Dataset", "assist2009")
    summary = preprocess_interactions(
        dataset_name="assist2009",
        normalized_df=load_assist2009(args.input),
        train_pid_path=os.path.join(output_dir, "assist2009_pid_train.csv"),
        test_pid_path=os.path.join(output_dir, "assist2009_pid_test.csv"),
        h_path=os.path.join(PROJECT_ROOT, "Dataset", "H", "2009.csv"),
        metadata_path=os.path.join(output_dir, "assist2009_meta.json"),
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
