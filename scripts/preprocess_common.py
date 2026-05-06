#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


MAX_STEP = 50
MIN_SEQUENCE_LENGTH = 3
TRAIN_RATIO = 0.8
RANDOM_STATE = 42


@dataclass
class PreprocessSummary:
    dataset: str
    train_users: int
    test_users: int
    num_questions: int
    num_skills: int
    train_interactions: int
    test_interactions: int
    train_sequences: int
    test_sequences: int
    dropped_train_interactions: int
    dropped_test_interactions: int
    train_pid_path: str
    test_pid_path: str
    h_path: str
    metadata_path: str


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, ".."))


def ensure_binary(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().all():
        raise ValueError("Could not parse correctness column as numeric values.")
    values = values.fillna(0).astype(int)
    values = values.clip(lower=0, upper=1)
    return values


def primary_skill(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text == ".":
        return ""
    for sep in ("~~", ";", "|", ","):
        if sep in text:
            parts = [part.strip() for part in text.split(sep) if part.strip() and part.strip() != "."]
            if parts:
                return parts[0]
    return text


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(f"Missing {label} column. Tried: {list(candidates)}")


def split_users(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    users = sorted(df["user_id"].astype(str).unique().tolist())
    if len(users) < 2:
        raise ValueError("Need at least two users to create an 80/20 split.")
    train_users, test_users = train_test_split(
        users,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    return sorted(train_users), sorted(test_users)


def normalize_interaction_frame(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    skill_col: str,
    correct_col: str,
    order_col: str,
) -> pd.DataFrame:
    frame = df[[user_col, item_col, skill_col, correct_col, order_col]].copy()
    frame.columns = ["user_id", "question_id", "skill_id", "correct", "order_id"]
    frame["user_id"] = frame["user_id"].astype(str).str.strip()
    frame["question_id"] = frame["question_id"].astype(str).str.strip()
    frame["skill_id"] = frame["skill_id"].astype(str).str.strip()
    frame["correct"] = ensure_binary(frame["correct"])
    frame = frame[
        (frame["user_id"] != "")
        & (frame["question_id"] != "")
        & (frame["skill_id"] != "")
    ].copy()
    frame = frame.sort_values(["user_id", "order_id"], kind="mergesort").reset_index(drop=True)
    return frame


def fit_mappings(train_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    questions = sorted(train_df["question_id"].unique().tolist())
    skills = sorted(train_df["skill_id"].unique().tolist())
    q_map = {qid: idx + 1 for idx, qid in enumerate(questions)}
    s_map = {sid: idx + 1 for idx, sid in enumerate(skills)}
    return q_map, s_map


def map_interactions(df: pd.DataFrame, q_map: Dict[str, int], s_map: Dict[str, int]) -> Tuple[pd.DataFrame, int]:
    mapped = df.copy()
    mapped["question_id"] = mapped["question_id"].map(q_map)
    mapped["skill_id"] = mapped["skill_id"].map(s_map)
    dropped = int((mapped["question_id"].isna() | mapped["skill_id"].isna()).sum())
    mapped = mapped.dropna(subset=["question_id", "skill_id"]).copy()
    mapped["question_id"] = mapped["question_id"].astype(int)
    mapped["skill_id"] = mapped["skill_id"].astype(int)
    mapped["correct"] = mapped["correct"].astype(int)
    return mapped, dropped


def _pad(values: List[int], target_len: int, pad_value: int = -1) -> List[int]:
    if len(values) >= target_len:
        return values[:target_len]
    return values + [pad_value] * (target_len - len(values))


def write_pid_sequences(
    df: pd.DataFrame,
    user_ids: Iterable[str],
    output_path: str,
    max_step: int = MAX_STEP,
    min_seq_len: int = MIN_SEQUENCE_LENGTH,
) -> int:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for user_id in user_ids:
            group = df[df["user_id"] == user_id].sort_values("order_id", kind="mergesort")
            if group.empty:
                continue
            questions = group["question_id"].tolist()
            skills = group["skill_id"].tolist()
            answers = group["correct"].tolist()
            for start in range(0, len(questions), max_step):
                q_chunk = questions[start : start + max_step]
                s_chunk = skills[start : start + max_step]
                a_chunk = answers[start : start + max_step]
                seq_len = len(q_chunk)
                if seq_len < min_seq_len:
                    continue
                f.write(f"{seq_len}\n")
                f.write(",".join(map(str, _pad(q_chunk, max_step))) + "\n")
                f.write(",".join(map(str, _pad(s_chunk, max_step))) + "\n")
                f.write(",".join(map(str, _pad(a_chunk, max_step))) + "\n")
                written += 1
    return written


def save_hypergraph(train_df: pd.DataFrame, num_questions: int, num_skills: int, output_path: str) -> None:
    incidence = np.zeros((num_questions, num_skills), dtype=np.int64)
    q_skill = train_df[["question_id", "skill_id"]].drop_duplicates()
    for row in q_skill.itertuples(index=False):
        incidence[int(row.question_id) - 1, int(row.skill_id) - 1] = 1
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(incidence).to_csv(output_path, header=False, index=False)


def save_metadata(summary: PreprocessSummary) -> None:
    with open(summary.metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)


def preprocess_interactions(
    dataset_name: str,
    normalized_df: pd.DataFrame,
    train_pid_path: str,
    test_pid_path: str,
    h_path: str,
    metadata_path: str,
    max_step: int = MAX_STEP,
    min_seq_len: int = MIN_SEQUENCE_LENGTH,
) -> PreprocessSummary:
    if normalized_df.empty:
        raise ValueError(f"No valid interactions found for dataset={dataset_name}.")

    train_users, test_users = split_users(normalized_df)
    train_raw = normalized_df[normalized_df["user_id"].isin(train_users)].copy()
    test_raw = normalized_df[normalized_df["user_id"].isin(test_users)].copy()

    q_map, s_map = fit_mappings(train_raw)
    train_df, dropped_train = map_interactions(train_raw, q_map, s_map)
    test_df, dropped_test = map_interactions(test_raw, q_map, s_map)

    train_sequences = write_pid_sequences(train_df, train_users, train_pid_path, max_step=max_step, min_seq_len=min_seq_len)
    test_sequences = write_pid_sequences(test_df, test_users, test_pid_path, max_step=max_step, min_seq_len=min_seq_len)
    save_hypergraph(train_df, len(q_map), len(s_map), h_path)

    summary = PreprocessSummary(
        dataset=dataset_name,
        train_users=len(train_users),
        test_users=len(test_users),
        num_questions=len(q_map),
        num_skills=len(s_map),
        train_interactions=int(len(train_df)),
        test_interactions=int(len(test_df)),
        train_sequences=train_sequences,
        test_sequences=test_sequences,
        dropped_train_interactions=dropped_train,
        dropped_test_interactions=dropped_test,
        train_pid_path=train_pid_path,
        test_pid_path=test_pid_path,
        h_path=h_path,
        metadata_path=metadata_path,
    )
    save_metadata(summary)
    return summary


def print_summary(summary: PreprocessSummary) -> None:
    print("=" * 72)
    print(f"{summary.dataset} preprocessing complete")
    print("=" * 72)
    print(f"Train users: {summary.train_users}")
    print(f"Test users: {summary.test_users}")
    print(f"Questions: {summary.num_questions}")
    print(f"Skills: {summary.num_skills}")
    print(f"Train interactions: {summary.train_interactions}")
    print(f"Test interactions: {summary.test_interactions}")
    print(f"Train sequences: {summary.train_sequences}")
    print(f"Test sequences: {summary.test_sequences}")
    print(f"Dropped train interactions: {summary.dropped_train_interactions}")
    print(f"Dropped test interactions: {summary.dropped_test_interactions}")
    print(f"Train pid: {summary.train_pid_path}")
    print(f"Test pid: {summary.test_pid_path}")
    print(f"H: {summary.h_path}")
    print(f"Metadata: {summary.metadata_path}")
