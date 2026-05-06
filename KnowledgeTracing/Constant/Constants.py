# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd


SUPPORTED_DATASETS = ("assist2009", "assist2017", "statics2011", "xes3g5m")

H_KEYS = {
    "assist2009": "2009",
    "assist2017": "2017",
    "statics2011": "statics2011",
    "xes3g5m": "xes3g5m",
}


@dataclass
class TrainConfig:
    dataset: str
    train_pid_path: str
    test_pid_path: str
    h_path: str
    num_questions: int
    num_skills: int
    max_step: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 50
    emb_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    kd_weight: float = 5e-6
    patience: int = 10
    seed: int = 42
    num_workers: int = 0


def repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def dataset_dir(dataset: str) -> str:
    return os.path.join(repo_root(), "Dataset", dataset)


def pid_paths(dataset: str) -> Tuple[str, str]:
    ds_dir = dataset_dir(dataset)
    if dataset == "assist2009":
        return (
            os.path.join(ds_dir, "assist2009_pid_train.csv"),
            os.path.join(ds_dir, "assist2009_pid_test.csv"),
        )
    if dataset == "assist2017":
        return (
            os.path.join(ds_dir, "assist2017_pid_train.csv"),
            os.path.join(ds_dir, "assist2017_pid_test.csv"),
        )
    if dataset == "statics2011":
        return (
            os.path.join(ds_dir, "Statics2011_pid_train.csv"),
            os.path.join(ds_dir, "Statics2011_pid_test.csv"),
        )
    if dataset == "xes3g5m":
        return (
            os.path.join(ds_dir, "xes3g5m_pid_train.csv"),
            os.path.join(ds_dir, "xes3g5m_pid_test.csv"),
        )
    raise KeyError(f"Unsupported dataset: {dataset}")


def h_path(dataset: str) -> str:
    return os.path.join(repo_root(), "Dataset", "H", f"{H_KEYS[dataset]}.csv")


def _read_pid_stats(path: str) -> Tuple[Optional[int], Optional[int], int]:
    if not os.path.exists(path):
        return None, None, 0
    max_q = None
    max_s = None
    chunks = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            len_line = f.readline()
            if not len_line:
                break
            q_line = f.readline()
            s_line = f.readline()
            a_line = f.readline()
            if not a_line:
                break
            chunks += 1
            q_vals = [int(token) for token in q_line.strip().split(",") if token.strip() and int(token) > 0]
            s_vals = [int(token) for token in s_line.strip().split(",") if token.strip() and int(token) > 0]
            if q_vals:
                max_q = max(max_q or 0, max(q_vals))
            if s_vals:
                max_s = max(max_s or 0, max(s_vals))
    return max_q, max_s, chunks


def load_h_matrix(dataset: str) -> pd.DataFrame:
    path = h_path(dataset)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing H file for dataset={dataset}: {path}. Run the preprocess script first."
        )
    h_df = pd.read_csv(path, header=None)
    if h_df.empty:
        raise ValueError(f"H file is empty for dataset={dataset}: {path}")
    return h_df


def infer_dataset_stats(dataset: str) -> Dict[str, int]:
    train_path, test_path = pid_paths(dataset)
    for path in (train_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing pid file for dataset={dataset}: {path}. Run the preprocess script first."
            )

    h_df = load_h_matrix(dataset)
    num_questions = int(h_df.shape[0])
    num_skills = int(h_df.shape[1])
    if num_questions <= 0 or num_skills <= 0:
        raise ValueError(f"Invalid H shape for dataset={dataset}: {h_df.shape}")

    train_q_max, train_s_max, train_chunks = _read_pid_stats(train_path)
    test_q_max, test_s_max, test_chunks = _read_pid_stats(test_path)
    q_candidates: List[int] = [value for value in (train_q_max, test_q_max) if value is not None]
    s_candidates: List[int] = [value for value in (train_s_max, test_s_max) if value is not None]
    if not q_candidates or not s_candidates:
        raise ValueError(f"PID files for dataset={dataset} contain no valid interactions.")
    pid_q_max = max(q_candidates)
    pid_s_max = max(s_candidates)

    if pid_q_max > num_questions:
        raise ValueError(
            f"Question id exceeds H rows for dataset={dataset}: pid_q_max={pid_q_max}, h_rows={num_questions}"
        )
    if pid_s_max > num_skills:
        raise ValueError(
            f"Skill id exceeds H cols for dataset={dataset}: pid_s_max={pid_s_max}, h_cols={num_skills}"
        )

    return {
        "num_questions": num_questions,
        "num_skills": num_skills,
        "train_chunks": train_chunks,
        "test_chunks": test_chunks,
        "pid_q_max": int(pid_q_max),
        "pid_s_max": int(pid_s_max),
    }


def build_config(dataset: str, **overrides) -> TrainConfig:
    if dataset not in SUPPORTED_DATASETS:
        raise KeyError(f"Unsupported dataset: {dataset}")
    stats = infer_dataset_stats(dataset)
    train_path, test_path = pid_paths(dataset)
    values = dict(
        dataset=dataset,
        train_pid_path=train_path,
        test_pid_path=test_path,
        h_path=h_path(dataset),
        num_questions=stats["num_questions"],
        num_skills=stats["num_skills"],
    )
    values.update(overrides)
    return TrainConfig(**values)
