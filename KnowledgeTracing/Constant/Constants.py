# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : Constant.py
# @Project: GOODKT
# @Comment :

import os
from typing import Dict, Tuple, Optional

import pandas as pd

# Keep original variable name for compatibility
Dpath = '../Dataset'

# Only support these 4 datasets
datasets = {
    'assist2009': 'assist2009',
    'assist2017': 'assist2017',
    'statics2011': 'statics2011',
    'xes3g5m': 'xes3g5m',
}

# These two dicts are kept for backward compatibility.
# Values are computed dynamically (no hard-coded constants).
numbers: Dict[str, int] = {}
skill: Dict[str, int] = {}

_H_MAP = {
    'assist2009': '2009',
    'assist2017': '2017',
    'statics2011': 'statics2011',
    'xes3g5m': 'xes3g5m',
}


def _repo_root() -> str:
    # .../DGEKT-master/KnowledgeTracing/Constant/Constants.py -> .../DGEKT-master
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', '..'))


def _pid_paths(dataset_name: str) -> Tuple[Optional[str], Optional[str]]:
    root = _repo_root()
    ds_dir = os.path.join(root, 'Dataset', dataset_name)

    if dataset_name == 'assist2009':
        return (
            os.path.join(ds_dir, 'assist2009_pid_train.csv'),
            os.path.join(ds_dir, 'assist2009_pid_test.csv'),
        )
    if dataset_name == 'assist2017':
        return (
            os.path.join(ds_dir, 'assist2017_pid_train.csv'),
            os.path.join(ds_dir, 'assist2017_pid_test.csv'),
        )
    if dataset_name == 'statics2011':
        return (
            os.path.join(ds_dir, 'Statics2011_pid_train.csv'),
            os.path.join(ds_dir, 'Statics2011_pid_test.csv'),
        )
    if dataset_name == 'xes3g5m':
        return (
            os.path.join(ds_dir, 'xes3g5m_pid_train.csv'),
            os.path.join(ds_dir, 'xes3g5m_pid_test.csv'),
        )
    return None, None


def _unique_counts_from_pid_file(file_path: str) -> Tuple[set, set]:
    """Parse *_pid_*.csv files (4 lines per student sequence) and return (q_set, s_set)."""
    q_set = set()
    s_set = set()
    if not os.path.exists(file_path):
        return q_set, s_set

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    i = 0
    # Expected format repeated per student:
    #   length
    #   qid1,qid2,...
    #   sid1,sid2,...
    #   correct1,correct2,...
    while i + 3 < len(lines):
        length_line = lines[i]
        if ',' in length_line:
            i += 1
            continue
        if not length_line.isdigit():
            i += 1
            continue

        q_line = lines[i + 1]
        s_line = lines[i + 2]

        for token in q_line.split(','):
            token = token.strip()
            if token and token.lstrip('-').isdigit():
                q_set.add(int(token))

        for token in s_line.split(','):
            token = token.strip()
            if token and token.lstrip('-').isdigit():
                s_set.add(int(token))

        i += 4

    return q_set, s_set


def _compute_from_pid(dataset_name: str) -> Tuple[Optional[int], Optional[int]]:
    train_path, test_path = _pid_paths(dataset_name)
    if not train_path or not os.path.exists(train_path):
        return None, None

    q_all = set()
    s_all = set()

    q_set, s_set = _unique_counts_from_pid_file(train_path)
    q_all |= q_set
    s_all |= s_set

    if test_path and os.path.exists(test_path):
        q_set, s_set = _unique_counts_from_pid_file(test_path)
        q_all |= q_set
        s_all |= s_set

    if not q_all or not s_all:
        return None, None

    q_dim = max(q_all)
    s_dim = max(s_all)

    # If ids are 0-based, allocate +1 to cover max index
    if min(q_all) <= 0:
        q_dim += 1
    if min(s_all) <= 0:
        s_dim += 1

    return int(q_dim), int(s_dim)


def _compute_from_raw(dataset_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Fallback: compute from raw CSV if pid files missing."""
    root = _repo_root()

    if dataset_name == 'assist2009':
        raw_path = os.path.join(root, 'Dataset', 'assist2009', 'skill_builder_data.csv')
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            return int(df['problem_id'].nunique()), int(df['skill_id'].nunique())

    if dataset_name == 'assist2017':
        raw_path = os.path.abspath(os.path.join(root, '..', 'Data', 'ASSIST2017', 'anonymized_full_release_competition_dataset.csv'))
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path, low_memory=False)
            return int(df['problemId'].nunique()), int(df['skill'].nunique())

    if dataset_name == 'statics2011':
        raw_path = os.path.join(root, 'Dataset', 'statics2011', 'AllData_student_step_2011F.csv')
        if os.path.exists(raw_path):
            df = pd.read_csv(raw_path)
            return int(df['Problem Name'].nunique()), int(df['KC (F2011)'].nunique())

    if dataset_name == 'xes3g5m':
        raw_train = os.path.join(root, 'Dataset', 'xes3g5m', 'train.csv')
        raw_test = os.path.join(root, 'Dataset', 'xes3g5m', 'test.csv')
        dfs = []
        if os.path.exists(raw_train):
            dfs.append(pd.read_csv(raw_train))
        if os.path.exists(raw_test):
            dfs.append(pd.read_csv(raw_test))
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            return int(df['problem_id'].nunique()), int(df['skill_id'].nunique())

    return None, None


def ensure_dataset_stats(dataset_name: str) -> None:
    """Populate numbers[dataset] and skill[dataset] if missing."""
    if dataset_name not in datasets:
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    if dataset_name in numbers and dataset_name in skill:
        return

    q_cnt, s_cnt = _compute_from_pid(dataset_name)
    if q_cnt is None or s_cnt is None:
        q_cnt, s_cnt = _compute_from_raw(dataset_name)

    if q_cnt is None or s_cnt is None:
        raise FileNotFoundError(
            f"Cannot compute statistics for dataset '{dataset_name}'. "
            f"Ensure pid files under DGEKT-master/Dataset/{dataset_name}/ or raw CSV exists."
        )

    numbers[dataset_name] = int(q_cnt)
    skill[dataset_name] = int(s_cnt)


# -------------------- Active config --------------------

# Choose dataset here
DATASET = datasets['assist2017']

# Dynamically computed (no hard-code)
ensure_dataset_stats(DATASET)
NUM_OF_QUESTIONS = numbers[DATASET]
NUM_OF_SKILLS = skill[DATASET]

# Hypergraph file key under DGEKT-master/Dataset/H/
H = _H_MAP[DATASET]


MAX_STEP = 50
BATCH_SIZE = 256
LR = 0.001
EPOCH = 50
EMB = 256
HIDDEN = 128  # sequence model's
kd_loss = 5.00E-06
LAYERS = 1
