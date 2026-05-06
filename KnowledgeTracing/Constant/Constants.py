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
                value = int(token)
                if value > 0:
                    q_set.add(value)

        for token in s_line.split(','):
            token = token.strip()
            if token and token.lstrip('-').isdigit():
                value = int(token)
                if value > 0:
                    s_set.add(value)

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


def _compute_num_questions_from_h(dataset_name: str) -> Optional[int]:
    """Read incidence matrix H and infer question count from its row size.

    In this repo's H files, row count is typically 2 * num_questions.
    """
    h_df = load_h_matrix(dataset_name)
    if h_df.empty:
        return None
    row_cnt = int(h_df.shape[0])
    if row_cnt % 2 != 0:
        return None
    return row_cnt // 2


def _is_header_like_row(row) -> bool:
    values = []
    for item in row:
        try:
            values.append(int(item))
        except (TypeError, ValueError):
            return False
    return values == list(range(len(values)))


def h_path(dataset_name: str) -> Optional[str]:
    root = _repo_root()
    h_key = _H_MAP.get(dataset_name)
    if not h_key:
        return None
    return os.path.join(root, 'Dataset', 'H', f'{h_key}.csv')


def load_h_matrix(dataset_name: str) -> pd.DataFrame:
    path = h_path(dataset_name)
    if path is None or not os.path.exists(path):
        return pd.DataFrame()

    h_df = pd.read_csv(path, header=None)
    if h_df.empty:
        return h_df

    if int(h_df.shape[0]) % 2 != 0 and _is_header_like_row(h_df.iloc[0].tolist()):
        h_df = h_df.iloc[1:].reset_index(drop=True)

    return h_df


def infer_pid_question_encoding(dataset_name: str, configured_q: Optional[int] = None) -> str:
    train_path, test_path = _pid_paths(dataset_name)
    q_all = set()
    for path in (train_path, test_path):
        if path and os.path.exists(path):
            q_set, _ = _unique_counts_from_pid_file(path)
            q_all |= q_set

    q = configured_q
    pid_q_max = max(q_all) if q_all else None

    if q is None or pid_q_max is None:
        return 'raw'

    q = int(q)
    pid_q_max = int(pid_q_max)
    if pid_q_max <= q:
        return 'raw'
    if pid_q_max <= 2 * q:
        return 'state_encoded'
    return 'invalid'


def dataset_dimension_report(dataset_name: str) -> Dict[str, Optional[object]]:
    report: Dict[str, Optional[object]] = {
        'dataset': dataset_name,
        'pid_train_path': None,
        'pid_test_path': None,
        'pid_q_min': None,
        'pid_q_max': None,
        'pid_q_unique': None,
        'pid_s_min': None,
        'pid_s_max': None,
        'pid_s_unique': None,
        'h_path': None,
        'h_rows': None,
        'h_cols': None,
        'h_q_count': None,
        'h_error': None,
        'configured_q': numbers.get(dataset_name),
        'configured_s': skill.get(dataset_name),
    }

    train_path, test_path = _pid_paths(dataset_name)
    report['pid_train_path'] = train_path
    report['pid_test_path'] = test_path

    q_all = set()
    s_all = set()
    for path in (train_path, test_path):
        if path and os.path.exists(path):
            q_set, s_set = _unique_counts_from_pid_file(path)
            q_all |= q_set
            s_all |= s_set

    if q_all:
        report['pid_q_min'] = int(min(q_all))
        report['pid_q_max'] = int(max(q_all))
        report['pid_q_unique'] = int(len(q_all))
    if s_all:
        report['pid_s_min'] = int(min(s_all))
        report['pid_s_max'] = int(max(s_all))
        report['pid_s_unique'] = int(len(s_all))

    current_h_path = h_path(dataset_name)
    report['h_path'] = current_h_path
    if current_h_path and os.path.exists(current_h_path):
        raw_h_df = pd.read_csv(current_h_path, header=None)
        report['h_rows'] = int(raw_h_df.shape[0])
        report['h_cols'] = int(raw_h_df.shape[1])

        h_df = load_h_matrix(dataset_name)
        sanitized_rows = int(h_df.shape[0]) if not h_df.empty else 0
        if sanitized_rows % 2 == 0 and sanitized_rows > 0:
            report['h_q_count'] = int(sanitized_rows // 2)
            if sanitized_rows != int(raw_h_df.shape[0]):
                report['h_error'] = (
                    f'H file contained a header-like first row and was sanitized: '
                    f'raw_rows={int(raw_h_df.shape[0])}, sanitized_rows={sanitized_rows}'
                )
        else:
            report['h_error'] = (
                f"H file row count must be even, got {report['h_rows']} "
                f"for dataset={dataset_name} ({current_h_path})"
            )

    report['pid_encoding'] = infer_pid_question_encoding(
        dataset_name, configured_q=report['configured_q']
    )

    return report


def ensure_dataset_stats(dataset_name: str) -> None:
    """Populate numbers[dataset] and skill[dataset] if missing."""
    if dataset_name not in datasets:
        raise KeyError(f"Unsupported dataset: {dataset_name}")

    if dataset_name in numbers and dataset_name in skill:
        return

    q_cnt, s_cnt = _compute_from_pid(dataset_name)
    if q_cnt is None or s_cnt is None:
        q_cnt, s_cnt = _compute_from_raw(dataset_name)

    # For this project, model input dimension must match H-based graph dimension.
    # So we always prioritize question count inferred from Dataset/H/*.csv when available.
    q_cnt_from_h = _compute_num_questions_from_h(dataset_name)
    if q_cnt_from_h is not None:
        q_cnt = q_cnt_from_h

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
PID_QUESTION_ENCODING = infer_pid_question_encoding(DATASET, configured_q=NUM_OF_QUESTIONS)


MAX_STEP = 50
BATCH_SIZE = 256
LR = 0.001
EPOCH = 50
EMB = 256
HIDDEN = 128  # sequence model's
kd_loss = 5.00E-06
LAYERS = 1
