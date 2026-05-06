#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess Statics2011 dataset for DGEKT model.

Output files:
  - Statics2011_pid_train.csv (4-line chunk format)
  - Statics2011_pid_test.csv (4-line chunk format)
  - H/statics2011.csv (hypergraph incidence matrix)
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_STEP = 50
TRAIN_TEST_RATIO = 0.8
RANDOM_STATE = 42
MIN_SEQUENCE_LENGTH = 3


def load_and_parse_data(csv_path: str) -> pd.DataFrame:
    """Load raw data and parse essential columns."""
    print(f"[1/5] Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype={'Anon Student Id': str})
    print(f"  Raw shape: {df.shape}")
    
    # Validate required columns
    required_cols = [
        'Anon Student Id', 'Problem Name', 'Step Name',
        'First Transaction Time', 'First Attempt', 'KC (F2011)'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Drop rows with missing critical fields
    df = df.dropna(subset=['Anon Student Id', 'Problem Name', 'Step Name', 'KC (F2011)']).copy()
    
    # Clean whitespace
    df['Anon Student Id'] = df['Anon Student Id'].astype(str).str.strip()
    df['Problem Name'] = df['Problem Name'].astype(str).str.strip()
    df['Step Name'] = df['Step Name'].astype(str).str.strip()
    df['KC (F2011)'] = df['KC (F2011)'].astype(str).str.strip()
    
    # Parse timestamp
    df['ts'] = pd.to_datetime(df['First Transaction Time'], errors='coerce')
    
    # Correctness
    fa = df['First Attempt'].astype(str).str.lower().str.strip()
    df['correct'] = (fa == 'correct').astype(int)
    
    # Question identifier
    df['question_text'] = df['Problem Name'] + '::' + df['Step Name']
    
    # Extract primary skill
    def extract_skill(skill_text: str):
        if pd.isna(skill_text) or skill_text == '.':
            return None
        skills = [s.strip() for s in str(skill_text).split('~~') if s.strip() and s.strip() != '.']
        return skills[0] if skills else None
    
    df['skill'] = df['KC (F2011)'].apply(extract_skill)
    
    # Remove rows without skill
    before = len(df)
    df = df.dropna(subset=['skill']).copy()
    print(f"  Dropped {before - len(df)} rows without skill")
    
    # Sort by user and timestamp
    df = df.sort_values(['Anon Student Id', 'ts'], kind='mergesort').reset_index(drop=True)
    
    print(f"  Processed shape: {df.shape}")
    return df[['Anon Student Id', 'question_text', 'skill', 'correct', 'ts']]


def create_id_mappings(
    df: pd.DataFrame,
    train_user_ids: List[str],
    train_questions: set
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Create ID mappings. Skills mapped ONLY from training data.
    
    CRITICAL FIX: Use ONLY questions from training set to prevent data leakage.
    This ensures test set questions are a subset of training set questions.
    """
    print(f"[2/5] Creating ID mappings...")
    
    train_df = df[df['Anon Student Id'].isin(train_user_ids)]
    
    # Questions ONLY from training set (prevents data leakage)
    all_questions = sorted(train_questions)
    
    # Skills ONLY from training
    train_skills = sorted(train_df['skill'].unique())
    
    # 1-indexed for DGEKT
    question_to_id = {q: idx + 1 for idx, q in enumerate(all_questions)}
    skill_to_id = {s: idx + 1 for idx, s in enumerate(train_skills)}
    
    # Question-skill mapping (training data only)
    train_q_skill = train_df[['question_text', 'skill']].drop_duplicates()
    q_skill_dict = {}
    for _, row in train_q_skill.iterrows():
        q = row['question_text']
        s = row['skill']
        if s in skill_to_id:
            q_skill_dict[q] = skill_to_id[s]
    
    print(f"  Questions: {len(question_to_id)}")
    print(f"  Skills (train): {len(skill_to_id)}")
    print(f"  Q-S pairs: {len(q_skill_dict)}")
    
    return question_to_id, skill_to_id, q_skill_dict


def group_by_user(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Group sequences by user."""
    print(f"[3/5] Grouping by user...")
    
    groups = {}
    for user_id, group in df.groupby('Anon Student Id', sort=False):
        group = group.sort_values('ts', kind='mergesort').reset_index(drop=True)
        groups[user_id] = group
    
    seq_lens = [len(g) for g in groups.values()]
    print(f"  Users: {len(groups)}")
    print(f"  Avg length: {np.mean(seq_lens):.1f}, Min: {min(seq_lens)}, Max: {max(seq_lens)}")
    
    return groups


def write_dgekt_format(
    user_groups: Dict[str, pd.DataFrame],
    user_ids: List[str],
    question_to_id: Dict[str, int],
    q_skill_dict: Dict[str, int],
    output_path: str,
    min_len: int = MIN_SEQUENCE_LENGTH,
    split_name: str = 'train'
) -> int:
    """Write DGEKT 4-line format."""
    written = 0
    discarded = 0
    skipped_oov_question = 0
    skipped_oov_qskill = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for user_id in user_ids:
            if user_id not in user_groups:
                continue
            
            group = user_groups[user_id]
            
            if len(group) < min_len:
                discarded += 1
                continue
            
            questions = []
            skills = []
            answers = []
            
            for _, row in group.iterrows():
                q_text = row['question_text']
                q_id = question_to_id.get(q_text)
                if q_id is None:
                    skipped_oov_question += 1
                    continue
                s_id = q_skill_dict.get(q_text)
                
                if s_id is None:
                    skipped_oov_qskill += 1
                    continue
                
                questions.append(str(q_id))
                skills.append(str(s_id))
                answers.append(str(row['correct']))
            
            if len(questions) < min_len:
                discarded += 1
                continue
            
            # Pad to MAX_STEP
            seq_len = min(len(questions), MAX_STEP)
            questions = questions[:MAX_STEP]
            skills = skills[:MAX_STEP]
            answers = answers[:MAX_STEP]
            
            while len(questions) < MAX_STEP:
                questions.append('-1')
                skills.append('-1')
                answers.append('-1')
            
            # Write 4-line chunk
            f.write(f"{seq_len}\n")
            f.write(",".join(questions) + "\n")
            f.write(",".join(skills) + "\n")
            f.write(",".join(answers) + "\n")
            
            written += 1
    
    print(f"  [{split_name}] Written: {written}, Discarded: {discarded}")
    print(
        f"  [{split_name}] Skipped interactions - "
        f"OOV question: {skipped_oov_question}, "
        f"OOV q-skill: {skipped_oov_qskill}"
    )
    return written


def build_hypergraph(
    user_groups: Dict[str, pd.DataFrame],
    train_user_ids: List[str],
    question_to_id: Dict[str, int],
    q_skill_dict: Dict[str, int],
    output_path: str
) -> None:
    """Build hypergraph from training data ONLY."""
    print(f"[4/5] Building hypergraph...")
    
    num_questions = len(question_to_id)
    num_skills = len(set(q_skill_dict.values()))
    
    incidence = np.zeros((num_questions, num_skills), dtype=int)
    
    # Fill from training data
    train_df = pd.concat([user_groups[uid] for uid in train_user_ids if uid in user_groups])
    
    for _, row in train_df.iterrows():
        q_text = row['question_text']
        s_id = q_skill_dict.get(q_text)
        
        if s_id is not None:
            q_id = question_to_id[q_text]
            incidence[q_id - 1, s_id - 1] = 1
    
    questions_with_skills = np.sum(np.sum(incidence, axis=1) > 0)
    
    print(f"  Matrix: {incidence.shape}")
    print(f"  Questions with skills: {questions_with_skills}/{num_questions}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(incidence).to_csv(output_path, header=False, index=False)
    print(f"  Saved: {output_path}")




def analyze_class_distribution(user_groups: Dict[str, pd.DataFrame], split_name: str) -> Tuple[int, int]:
    """分析类别分布，返回 (correct_count, incorrect_count)"""
    correct = 0
    incorrect = 0
    
    for group in user_groups.values():
        correct += int((group['correct'] == 1).sum())
        incorrect += int((group['correct'] == 0).sum())
    
    total = correct + incorrect
    print(f"  {split_name}:")
    print(f"    ✓ Correct (1): {correct:,} ({correct/total*100:.2f}%)")
    print(f"    ✗ Incorrect (0): {incorrect:,} ({incorrect/total*100:.2f}%)")
    ratio = max(correct, incorrect) / min(correct, incorrect) if min(correct, incorrect) > 0 else 0
    print(f"    Imbalance ratio: {ratio:.2f}:1")
    
    return correct, incorrect

def main():
    """Main pipeline."""
    # 使用相对于脚本位置的路径，而不是当前工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # 上一级目录
    
    input_csv = os.path.join(project_root, 'Dataset/statics2011/AllData_student_step_2011F.csv')
    output_dir = os.path.join(project_root, 'Dataset', 'statics2011')
    train_output = os.path.join(output_dir, 'Statics2011_pid_train.csv')
    test_output = os.path.join(output_dir, 'Statics2011_pid_test.csv')
    hypergraph_output = os.path.join(output_dir, 'H', 'statics2011.csv')
    
    print("=" * 70)
    print("DGEKT Statics2011 Preprocessing")
    print("=" * 70)
    
    # Load
    df = load_and_parse_data(input_csv)
    
    # Split users (80/20)
    all_users = df['Anon Student Id'].unique().tolist()
    train_users, test_users = train_test_split(
        all_users,
        test_size=1 - TRAIN_TEST_RATIO,
        random_state=RANDOM_STATE
    )
    print(f"\n[User Split] Train: {len(train_users)}, Test: {len(test_users)}")
    
    # CRITICAL FIX: Extract questions ONLY from training users
    train_df = df[df['Anon Student Id'].isin(train_users)]
    train_questions = set(train_df['question_text'].unique())
    test_df = df[df['Anon Student Id'].isin(test_users)]
    test_only_questions = set(test_df['question_text'].unique()) - train_questions
    
    print(f"[Question Analysis] Total train questions: {len(train_questions)}")
    print(f"  Test-only questions (will be filtered): {len(test_only_questions)}")
    print(f"  This prevents data leakage!")
    
    # Create mappings with train_questions
    question_to_id, skill_to_id, q_skill_dict = create_id_mappings(df, train_users, train_questions)
    
    # Group
    user_groups = group_by_user(df)
    
    # Write files
    print(f"\n[5/5] Writing output files...")
    train_written = write_dgekt_format(
        user_groups, train_users, question_to_id, q_skill_dict, train_output, split_name='train'
    )
    test_written = write_dgekt_format(
        user_groups, test_users, question_to_id, q_skill_dict, test_output, split_name='test'
    )
    
    # Build hypergraph
    build_hypergraph(user_groups, train_users, question_to_id, q_skill_dict, hypergraph_output)
    
    # Analyze class distribution
    print(f"[4.5/5] Analyzing class distribution...")
    train_user_groups = {uid: user_groups[uid] for uid in train_users if uid in user_groups}
    test_user_groups = {uid: user_groups[uid] for uid in test_users if uid in user_groups}
    
    train_c, train_ic = analyze_class_distribution(train_user_groups, "Training set")
    test_c, test_ic = analyze_class_distribution(test_user_groups, "Test set")
    
    # Summary
    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)
    print(f"Questions: {len(question_to_id)}")
    print(f"Skills: {len(skill_to_id)}")
    print(f"Train: {train_written} sequences")
    print(f"Test: {test_written} sequences")
    print(f"\nOutput:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    print(f"  - {hypergraph_output}")


if __name__ == '__main__':
    main()
