#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess XES3G5M dataset for DGEKT model.

This script processes the XES3G5M dataset and generates:
  1. xes3g5m_pid_train.csv - Training sequences (4-line chunk format)
  2. xes3g5m_pid_test.csv - Test sequences (4-line chunk format)
  3. H/xes3g5m.csv - Hypergraph incidence matrix (questions × skills)

Key design:
  - Strict 80/20 user split (80% train, 20% test)
  - Hypergraph built ONLY from training set
  - Test set can have questions unseen in training (retained for prediction)
  - All sequences padded to MAX_STEP=50
  - Handles OOV (out-of-vocabulary) questions gracefully
"""

import os
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict


MAX_STEP = 50
TRAIN_TEST_RATIO = 0.8
RANDOM_STATE = 42
MIN_SEQUENCE_LENGTH = 3


def parse_list_column(value: str) -> List[int]:
    """Parse comma-separated string to list of integers."""
    if pd.isna(value) or value == '':
        return []
    try:
        return [int(x.strip()) for x in str(value).split(',') if x.strip()]
    except (ValueError, AttributeError):
        return []


def load_and_parse_data(csv_path: str) -> pd.DataFrame:
    """Load XES3G5M CSV and parse sequence data."""
    print(f"[1/5] Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, encoding='utf-8', dtype={'uid': str})
    print(f"  Raw rows: {len(df):,}")
    
    # Parse sequences
    data = []
    for _, row in df.iterrows():
        uid = str(row['uid']).strip()
        questions = parse_list_column(row.get('questions', ''))
        concepts = parse_list_column(row.get('concepts', ''))
        responses = parse_list_column(row.get('responses', ''))
        
        # Ensure consistent length
        if not questions or not concepts or not responses:
            continue
        
        min_len = min(len(questions), len(concepts), len(responses))
        questions = questions[:min_len]
        concepts = concepts[:min_len]
        responses = responses[:min_len]
        
        # Validate values
        valid = True
        for q, c, r in zip(questions, concepts, responses):
            if q <= 0 or c <= 0 or r < 0 or r > 1:
                valid = False
                break
        
        if not valid or not questions:
            continue
        
        data.append({
            'uid': uid,
            'questions': questions,
            'concepts': concepts,
            'responses': responses
        })
    
    print(f"  Valid sequences: {len(data):,}")
    
    if not data:
        raise ValueError("No valid sequences found in data")
    
    return pd.DataFrame(data)


def split_users(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split users into train/test sets (80/20)."""
    print(f"\n[User Split]")
    all_users = df['uid'].unique().tolist()
    train_users, test_users = train_test_split(
        all_users,
        test_size=1 - TRAIN_TEST_RATIO,
        random_state=RANDOM_STATE
    )
    print(f"  Train users: {len(train_users)}")
    print(f"  Test users: {len(test_users)}")
    return train_users, test_users


def create_id_mappings(
    df: pd.DataFrame,
    train_users: List[str]
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Create ID mappings.
    - Questions: ALL unique questions (train + test)
    - Skills: ONLY from training set
    - Q2C: Question to skill mapping (from training set)
    """
    print(f"\n[2/5] Creating ID mappings...")
    
    train_df = df[df['uid'].isin(train_users)]
    
    # Collect all unique questions and skills
    all_questions = set()
    train_skills = set()
    q2c_mapping = {}  # question -> primary skill
    
    # Process training data
    for _, row in train_df.iterrows():
        questions = row['questions']
        concepts = row['concepts']
        
        for q, c in zip(questions, concepts):
            all_questions.add(q)
            train_skills.add(c)
            
            # Track question-skill relationship (prefer most common skill)
            if q not in q2c_mapping:
                q2c_mapping[q] = c
    
    # Process all data to find test-only questions
    test_only_questions = set()
    for _, row in df.iterrows():
        for q in row['questions']:
            all_questions.add(q)
            if q not in q2c_mapping:
                test_only_questions.add(q)
    
    # Create mappings (1-indexed for DGEKT)
    question_to_id = {q: idx + 1 for idx, q in enumerate(sorted(all_questions))}
    skill_to_id = {s: idx + 1 for idx, s in enumerate(sorted(train_skills))}
    
    print(f"  Total questions: {len(question_to_id)}")
    print(f"  Total skills (from train): {len(skill_to_id)}")
    print(f"  Test-only questions: {len(test_only_questions)}")
    print(f"  Q-S mappings: {len(q2c_mapping)}")
    
    return question_to_id, skill_to_id, q2c_mapping


def write_dgekt_format(
    df: pd.DataFrame,
    user_ids: List[str],
    question_to_id: Dict[int, int],
    q2c_mapping: Dict[int, int],
    skill_to_id: Dict[int, int],
    output_path: str,
    min_len: int = MIN_SEQUENCE_LENGTH
) -> int:
    """
    Write sequences in DGEKT 4-line format.
    
    Format per user:
      Line 1: Sequence length
      Line 2: Question IDs (comma-separated)
      Line 3: Skill IDs (comma-separated)
      Line 4: Responses (comma-separated, 0/1)
    """
    written = 0
    discarded = 0
    skipped_interactions = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df[df['uid'].isin(user_ids)].iterrows():
            questions = row['questions']
            responses = row['responses']
            
            # Filter: sequences with unknown skills
            valid_q = []
            valid_s = []
            valid_r = []
            
            for q, r in zip(questions, responses):
                q_id = question_to_id.get(q)
                s_id = q2c_mapping.get(q)
                
                if s_id is None or s_id not in skill_to_id:
                    skipped_interactions += 1
                    continue
                
                valid_q.append(str(q_id))
                valid_s.append(str(skill_to_id[s_id]))
                valid_r.append(str(int(r)))
            
            # Must have minimum length
            if len(valid_q) < min_len:
                discarded += 1
                continue
            
            # Pad or truncate to MAX_STEP
            seq_len = min(len(valid_q), MAX_STEP)
            valid_q = valid_q[:MAX_STEP]
            valid_s = valid_s[:MAX_STEP]
            valid_r = valid_r[:MAX_STEP]
            
            # Pad with -1
            while len(valid_q) < MAX_STEP:
                valid_q.append('-1')
                valid_s.append('-1')
                valid_r.append('-1')
            
            # Write 4-line chunk
            f.write(f"{seq_len}\n")
            f.write(",".join(valid_q) + "\n")
            f.write(",".join(valid_s) + "\n")
            f.write(",".join(valid_r) + "\n")
            
            written += 1
    
    print(f"  Written: {written}, Discarded: {discarded}, OOV skipped: {skipped_interactions}")
    return written


def build_hypergraph(
    df: pd.DataFrame,
    train_users: List[str],
    question_to_id: Dict[int, int],
    skill_to_id: Dict[int, int],
    q2c_mapping: Dict[int, int],
    output_path: str
) -> None:
    """
    Build hypergraph incidence matrix from TRAINING DATA ONLY.
    
    Matrix shape: [num_questions, num_skills]
    Entry [i][j] = 1 if question i is tagged with skill j
    """
    print(f"\n[4/5] Building hypergraph from training data...")
    
    num_questions = len(question_to_id)
    num_skills = len(skill_to_id)
    
    incidence = np.zeros((num_questions, num_skills), dtype=int)
    
    # Fill from training data only
    train_df = df[df['uid'].isin(train_users)]
    
    for _, row in train_df.iterrows():
        questions = row['questions']
        for q in questions:
            if q in q2c_mapping:
                s = q2c_mapping[q]
                if s in skill_to_id:
                    q_id = question_to_id[q]
                    s_id = skill_to_id[s]
                    incidence[q_id - 1, s_id - 1] = 1
    
    # Statistics
    questions_with_skills = np.sum(np.sum(incidence, axis=1) > 0)
    skills_covered = np.sum(np.sum(incidence, axis=0) > 0)
    
    print(f"  Matrix shape: {incidence.shape}")
    print(f"  Questions with skills: {questions_with_skills}/{num_questions}")
    print(f"  Skills covered: {skills_covered}/{num_skills}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(incidence).to_csv(output_path, header=False, index=False)
    print(f"  Saved: {output_path}")


def analyze_class_distribution(
    df: pd.DataFrame,
    user_ids: List[str],
    split_name: str
) -> Tuple[int, int]:
    """Analyze class distribution in a split."""
    correct = 0
    incorrect = 0
    
    for _, row in df[df['uid'].isin(user_ids)].iterrows():
        responses = row['responses']
        correct += sum(responses)
        incorrect += len(responses) - sum(responses)
    
    total = correct + incorrect
    print(f"  {split_name}:")
    print(f"    ✓ Correct (1): {correct:,} ({correct/total*100:.2f}%)")
    print(f"    ✗ Incorrect (0): {incorrect:,} ({incorrect/total*100:.2f}%)")
    if min(correct, incorrect) > 0:
        ratio = max(correct, incorrect) / min(correct, incorrect)
        print(f"    Imbalance ratio: {ratio:.2f}:1")
    
    return correct, incorrect


def main():
    """Main preprocessing pipeline."""
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    input_train = os.path.join(project_root, 'Dataset/xes3g5m/train.csv')
    input_test = os.path.join(project_root, 'Dataset/xes3g5m/test.csv')
    output_dir = os.path.join(project_root, 'Dataset')
    
    train_output = os.path.join(output_dir, 'xes3g5m_pid_train.csv')
    test_output = os.path.join(output_dir, 'xes3g5m_pid_test.csv')
    hypergraph_output = os.path.join(output_dir, 'H', 'xes3g5m.csv')
    
    print("=" * 70)
    print("DGEKT XES3G5M Preprocessing")
    print("=" * 70)
    
    # Load data
    train_df = load_and_parse_data(input_train)
    test_df = load_and_parse_data(input_test)
    
    # Combine for user splitting (ensure no data leakage)
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Split users (80/20)
    train_users, test_users = split_users(df)
    
    # Create mappings
    question_to_id, skill_to_id, q2c_mapping = create_id_mappings(df, train_users)
    
    # Write DGEKT format
    print(f"\n[5/5] Writing output files...")
    
    train_written = write_dgekt_format(
        df, train_users, question_to_id, q2c_mapping, skill_to_id,
        train_output
    )
    
    test_written = write_dgekt_format(
        df, test_users, question_to_id, q2c_mapping, skill_to_id,
        test_output
    )
    
    # Build hypergraph
    build_hypergraph(
        df, train_users, question_to_id, skill_to_id, q2c_mapping,
        hypergraph_output
    )
    
    # Analyze class distribution
    print(f"\n[Class Distribution Analysis]")
    analyze_class_distribution(df, train_users, "Training set")
    analyze_class_distribution(df, test_users, "Test set")
    
    # Summary
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"Total questions: {len(question_to_id)}")
    print(f"Total skills: {len(skill_to_id)}")
    print(f"Train sequences: {train_written}")
    print(f"Test sequences: {test_written}")
    print(f"\nOutput files:")
    print(f"  1. {train_output}")
    print(f"  2. {test_output}")
    print(f"  3. {hypergraph_output}")


if __name__ == '__main__':
    main()
