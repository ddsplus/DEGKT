# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List


@dataclass
class PIDRecord:
    seq_len: int
    question_ids: List[int]
    skill_ids: List[int]
    answers: List[int]


def parse_pid_file(path: str, max_step: int) -> List[PIDRecord]:
    records: List[PIDRecord] = []
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

            seq_len = int(len_line.strip())
            questions = [int(token.strip()) for token in q_line.strip().split(",") if token.strip()]
            skills = [int(token.strip()) for token in s_line.strip().split(",") if token.strip()]
            answers = [int(token.strip()) for token in a_line.strip().split(",") if token.strip()]

            questions = questions[:max_step]
            skills = skills[:max_step]
            answers = answers[:max_step]
            if len(questions) < max_step:
                pad = max_step - len(questions)
                questions.extend([-1] * pad)
                skills.extend([-1] * pad)
                answers.extend([-1] * pad)

            seq_len = min(seq_len, max_step)
            if seq_len <= 1:
                continue

            records.append(
                PIDRecord(
                    seq_len=seq_len,
                    question_ids=questions,
                    skill_ids=skills,
                    answers=answers,
                )
            )
    return records
