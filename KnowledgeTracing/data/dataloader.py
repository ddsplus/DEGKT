# -*- coding: utf-8 -*-

from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

from KnowledgeTracing.Constant.Constants import TrainConfig
from KnowledgeTracing.data.preprocess import parse_pid_file


class PIDSequenceDataset(Dataset):
    def __init__(self, path: str, max_step: int):
        self.records = parse_pid_file(path, max_step=max_step)
        self.max_step = max_step

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        mask = [1] * record.seq_len + [0] * (self.max_step - record.seq_len)
        return {
            "question_ids": torch.tensor(record.question_ids, dtype=torch.long),
            "skill_ids": torch.tensor(record.skill_ids, dtype=torch.long),
            "answers": torch.tensor(record.answers, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "seq_len": torch.tensor(record.seq_len, dtype=torch.long),
        }


def get_loader(path: str, config: TrainConfig, shuffle: bool) -> DataLoader:
    dataset = PIDSequenceDataset(path, max_step=config.max_step)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )


def get_loaders(config: TrainConfig):
    return (
        get_loader(config.train_pid_path, config, shuffle=True),
        get_loader(config.test_pid_path, config, shuffle=False),
    )
