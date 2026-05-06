# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : OneHot.py
# @Project: GOODKT
# @Comment :
from torch.utils.data.dataset import Dataset
from KnowledgeTracing.Constant import Constants as C
import torch


class OneHot(Dataset):
    def __init__(self, ques, ans):
        self.ques = ques
        self.ans = ans
        self.numofques = C.NUM_OF_QUESTIONS

    def __len__(self):
        return len(self.ques)

    def __getitem__(self, index):
        questions = self.ques[index]
        answers = self.ans[index]
        lab = self.onehot(questions, answers)
        return lab

    def onehot(self, questions, answers):
        label = torch.zeros(C.MAX_STEP, 2 * self.numofques)
        for i in range(C.MAX_STEP):
            qid = int(questions[i])
            ans = int(answers[i])

            if C.PID_QUESTION_ENCODING == 'state_encoded':
                if 1 <= qid <= 2 * self.numofques:
                    label[i][qid - 1] = 1
            else:
                if ans > 0 and qid > 0:
                    label[i][qid - 1] = 1
                elif ans == 0 and qid > 0:
                    label[i][self.numofques + qid - 1] = 1
        return label
