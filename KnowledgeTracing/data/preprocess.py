# -*- coding: utf-8 -*-
# @Time : 2022/4/28 19:18
# @Author : Yumo
# @File : preprocess.py
# @Project: GOODKT
# @Comment :
import numpy as np
import itertools
import tqdm


class DataReader:
    def __init__(self, path, maxstep):
        self.path = path
        self.maxstep = maxstep


    def getTrainData(self):
        trainqus = np.array([])
        trainans = np.array([])

        with open(self.path, 'r', encoding='UTF-8-sig') as train:
            for seq_len_line, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[train] * 4), desc='loading train data:    ',
                                                        mininterval=2):
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                seq_len = min(len(ques), len(ans))
                ques = ques[:seq_len]
                ans = ans[:seq_len]

                mod = 0 if seq_len % self.maxstep == 0 else (self.maxstep - seq_len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)

                trainqus = np.append(trainqus, ques).astype(int)
                trainans = np.append(trainans, ans).astype(int)
        return trainqus.reshape([-1, self.maxstep]), trainans.reshape([-1, self.maxstep])

    def getTestData(self):
        testqus = np.array([])
        testans = np.array([])
        with open(self.path, 'r', encoding='UTF-8-sig') as test:
            for seq_len_line, ques, _, ans in tqdm.tqdm(itertools.zip_longest(*[test] * 4), desc='loading test data:    ',
                                                        mininterval=2):
                ques = np.array(ques.strip().strip(',').split(',')).astype(int)
                ans = np.array(ans.strip().strip(',').split(',')).astype(int)
                seq_len = min(len(ques), len(ans))
                ques = ques[:seq_len]
                ans = ans[:seq_len]
                mod = 0 if seq_len % self.maxstep == 0 else (self.maxstep - seq_len % self.maxstep)
                zero = np.zeros(mod) - 1
                ques = np.append(ques, zero)
                ans = np.append(ans, zero)
                testqus = np.append(testqus, ques).astype(int)
                testans = np.append(testans, ans).astype(int)
        return testqus.reshape([-1, self.maxstep]), testans.reshape([-1, self.maxstep])
