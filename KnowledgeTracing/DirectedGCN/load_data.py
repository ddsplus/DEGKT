# -*- coding: utf-8 -*-

from collections import Counter
from typing import Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from KnowledgeTracing.Constant.Constants import TrainConfig


def _normalize_sparse(mx: sp.spmatrix) -> sp.coo_matrix:
    rowsum = np.array(mx.sum(1)).flatten()
    rowsum[rowsum == 0.0] = 1.0
    inv = 1.0 / rowsum
    r_mat = sp.diags(inv)
    return r_mat.dot(mx).tocoo()


def _to_torch_sparse(mx: sp.coo_matrix) -> torch.Tensor:
    mx = mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((mx.row, mx.col)).astype(np.int64))
    values = torch.from_numpy(mx.data)
    return torch.sparse_coo_tensor(indices, values, size=mx.shape).coalesce()


def build_transition_adjacency(config: TrainConfig) -> Dict[str, torch.Tensor]:
    q = config.num_questions
    edge_counter = Counter()

    with open(config.train_pid_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            len_line = f.readline()
            if not len_line:
                break
            q_line = f.readline()
            _ = f.readline()
            a_line = f.readline()
            if not a_line:
                break

            seq_len = int(len_line.strip())
            questions = [int(token.strip()) for token in q_line.strip().split(",") if token.strip()]
            answers = [int(token.strip()) for token in a_line.strip().split(",") if token.strip()]
            seq_len = min(seq_len, len(questions), len(answers))
            if seq_len <= 1:
                continue

            states = []
            for idx in range(seq_len):
                qid = int(questions[idx])
                ans = int(answers[idx])
                if qid <= 0 or ans not in (0, 1):
                    continue
                state = qid if ans == 1 else qid + q
                states.append(state - 1)
            for src, dst in zip(states[:-1], states[1:]):
                edge_counter[(src, dst)] += 1.0

    n_nodes = 2 * q
    if edge_counter:
        rows, cols, vals = zip(*((src, dst, weight) for (src, dst), weight in edge_counter.items()))
        base = sp.coo_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes), dtype=np.float32)
    else:
        base = sp.coo_matrix((n_nodes, n_nodes), dtype=np.float32)
    base = base + sp.eye(n_nodes, dtype=np.float32, format="coo")
    forward = _normalize_sparse(base)
    backward = _normalize_sparse(base.transpose())
    return {"adj_out": _to_torch_sparse(forward), "adj_in": _to_torch_sparse(backward)}


def build_hypergraph_inputs(config: TrainConfig) -> Dict[str, torch.Tensor]:
    h_df = pd.read_csv(config.h_path, header=None)
    h = h_df.values.astype(np.float32)
    h_state = np.vstack([h, h])
    rows, cols = np.nonzero(h_state)
    values = h_state[rows, cols]
    n_nodes, n_edges = h_state.shape

    incidence = sp.coo_matrix((values, (rows, cols)), shape=(n_nodes, n_edges), dtype=np.float32)
    dv = np.array(incidence.sum(axis=1)).flatten()
    de = np.array(incidence.sum(axis=0)).flatten()
    dv[dv == 0.0] = 1.0
    de[de == 0.0] = 1.0

    dv_inv_sqrt = np.power(dv, -0.5)
    de_inv = np.power(de, -1.0)

    # Sparse construction of G = Dv^{-1/2} * H * De^{-1} * H^T * Dv^{-1/2}
    # Avoid forming a dense (2Q x 2Q) matrix.
    dv_mat = sp.diags(dv_inv_sqrt, format="csr")
    de_mat = sp.diags(de_inv, format="csr")
    g_sp = dv_mat.dot(incidence.tocsr()).dot(de_mat).dot(incidence.tocsr().transpose()).dot(dv_mat)
    g_sp = g_sp.tocoo()

    return {
        "G": _to_torch_sparse(g_sp),
        "incidence": _to_torch_sparse(incidence),
        "dv_inv_sqrt": torch.tensor(dv_inv_sqrt, dtype=torch.float32).unsqueeze(1),
        "de_inv": torch.tensor(de_inv, dtype=torch.float32).unsqueeze(1),
    }
