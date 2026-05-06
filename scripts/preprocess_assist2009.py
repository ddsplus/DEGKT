import os
import numpy as np
import pandas as pd
from collections import defaultdict


def main():
    # 1. 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 2. 构建正确的数据路径
    data_path = os.path.join(project_root, 'Dataset', 'assist2009', 'skill_builder_data.csv')
    output_train_path = os.path.join(project_root, 'Dataset', 'assist2009', 'assist2009_pid_train.csv')
    output_test_path = os.path.join(project_root, 'Dataset', 'assist2009', 'assist2009_pid_test.csv')
    h_dir = os.path.join(project_root, 'Dataset', 'H')
    h_path = os.path.join(h_dir, '2009.csv')
    
    # 3. 读取原始数据
    df = pd.read_csv(data_path, encoding='latin1', delimiter=',', dtype={'skill_id': str, 'skill_name': str})

    # 4. 按用户划分训练集/测试集（8:2）
    unique_users = df['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)
    train_users = unique_users[:int(0.8 * len(unique_users))]
    test_users = unique_users[int(0.8 * len(unique_users)):]  # 修正：移除行尾的\

    # 5. 仅基于训练集构建ID映射（防泄露关键）
    train_df = df[df['user_id'].isin(train_users)]
    
    # 问题ID映射
    problem_ids = train_df['problem_id'].unique()
    qid_map = {pid: i for i, pid in enumerate(problem_ids)}
    
    # 概念ID映射
    skill_names = train_df['skill_name'].unique()
    cid_map = {skill: i for i, skill in enumerate(skill_names)}

    # 6. 生成预处理数据文件
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)
    
    def write_sequences(users, output_file):
        with open(output_file, 'w') as f:
            for user in users:
                user_df = df[df['user_id'] == user].sort_values('order_id')
                # 获取问题ID和概念ID（测试集新ID自动扩展）
                qids = [qid_map.get(pid, len(qid_map)) for pid in user_df['problem_id']]
                cids = [cid_map.get(skill, len(cid_map)) for skill in user_df['skill_name']]
                ans = user_df['correct'].tolist()
                
                # 写入序列（每4行）
                f.write(f"{len(qids)}\n")
                f.write(','.join(map(str, qids)) + '\n')
                f.write(','.join(map(str, cids)) + '\n')
                f.write(','.join(map(str, ans)) + '\n')

    write_sequences(train_users, output_train_path)
    write_sequences(test_users, output_test_path)

    # 7. 仅使用训练集构建超图（防泄露关键）
    os.makedirs(h_dir, exist_ok=True)
    skills = list(cid_map.keys())
    n_skills = len(skills)
    skill_matrix = np.zeros((n_skills, n_skills))

    for _, group in train_df.groupby('problem_id'):
        skill_indices = [cid_map[skill] for skill in group['skill_name'].unique()]
        for i in skill_indices:
            for j in skill_indices:
                if i != j:
                    skill_matrix[i, j] += 1

    # 归一化超图
    skill_matrix = skill_matrix / (skill_matrix.sum(axis=1, keepdims=True) + 1e-10)
    np.savetxt(h_path, skill_matrix, delimiter=',')

    print(f"Preprocessing complete!\n"
          f"Train users: {len(train_users)}\n"
          f"Test users: {len(test_users)}\n"
          f"Total problems: {len(qid_map)}\n"
          f"Total skills: {len(cid_map)}\n"
          f"Data saved to: {output_train_path}")

if __name__ == '__main__':
    main()