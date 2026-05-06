import os
import numpy as np
import pandas as pd


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    data_path = os.path.join(project_root, 'Dataset', 'assist2009', 'skill_builder_data.csv')
    output_dir = os.path.join(project_root, 'Dataset', 'assist2009')
    output_train_path = os.path.join(output_dir, 'assist2009_pid_train.csv')
    output_test_path = os.path.join(output_dir, 'assist2009_pid_test.csv')
    h_dir = os.path.join(project_root, 'Dataset', 'H')
    h_path = os.path.join(h_dir, '2009.csv')

    df = pd.read_csv(data_path, encoding='latin1', delimiter=',', dtype={'skill_name': str})
    df = df.dropna(subset=['user_id', 'order_id', 'problem_id', 'skill_name', 'correct']).copy()

    unique_users = df['user_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_users)
    train_users = unique_users[:int(0.8 * len(unique_users))]
    test_users = unique_users[int(0.8 * len(unique_users)):]

    train_df = df[df['user_id'].isin(train_users)].copy()

    # Build contiguous 1-based IDs from training set only.
    problem_ids = sorted(train_df['problem_id'].unique().tolist())
    skill_names = sorted(train_df['skill_name'].astype(str).unique().tolist())
    qid_map = {pid: i + 1 for i, pid in enumerate(problem_ids)}
    cid_map = {sid: i + 1 for i, sid in enumerate(skill_names)}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(h_dir, exist_ok=True)

    def write_sequences(users, output_file):
        written = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for user in users:
                user_df = df[df['user_id'] == user].sort_values('order_id')
                qids, cids, ans = [], [], []

                for _, row in user_df.iterrows():
                    qid = qid_map.get(row['problem_id'])
                    cid = cid_map.get(str(row['skill_name']))
                    if qid is None or cid is None:
                        continue
                    qids.append(qid)
                    cids.append(cid)
                    ans.append(int(row['correct']))

                if len(qids) < 3:
                    continue

                f.write(f"{len(qids)}\n")
                f.write(','.join(map(str, qids)) + '\n')
                f.write(','.join(map(str, cids)) + '\n')
                f.write(','.join(map(str, ans)) + '\n')
                written += 1
        return written

    train_written = write_sequences(train_users, output_train_path)
    test_written = write_sequences(test_users, output_test_path)

    # Build H as question-skill incidence (rows=questions, cols=skills).
    incidence = np.zeros((len(qid_map), len(cid_map)), dtype=int)
    q_skill = train_df[['problem_id', 'skill_name']].drop_duplicates()
    for _, row in q_skill.iterrows():
        qid = qid_map.get(row['problem_id'])
        cid = cid_map.get(str(row['skill_name']))
        if qid is not None and cid is not None:
            incidence[qid - 1, cid - 1] = 1

    pd.DataFrame(incidence).to_csv(h_path, header=False, index=False)

    print('Preprocessing complete!')
    print(f'Train users: {len(train_users)}')
    print(f'Test users: {len(test_users)}')
    print(f'Total questions(train): {len(qid_map)}')
    print(f'Total skills(train): {len(cid_map)}')
    print(f'Train seq written: {train_written}')
    print(f'Test seq written: {test_written}')
    print(f'Data saved to: {output_dir}')
    print(f'H saved to: {h_path}')


if __name__ == '__main__':
    main()
