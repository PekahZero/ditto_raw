import os
import sys
import jsonlines
import pickle
import numpy as np
import argparse
import csv
from sklearn.metrics import recall_score
from tensorboardX import SummaryWriter

from tqdm import tqdm

sys.path.append("sentence-transformers")

from sentence_transformers import SentenceTransformer

def encode_all(path, input_fn, model, overwrite=False):
    """Encode a collection of entries and output to a file

    Args:
        path (str): the input path
        input_fn (str): the file of the serialzied entries
        model (SentenceTransformer): the transformer model
        overwrite (boolean, optional): whether to overwrite out_fn

    Returns:
        List of str: the serialized entries
        List of np.ndarray: the encoded vectors
    """
    input_fn = os.path.join(path, input_fn)
    output_fn = input_fn + '.mat'

    # read from input_fn
    lines = open(input_fn).read().split('\n')

    # encode and dump
    if not os.path.exists(output_fn) or overwrite:
        vectors = model.encode(lines)
        vectors = [v / np.linalg.norm(v) for v in vectors]
        pickle.dump(vectors, open(output_fn, 'wb'))
    else:
        vectors = pickle.load(open(output_fn, 'rb'))
    return lines, vectors


def blocked_matmul(mata, matb,
                   threshold=None,
                   k=None,
                   batch_size=512):
    """Find the most similar pairs of vectors from two matrices (top-k or threshold)

    Args:
        mata (np.ndarray): the first matrix
        matb (np.ndarray): the second matrix
        threshold (float, optional): if set, return all pairs of cosine
            similarity above the threshold
        k (int, optional): if set, return for each row in matb the top-k
            most similar vectors in mata
        batch_size (int, optional): the batch size of each block

    Returns:
        list of tuples: the pairs of similar vectors' indices and the similarity
    """
    mata = np.array(mata)
    matb = np.array(matb)
    results = []
    for start in tqdm(range(0, len(matb), batch_size)):
        block = matb[start:start+batch_size]
        sim_mat = np.matmul(mata, block.transpose())
        if k is not None:
            indices = np.argpartition(-sim_mat, k, axis=0)
            for row in indices[:k]:
                for idx_b, idx_a in enumerate(row):
                    idx_b += start
                    results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))

        elif threshold is not None:
            indices = np.argwhere(sim_mat >= threshold)
            # total += len(indices)
            for idx_a, idx_b in indices:
                idx_b += start
                results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
    # seen = {}
    # unique_results = []
    # for idx_a, idx_b, sim in results:
    #     if (idx_a, idx_b) not in seen:
    #         seen[(idx_a, idx_b)] = True
    #         unique_results.append((idx_a, idx_b, sim))
    # results = unique_results

    return results


def dump_pairs(out_path,out_fn, entries_a, entries_b, pairs):
    """Dump the pairs to a jsonl file
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    with jsonlines.open(os.path.join(out_path, out_fn), mode='w') as writer:
        for idx_a, idx_b, score in pairs:
            writer.write([entries_a[idx_a], entries_b[idx_b], str(score)])

# path :数据集名称 Structed/XXXXX/
# 返回成对的标签
def read_ground_truth(path):
    """Read groundtruth matches from train/valid/test sets.
    读取已知匹配的真实结果。 读取真实的标签

    Args:
        path (str): the path to the datasets

    Returns:
        List of tuple: matched pairs
        int: the total number of original match / non-match
    """
    res = []
    total = 0
    for fn in ['train.csv', 'valid.csv', 'test.csv']:
        reader = csv.DictReader(open(os.path.join(path, fn)))
        for row in reader:
            lid = int(row['ltable_id'])
            rid = int(row['rtable_id'])
            lbl = row['label']
            # 是同一对，添加
            if int(lbl) == 1:
                res.append((lid, rid))
            total += 1
    return res, total

def evaluate_pairs(pairs, ground_truth, k=None):
    """Return the recall given the set of pairs and ground truths.

    Args:
        pairs (list): the computed list
        ground_truth (list): the ground truth list
        k (int, optional): if set, compute recall only for
                           the top k for each right index

    Returns:
        float: the recall
    """
    if k:
        r_index = {}
        # 遍历计算出的匹配对列表
        for l, r, score in pairs:
            if r not in r_index:
                r_index[r] = []
            r_index[r].append((score, l))
            # # 将 (score, l) 元组添加到 r 对应的列表中
            # r_index[r] right序号为r的entity, 
            # 和left序号为l的entity，配对的几率为score

        # 对每个right, 取前K个最高匹配的(l,r)
        # pairs 候选对
        pairs = []
        for r in r_index:
            r_index[r].sort(reverse=True)
            for _, l in r_index[r][:k]:
                pairs.append((l, r))

        # 去重
        pairs = set(pairs)
        # 创建一个与 ground_truth 等长的由 1 构成的列表 y_true
        y_true = [1 for _ in ground_truth]
        # 根据匹配对是否在 pairs 中生成预测标签列表 y_pred
        y_pred = [int(p in pairs) for p in ground_truth]
        # 返回计算出的召回率和筛选后的匹配对数量 
        return recall_score(y_true, y_pred), len(pairs)
    else:
        print('pairs =', len(pairs), pairs[:10])  # 打印 pairs 的长度和前 10 个元素
        print('ground_truth =', len(ground_truth))  # 打印 ground_truth 的长度
        pairs = [(l, r) for l, r, _ in pairs]  # 将 pairs 中的每个元素转换为只包含 l 和 r 的元组
        pairs = set(pairs)  # 将 pairs 转换为集合形式
        y_true = [1 for _ in ground_truth]  # 创建一个与 ground_truth 等长的由 1 构成的列表 y_true
        y_pred = [int(p in pairs) for p in ground_truth]  # 根据匹配对是否在 pairs 中生成预测标签列表 y_pred
        return recall_score(y_true, y_pred)  # 返回计算出的召回率

def evaluate_blocking(pairs, hp):
    
        print('Read ground truth')
        # 读取真正的标签
        ground_truth, total = read_ground_truth(hp.input_path)

    # 如果超参数 hp.k 不存在（为空），则计算所有候选配对的召回率
    # 调用 evaluate_pairs 函数计算召回率，并将其存储在变量 recall 中。
    # 候选集大小即为配对的数量
        if hp.k:
            recalls, sizes = [], []
            for k in range(1, hp.k+1):
                recall, size = evaluate_pairs(pairs, ground_truth, k)
                recalls.append(recall)
                sizes.append(size)
            return recalls, sizes
        else:
            recall = evaluate_pairs(pairs, ground_truth)
            return recall, len(pairs)
        
# 输入2个文件 (行数可以不同)
# 输出了json文件 [entry1, entry2, confidence] 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./input/")
    parser.add_argument("--left_fn", type=str, default='table_a_small.txt')
    parser.add_argument("--right_fn", type=str, default='table_b.txt')
    parser.add_argument("--output_path", type=str, default='./output/')
    parser.add_argument("--output_fn", type=str, default='candidates.jsonl')
    parser.add_argument("--logdir", type=str, default='./log')
    parser.add_argument("--model_fn", type=str, default="./models/Structured_Beer")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--k", type=int, default=10) # top-k
    parser.add_argument("--threshold", type=float, default=None) # 0.6
    hp = parser.parse_args()

    # load the model
    model = SentenceTransformer(hp.model_fn)

    # generate the vectors
    mata = matb = None
    entries_a = entries_b = None
    if hp.left_fn is not None:
        entries_a, mata = encode_all(hp.input_path, hp.left_fn, model)
    if hp.right_fn is not None:
        entries_b, matb = encode_all(hp.input_path, hp.right_fn, model)

    if mata and matb:
        pairs = blocked_matmul(mata, matb,
                   threshold=hp.threshold,
                   k=hp.k,
                   batch_size=hp.batch_size)
        
        # 保存文件
        dump_pairs(hp.output_path, 
                   hp.output_fn,
                   entries_a,
                   entries_b,
                   pairs)
    

        # 检测blocker的召回率recall 和 #cand
        
        # pairs = run_blocking(left_dataset, right_dataset, model, hp)

        
        recall, new_size = evaluate_blocking(pairs,hp)
        
        
        if isinstance(recall, list):
            scalars = {}
            for i in range(len(recall)):
                scalars['recall_%d' % i] = recall[i]
                scalars['new_size_%d' % i] = new_size[i]
        else:
            scalars = {'recall': recall,
                        'new_size': new_size}
        
        writer = SummaryWriter(log_dir=hp.logdir)
        # 不再体现
        run_tag = hp.input_path # 数据集的种类
        writer.add_scalars(run_tag+"_tag", scalars, 0)
        print(f"recall={recall}, num_candidates={new_size}")
        
        # for sz, r in zip(new_size, recall):
        #     mlflow.log_metric("recall_%d" % sz, r)
        
        # print(f"epoch {epoch}: recall={recall}, num_candidates={new_size}")
        

