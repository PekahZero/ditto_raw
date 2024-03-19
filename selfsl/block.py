import os
import numpy as np
import random
import torch
import csv

from .bt_dataset import BTDataset
from .utils import blocked_matmul

from torch.utils import data
from sklearn.metrics import recall_score
from tqdm import tqdm

# 数据集编码
# model：BT/SimCLR 模型用于编码
def encode_all(dataset, model, hp):
    """Encode all records using to the model.
    
    对数据集中的所有记录使用模型进行编码。

    Args:
        dataset (BTDataset): 要编码的数据集
        model (BarlowTwinsSimCLR): BT/SimCLR 模型用于编码
        hp (Namespace): 超参数

    Returns:
        np.array: 数据集的编码表示
    """
    iterator = data.DataLoader(dataset=dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=0,
                               collate_fn=dataset.pad)

    all_encs = []
    with torch.no_grad():
        for batch in tqdm(iterator):
            x, _ = batch
            x = x.to(model.device)
            if hasattr(hp, "lm_only") and hp.lm_only:
                enc = model.bert(x)[0][:, 0, :]
            else:
                enc = model.bert(x)[0][:, 0, :]
                # enc = model.projector(enc)
            all_encs += enc.cpu().numpy().tolist()

    res = np.array(all_encs)
    res = [v / np.linalg.norm(v) for v in res] # 对编码表示进行归一化处理
    return res




def run_blocking(left_dataset, right_dataset, model, hp):
    """Run the Barlow Twins blocking method.

    Args:
        left_dataset (BTDataset): the left table
        right_dataset (BTDataset): the right table
        model (BarlowTwinsSimCLR): the BT/SimCLR model
        hp (Namespace): hyper-parameters

    Returns:
        list of tuple: the list of candidate pairs
    """
    
    # encode both datasets
    # 对两个数据集进行编码
    mata = encode_all(left_dataset, model, hp)
    matb = encode_all(right_dataset, model, hp)

    # matmul to compute similarity
    # 使用矩阵乘法计算相似度并找到候选配对
    # 选取k个
    pairs = blocked_matmul(mata, matb,
                           threshold=hp.threshold,
                           k=hp.k,
                           batch_size=hp.batch_size)
    return pairs


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
    # csv文件 预先存在？？？？------原网站有？？？
    # 创建的
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


def evaluate_blocking(model, hp):
    """Evaluate an embedding model for blocking.

    Args:
        model (BarlowTwinsSimCLR): the embedding model
        hp (NameSpace): hyper-parameters

    Returns:
        List of float: the list of recalls
        List of int: the list of candidate set sizes
    """
    path = '../data/er_magellan/%s' % hp.task

    left_path = os.path.join(path, 'tableA.txt')
    right_path = os.path.join(path, 'tableB.txt')

    # BT blocking
    print('encode left')
    left_dataset = BTDataset(left_path,
                             lm=hp.lm,
                             size=None,
                             max_len=hp.max_len)

    print('encode right')
    right_dataset = BTDataset(right_path,
                              lm=hp.lm,
                              size=None,
                              max_len=hp.max_len)

    print('blocked MM:')
    pairs = run_blocking(left_dataset, right_dataset, model, hp)

    print('Read ground truth')
    ground_truth, total = read_ground_truth(path)

# 如果超参数 hp.k 不存在（为空），则计算所有候选配对的召回率
# 调用 evaluate_pairs 函数计算召回率，并将其存储在变量 recall 中。
# 候选集大小即为配对的数量
    if hp.k:
        recalls, sizes = [], []
        for k in tqdm(range(1, hp.k+1)):
            recall, size = evaluate_pairs(pairs, ground_truth, k)
            recalls.append(recall)
            sizes.append(size)
        return recalls, sizes
    else:
        recall = evaluate_pairs(pairs, ground_truth)
        return recall, len(pairs)


# 以上代码实现了一个用于数据匹配的模型评估框架，具体包括以下几个函数：

# encode_all(dataset, model, hp): 该函数用于对数据集中的所有记录进行编码。通过将数据集分成批次加载，使用给定的模型对每个记录进行编码，并将编码结果保存在 all_encs 中。最后对编码结果进行归一化处理并返回。

# run_blocking(left_dataset, right_dataset, model , hp): 运行 Barlow Twins 阻塞方法。首先对左右两个数据集进行编码，然后通过阻塞矩阵乘法计算相似度，根据给定的阈值和 k 值返回候选匹配对列表。

# read_ground_truth(path): 从训练/验证/测试集中读取地面真相匹配。遍历每个数据集文件，将标签为 1 的匹配对添加到结果列表中，并返回结果列表以及原始匹配/非匹配总数。

# evaluate_pairs(pairs, ground_truth, k=None): 计算给定的匹配对列表与地面真相匹配的召回率。可选择性地只考虑前 k 个最佳匹配对。根据匹配情况计算召回率并返回。

# evaluate_blocking(model, hp): 评估嵌入模型的阻塞效果。首先加载左右数据集，然后运行阻塞方法得到匹配对列表。接着读取地面真相匹配，最后计算匹配对的召回率或者根据 k 值计算不同 k 下的召回率。返回召回率和匹配对大小等信息。

# 这些函数共同构成了一个数据匹配模型的评估流程，用于评估模型在数据匹配任务上的性能表现。