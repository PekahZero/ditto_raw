import torch
import numpy as np
import sklearn.metrics as metrics
import mlflow
import pickle

from tqdm import tqdm


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
    # 每行在matb中与mata中相似度最高的前k个向量
    # 并将它们的索引以及相似度加入到结果列表中
    mata = np.array(mata)
    matb = np.array(matb)
    results = []
    # 如果指定了k，函数将返回每行中与另一个矩阵中最相似的前k个向量。
    # 如果指定了threshold，函数将返回所有余弦相似度大于等于阈值的向量对。
    for start in range(0, len(matb), batch_size):
        # mata . block.T()
        block = matb[start:start+batch_size] # 
        sim_mat = np.matmul(mata, block.transpose()) # np.matmul 点积 # 相似度矩阵
        if k is not None:
            indices = np.argpartition(-sim_mat, k, axis=0) 
            # 找出sim_mat中每列（axis=0）前k个最大元素的索引
            # 因为这里用的是-sim_mat，所以实际上找出的是前k个最小值的索引。
            for row in indices[:k]: 
                # 这里遍历了indices数组的前k行，也就是针对每一列在matb中找到了与mata中相似度最高的k个向量的索引。
                for idx_b, idx_a in enumerate(row):
                    idx_b += start
                    results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
        elif threshold is not None:
            indices = np.argwhere(sim_mat >= threshold)
            for idx_a, idx_b in indices:
                idx_b += start
                results.append((idx_a, idx_b, sim_mat[idx_a][idx_b-start]))
    return results


def evaluate(model, iterator, threshold=None):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            if model.task_type == 'er_magellan':
                x1, x2, x12, y = batch
                x1 = x1.cuda()
                x2 = x2.cuda()
                x12 = x12.cuda()
                # y = y.cuda()
                logits = model(x1, x2, x12)
            else:
                x, y = batch
                x = x.cuda()
                # y = y.cuda()
                logits = model(x)

            # print(probs)
            probs = logits.softmax(dim=1)[:, 1]

            # print(logits)
            # pred = logits.argmax(dim=1)
            all_probs += probs.cpu().numpy().tolist()
            # all_p += pred.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]

        # dump the results
        pickle.dump(pred, open('test_results.pkl', 'wb'))
        mlflow.log_artifact('test_results.pkl')

        f1 = metrics.f1_score(all_y, pred)
        p = metrics.precision_score(all_y, pred)
        r = metrics.recall_score(all_y, pred)
        return f1, p, r
    else:
        best_th = 0.5
        p = r = f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            new_p = metrics.precision_score(all_y, pred)
            new_r = metrics.recall_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                p = new_p
                r = new_r
                best_th = th

        return f1, p, r, best_th
