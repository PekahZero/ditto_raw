from pathlib import Path
import copy
import argparse
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn.functional as F
import mlflow

from torch import nn, optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils import data
from apex import amp
from tensorboardX import SummaryWriter
from tqdm import tqdm
from .augment import Augmenter
from .bt_dataset import BTDataset
from .dataset import DMDataset
from .block import evaluate_blocking
from .bootstrap import bootstrap, bootstrap_cleaning

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'xlnet': 'xlnet-large-cased',
         'bert': 'bert-base-uncased'}

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# 自监督学习
class BarlowTwinsSimCLR(nn.Module):
    # the encoder is bert+projector
    def __init__(self, hp, device='cuda', lm='roberta'):
        super().__init__()
        self.hp = hp
        self.bert = AutoModel.from_pretrained(lm_mp[lm])
        self.device = device
        hidden_size = 768

        # projector
        sizes = [hidden_size] + list(map(int, hp.projector.split('-')))
        self.projector = nn.Linear(hidden_size, sizes[-1])

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        # a fully connected layer for fine tuning
        #self.fc = torch.nn.Linear(hidden_size * 2, 2)
        if hp.task_type == 'er_magellan':
            self.fc = nn.Linear(sizes[-1] * 2, 2)
        else:
            self.fc = nn.Linear(sizes[-1], 2)

        # contrastive
        self.criterion = nn.CrossEntropyLoss().to(device)


    def info_nce_loss(self, features,
            batch_size,
            n_views,
            temperature=0.07):
        """Copied from https://github.com/sthalles/SimCLR/blob/master/simclr.py
        """
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return logits, labels


    def forward(self, flag, y1, y2, y12, da=None, cutoff_ratio=0.1):
        if flag in [0, 1]:
            # pre-training
            # encode
            batch_size = len(y1)
            y1 = y1.to(self.device) # original
            y2 = y2.to(self.device) # augment
            
            # 如果da == 'cutoff'，
            # 表示采用截断处理方式，会对输入数据进行截断处理，
            # 并使用BERT模型的词嵌入和位置嵌入来构造新的数据y2_word_embeds。
            # 具体操作包括对y2的词嵌入进行修改，
            # 以及对位置嵌入进行采样和相应修改。
            # 最后将y1和y2的词嵌入拼接在一起，得到y_embeds，
            # 再通过BERT模型编码得到表示句子语义的向量z
            if da == 'cutoff':
                seq_len = y2.size()[1]
                y1_word_embeds = self.bert.embeddings.word_embeddings(y1)
                y2_word_embeds = self.bert.embeddings.word_embeddings(y2)

                # modify the word embeddings of y2
                # l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                # s = random.randint(0, seq_len - l - 1)

                # y2_word_embeds[:, s:s+l, :] = 0.0

                # modify the position embeddings of y2
                position_ids = torch.LongTensor([list(range(seq_len))]).to(self.device)
                # position_ids = self.bert.embeddings.position_ids[:, :seq_len]
                pos_embeds = self.bert.embeddings.position_embeddings(position_ids)

                # sample again
                l = random.randint(1, int(seq_len * cutoff_ratio)+1)
                s = random.randint(0, seq_len - l - 1)
                y2_word_embeds[:, s:s+l, :] -= pos_embeds[:, s:s+l, :]

                # merge y1 and y2
                y_embeds = torch.cat((y1_word_embeds, y2_word_embeds))
                z = self.bert(inputs_embeds=y_embeds)[0][:, 0, :]
            else:
                # cat y1 and y2 for faster training
                y = torch.cat((y1, y2))
                z = self.bert(y)[0][:, 0, :]
                z = self.projector(z)
# 如果flag为0，采用SimCLR方法，使用信息最大化的对比损失（info_nce_loss）计算损失值。
# 具体过程包括计算特征向量z的对比损失的logits和labels，然后通过交叉熵损失函数计算总体损失值。
            if flag == 0:
                # simclr
                logits, labels = self.info_nce_loss(z, batch_size, 2)
                loss = self.criterion(logits, labels)
                return loss
# 如果flag为1，采用Barlow Twins方法，
# 首先将z分成两部分z1和z2，然后计算它们之间的经验交叉相关矩阵c。
# 之后根据交叉相关矩阵c，计算损失值。
# 具体包括对交叉相关矩阵的对角线元素和非对角线元素进行平方和计算，
# 并根据超参数scale_loss和lambd进行加权求和得到最终的损失值。

            elif flag == 1:
                # barlow twins
                z1 = z[:batch_size]
                z2 = z[batch_size:]

                # empirical cross-correlation matrix
                c = (self.bn(z1).T @ self.bn(z2)) / (len(z1))

                # sum the cross-correlation matrix between all gpus
                #c.div_(self.hp.batch_size)
                #torch.distributed.all_reduce(c)

                # use --scale-loss to multiply the loss by a constant factor
                # on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.hp.scale_loss)
                on_diag = ((torch.diagonal(c) - 1) ** 2).sum() * self.hp.scale_loss
                # off_diag = off_diagonal(c).pow_(2).sum().mul(self.hp.scale_loss)
                off_diag = (off_diagonal(c) ** 2).sum() * self.hp.scale_loss
                loss = on_diag + self.hp.lambd * off_diag
                return loss
        elif flag == 2:
            # fine tuning
            if self.hp.task_type == 'er_magellan':
                x1 = y1
                x2 = y2
                x12 = y12

                x1 = x1.to(self.device) # (batch_size, seq_len)
                x2 = x2.to(self.device) # (batch_size, seq_len)
                x12 = x12.to(self.device) # (batch_size, seq_len)
                # left+right
                enc_pair = self.projector(self.bert(x12)[0][:, 0, :]) # (batch_size, emb_size)
                #enc_pair = self.bert(x12)[0][:, 0, :] # (batch_size, emb_size)
                batch_size = len(x1)
                # left and right
                enc = self.projector(self.bert(torch.cat((x1, x2)))[0][:, 0, :])
                #enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
                enc1 = enc[:batch_size] # (batch_size, emb_size)
                enc2 = enc[batch_size:] # (batch_size, emb_size)
                # return self.fc(torch.cat((enc1, enc2, (enc1 - enc2).abs()), dim=1)) # .squeeze() # .sigmoid()
                return self.fc(torch.cat((enc_pair, (enc1 - enc2).abs()), dim=1)) # .squeeze() # .sigmoid()
            # else: # cleaning
            #     x1 = y1
            #     x1 = x1.to(self.device) # (batch_size, seq_len)
            #     enc = self.projector(self.bert(x1)[0][:, 0, :]) # (batch_size, emb_size)
            #     return self.fc(enc)

def evaluate(model, iterator, threshold=None, ec_task=None, dump=False):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class
        ec_task (string, optional): if set, evaluate error correction
        dump (boolean, optional): if true, dump the test results

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
            if len(batch) == 4:
                x1, x2, x12, y = batch
                x1 = x1.cuda()
                x2 = x2.cuda()
                x12 = x12.cuda()
                # y = y.cuda()
                logits = model(2, x1, x2, x12)
            else:
                x, y = batch
                x = x.cuda()
                # y = y.cuda()
                logits = model(2, x, None, None)

            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        if dump:
            import pickle
            pickle.dump(pred, open('test_results.pkl', 'wb'))
            mlflow.log_artifact('test_results.pkl')
            

        f1 = metrics.f1_score(all_y, pred, zero_division=0)
        p = metrics.precision_score(all_y, pred, zero_division=0)
        r = metrics.recall_score(all_y, pred, zero_division=0)
        # error correction
        if ec_task:
            path = 'data/cleaning/%s/test.txt.ec' % ec_task
            # new_pred = pred.copy()
            current = 0
            indices = open(path).readlines()
            indices = [idx[:-1].split('\t') for idx in indices]
            tp, fp, fn = 0.0, 0.0, 0.0

            while current < len(pred):
                start = current
                while current + 1 < len(pred) and \
                      indices[current][:2] == indices[current + 1][:2]:
                    current += 1
                
                max_idx = -1
                max_prob = 0.0
                for idx in range(start, current+1):
                    if all_probs[idx] > max_prob:
                        max_prob = all_probs[idx]
                        max_idx = idx
                    
                    # new_pred[idx] = 0

                # predict
                original, res, ground_truth = indices[max_idx][2:]
                correction = res if pred[max_idx] == 1 else original

                # if start == 0:
                #     print(original, res, ground_truth)

                if original != ground_truth and correction == ground_truth:
                    tp += 1
                if original == ground_truth and correction != ground_truth:
                    fp += 1
                if original != ground_truth and correction != ground_truth:
                    fn += 1

                current += 1

            ec_f1 = tp / (tp + (fp + fn) / 2 + 1e-16) # metrics.f1_score(all_y, new_pred, zero_division=0)
            # print(tp, fp, fn)
            return f1, p, r, ec_f1
        else:
            return f1, p, r
    else:
        best_th = 0.5
        p = r = f1 = 0.0 # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred, zero_division=0)
            new_p = metrics.precision_score(all_y, pred, zero_division=0)
            new_r = metrics.recall_score(all_y, pred, zero_division=0)

            if new_f1 > f1:
                f1 = new_f1
                p = new_p
                r = new_r
                best_th = th

        return f1, p, r, best_th


# def creat_iter_batches(u_set, clusters, batch_size):
#     N = len(u_set)
#     indices = []
#     random.shuffle(clusters)
#     # 随机洗牌簇的顺序
    
#     for c in clusters: # 随机洗牌每个簇中的索引
#         random.shuffle(c)
#         indices += c
#     # 遍历每个簇，随机洗牌其中的索引
    
#     batch = []
#     # 根据洗牌后的索引构建批次数据，并进行填充
#     for i, idx in enumerate(indices):
#         # batch.append(u_set[i]) # 将数据集中的样本按照洗牌后的顺序加入批次中 u_set[i] ?? 
#         batch.append(u_set[idx]) #  u_set.instance[idx]
#         if len(batch) == batch_size or i == N - 1: # 如果批次数据达到指定大小或者遍历完所有数据
#             yield u_set.pad(batch) # 引用？
#             batch.clear()
            
# # 生成批次数据，使得相似的条目被分组在一起
def create_batches(u_set, batch_size, n_ssl_epochs, num_clusters=50):
    # 对无标签数据集进行基于TF-IDF向量化的特征提取
    N = len(u_set)
    tfidf = TfidfVectorizer().fit_transform(u_set.instances)

    # 使用KMeans算法将数据集划分为指定数量的簇
    kmeans = KMeans(n_clusters=num_clusters,n_init=10).fit(tfidf)

    # 将每个样本分配到对应的簇中
    # 生成 num_clusters 个簇
    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(idx) 
        
    # 计算簇内的假阴性率（False Negative Rate）
    # 即在同一簇中被误判为不相似的实例对的比例。
    # report FNR within clusters
    if u_set.ground_truth is not None:
        total = 0
        matches = 0
        for cluster in clusters:
            for idx1 in cluster:
                for idx2 in cluster:
                    if idx1 == idx2:
                        continue
                    total += 1
                    if (idx1, idx2) in u_set.ground_truth:
                        matches += 1
        fnr = matches / total  # 计算假阴性率
        mlflow.log_metric("clustering_FNR", fnr)
    
    def create_iter():
        indices = []
        random.shuffle(clusters)
        # 随机洗牌簇的顺序
        
        for c in clusters: # 随机洗牌每个簇中的索引
            random.shuffle(c)
            indices += c
        # 遍历每个簇，随机洗牌其中的索引
        
        batch = []
        # 根据洗牌后的索引构建批次数据，并进行填充
        for i, idx in enumerate(indices):
            # batch.append(u_set[i]) # 将数据集中的样本按照洗牌后的顺序加入批次中 u_set[i] ?? 
            batch.append(u_set[idx]) #  u_set.instance[idx]
            if len(batch) == batch_size or i == N - 1: # 如果批次数据达到指定大小或者遍历完所有数据
                yield u_set.pad(batch) # 引用？
                batch.clear()
                
    for _ in range(n_ssl_epochs):
        # print("ssl_epoch", n_ssl_epochs)
        yield create_iter()
    
# 生成批次数据，使得相似的条目被分组在一起




# 单步执行 train_step
def selfsl_step(train_nolabel_iter, train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_nolabel_iter (Iterator): the unlabeled data loader
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    
    loss_list = []
    # train Barlow Twins or SimCLR
    
    for i, batch in enumerate(train_nolabel_iter):
        yA, yB = batch
        yA = yA.cuda()
        yB = yB.cuda()
        optimizer.zero_grad()
        # loss = model(i%2, yA, yB, [], da=hp.da)
        if hp.ssl_method == 'simclr':
            # simclr
            loss = model(0, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
        elif hp.ssl_method == 'barlow_twins':
            # barlow twins
            loss = model(1, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
        else:
            # combined
            alpha = 1 - hp.alpha_bt
            loss1 = model(0, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            loss2 = model(1, yA, yB, [], da=hp.da, cutoff_ratio=hp.cutoff_ratio)
            loss = alpha * loss1 + (1 - alpha) * loss2

        # print('loss:' , loss.item())
        loss_list.append(loss.item())
        
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        # loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        if i % 10 == 0: # monitoring
            print(f"    step: {i}, loss: {loss.item()}")
        #print(f"    step: {i}, loss: {loss.item()}")
        
        del loss
        
    return sum(loss_list)/len(loss_list)


def fine_tune_step(train_iter, model, optimizer, scheduler, hp):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    loss_list = []
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        if len(batch) == 4:

            x1, x2, x12, y = batch
            
            x1 = x1.cuda()
            x2 = x2.cuda()
            x12 = x12.cuda()
            y = y.cuda()

            prediction = model(2, x1, x2, x12)
        else:
            x, y = batch
            x = x.cuda()
            y = y.cuda()
            prediction = model(2, x, None, None)

        loss = criterion(prediction, y.to(model.device))
        # loss = criterion(prediction, y.float().to(model.device))
        if hp.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        # loss.backward()
        
        optimizer.step()
        scheduler.step()
        loss_list.append(loss.item())
        if i % 10 == 0: # monitoring
            print(f"    fine tune step: {i}, loss: {loss.item()}")
        del loss
    return sum(loss_list)/len(loss_list)


def train(trainset_nolabel, trainset, validset, testset, run_tag, hp):
    """Train and evaluate the model

    Args:
        trainset_nolabel (BTDataset) : 
        trainset (DMDataset): the training set
        validset (DMDataset): the validation set
        testset (DMDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    
    # 数据准备
    padder = trainset.pad
    # create the DataLoaders
    
    
    # 是否使用聚类：
    # 不使用聚类
    if not hp.clustering:
        # 用于无标签数据的数据加载器
        train_nolabel_iter = data.DataLoader(dataset=trainset_nolabel,
                                             batch_size=hp.batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=trainset_nolabel.pad)
    # 使用聚类----------------------------------------------------------------------------
    else:
        # 根据超参数创建批次
        train_nolabel_iters = create_batches(trainset_nolabel,
                                            hp.batch_size,
                                            hp.n_ssl_epochs,
                                            num_clusters=hp.num_clusters)

    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size//2,    # half of barlow twins'
                                 shuffle=True, # TODO: do not shuffle for data cleaning task
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=validset.pad)
    test_iter = data.DataLoader(dataset=testset,
                                 batch_size=hp.batch_size*16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=testset.pad)
    all_pairs_iter = None
    
    # if all_pairs.txt is avaialble
     # 如果all_pairs.txt可用
    # all_pairs_path = 'data/%s/%s/all_pairs.txt' % (hp.task_type, hp.task)
    # if os.path.exists(all_pairs_path):
    #     all_pair_set = DMDataset(all_pairs_path,
    #                      lm=hp.lm,
    #                      size=None,
    #                      max_len=hp.max_len)

    #     all_pairs_iter = data.DataLoader(dataset=all_pair_set,
    #                                  batch_size=hp.batch_size*16,
    #                                  shuffle=False,
    #                                  num_workers=0,
    #                                  collate_fn=testset.pad)
    # else:
    #     all_pairs_iter = None

    # 初始化模型、优化器和LR调度器
    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # barlow twins
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)

    # model = model.cuda() # # 将模型移动到GPU
    model = model.cuda(device=device)
    
    optimizer = AdamW(model.parameters(), lr=hp.lr,no_deprecation_warning=True)
    
    if hp.fp16:
        opt_level = 'O2' if hp.ssl_method == 'combined' else 'O2'
        model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    # number of steps
    # 步数
    num_ssl_epochs = hp.n_ssl_epochs
    num_ssl_steps = len(trainset_nolabel) // hp.batch_size * num_ssl_epochs
    
    num_finetune_steps = len(trainset) // (hp.batch_size // 2) * (hp.n_epochs - num_ssl_epochs)
    if num_finetune_steps < 0:
        num_finetune_steps = 0
        
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_ssl_steps+num_finetune_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)
    start_epoch = 1

    # load checkpoint if saved
    # 加载模型------------------------------------------------------
    if hp.use_saved_ckpt:
        ckpt_path = os.path.join(hp.logdir, hp.task, 'ssl.pt')
        # config_path = os.path.join(hp.logdir, hp.task, 'config.json')
        ckpt = torch.load(ckpt_path)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    best_dev_f1 = best_test_f1 = 0.0
    for epoch in range(start_epoch, hp.n_epochs+1):
        # bootstrap the training set
        # bootstrap训练集-----------------------------------------------------
        #  (hp.bootstrap or hp.zero && task==em
        if epoch == num_ssl_epochs + 1 and hp.task_type in ['er_magellan'] and (hp.bootstrap or hp.zero):

            new_trainset, TPR, TNR, FPR, FNR = bootstrap(model, hp)
            # logging
            writer.add_scalars(run_tag,
                                {'new_size': len(new_trainset),
                                'TPR': TPR,
                                'TNR': TNR,
                                'FPR': FPR,
                                'FNR': FNR}, epoch)

            new_size = len(new_trainset)
            for variable in ["new_size", "TPR", "TNR", "FPR", "FNR"]:
                mlflow.log_metric(variable, eval(variable))

            # 改变训练集
            train_iter = data.DataLoader(dataset=new_trainset,
                                         batch_size=hp.batch_size//2,    # half of barlow twins'
                                         shuffle=True,
                                         num_workers=0,
                                         collate_fn=padder)

        # train-------------------------------------------------------------------------
        print(f"epoch {epoch}")
        
        # pre-train ------------------------------------
        if epoch <= num_ssl_epochs:
            model.train()
            if hp.clustering:
                train_nolabel_iter = next(train_nolabel_iters)
            loss_avr = selfsl_step(train_nolabel_iter, train_iter, model, optimizer, scheduler, hp)
            scalars = {"ssl_loss": loss_avr}
            writer.add_scalars(run_tag, scalars, epoch)
            
            # # logging blocking--------------------
            # # k=20 则有20个
            # if hp.blocking:
            #     # 返回的 recal_score，数据集大小
            #     recall, new_size = evaluate_blocking(model, hp)
                
            #     if isinstance(recall, list):
            #         scalars = {}
            #         for i in range(len(recall)):
            #             scalars['recall_%d' % i] = recall[i]
            #             scalars['new_size_%d' % i] = new_size[i]
            #     else:
            #         scalars = {'recall': recall,
            #                    'new_size': new_size}
            #     # 不再体现
            #     writer.add_scalars(run_tag+'blocking', scalars, epoch)
                
            #     for sz, r in zip(new_size, recall):
            #         mlflow.log_metric("recall_%d" % sz, r)
                
            #     print(f"epoch {epoch}: recall={recall}, num_candidates={new_size}")
                
        # 微调------------------
        else:
            model.train()
            loss_avr = fine_tune_step(train_iter, model, optimizer, scheduler, hp)
            scalars = {"finetune_loss": loss_avr}
            writer.add_scalars(run_tag, scalars, epoch)

        # eval
        model.eval()
        dev_f1, dev_p, dev_r, th = evaluate(model, valid_iter)
        test_f1, test_p, test_r = evaluate(model, test_iter, threshold=th, dump=True)
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            
            # 更新模型
            if hp.save_ckpt :
                # create the directory if not exist
                directory = os.path.join(hp.logdir, hp.task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(hp.logdir, hp.task, 'ssl.pt')
                # config_path = os.path.join(hp.logdir, hp.task, 'config.json')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)
                
        print(f"epoch {epoch}: dev_f1={dev_f1}, test_f1={test_f1}, best_f1={best_test_f1}")
        print(f"epoch {epoch}: dev_p={dev_p}, dev_r={dev_r}, test_p={test_p}, test_r={test_r}")

        # run on all pairs
        if epoch == hp.n_epochs and all_pairs_iter is not None:
            evaluate(model, all_pairs_iter, threshold=th, dump=True)
            # evaluate(model, test_iter, threshold=th, dump=True)

        # logging
        scalars = {'dev_f1': dev_f1,
                    'dev_p': dev_p,
                    'dev_r': dev_r,
                    'test_f1': test_f1,
                    'test_p': test_p,
                    'test_r': test_r,
                    'best_f1' : best_test_f1,
                    'th':th,
                    'finetune_loss': loss_avr}
        for variable in ["dev_f1", "dev_p", "dev_r", "test_f1", "test_p", "test_r","best_test_f1","th","loss_avr"]:
            mlflow.log_metric(variable, eval(variable))
        writer.add_scalars(run_tag, scalars, epoch)

        # 保存checkpoint--------------------------------------
        # saving checkpoint at the last ssl step 
        if hp.save_ckpt and epoch == num_ssl_epochs:
            # create the directory if not exist
            directory = os.path.join(hp.logdir, hp.task+'_ssl')
            if not os.path.exists(directory):
                os.makedirs(directory)

            # save the checkpoints for each component
            ckpt_path = os.path.join(hp.logdir, hp.task+'_ssl', 'ssl.pt')
            config_path = os.path.join(hp.logdir, hp.task+'_ssl', 'config.json')
            ckpt = {'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch}
            torch.save(ckpt, ckpt_path)
        
        # -------------------------------------------------------
        # check if learning rate drops to 0
        if scheduler.get_last_lr()[0] < 1e-9:
            scalars = {'lr_to_0': epoch}
            writer.add_scalars(run_tag, scalars,epoch)
            mlflow.log_metric('lr_to_0', epoch)
            print("The learning rate drops to 0")
            # break
        
        # 清除缓存
        # torch.cuda.empty_cache()
        
        if hp.blocking:
            # 返回的 recal_score，数据集大小
            recall, new_size = evaluate_blocking(model, hp)
            
            if isinstance(recall, list):
                scalars = {}
                for i in range(len(recall)):
                    scalars['recall_%d' % i] = recall[i]
                    scalars['new_size_%d' % i] = new_size[i]
            else:
                scalars = {'recall': recall,
                            'new_size': new_size}
            # 不再体现
            writer.add_scalars(run_tag+'blocking', scalars, epoch)
            
            for sz, r in zip(new_size, recall):
                mlflow.log_metric("recall_%d" % sz, r)
            
            print(f"epoch {epoch}: recall={recall}, num_candidates={new_size}")
        
    writer.close()
