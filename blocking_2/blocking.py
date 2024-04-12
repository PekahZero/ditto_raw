import os
import argparse
import json
import sys
import numpy as np
import random
import torch
import csv
import jsonlines

sys.path.append('../')

from selfsl.bt_dataset import BTDataset
from selfsl.barlow_twins_simclr import BarlowTwinsSimCLR
from selfsl.block import *
from torch.utils import data
from apex import amp
from sklearn.metrics import recall_score
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer



def load_model(hp):
    """Load the model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BarlowTwinsSimCLR(hp,
                        device=device,
                        lm=hp.lm)
    
    # 不是lm_only,可以使用之前经过fine-tune的model
    if not hp.lm_only:
        # 路径存在问题
        # 不是从ckpt加载
        # -----------barlow_twins_simclr.py------------------
            # if hp.save_ckpt and epoch == num_ssl_epochs:
            #     # create the directory if not exist
            #     directory = os.path.join(hp.logdir, hp.task)
            #     if not os.path.exists(directory):
            #         os.makedirs(directory)

            #     # save the checkpoints for each component
            #     ckpt_path = os.path.join(hp.logdir, hp.task, 'ssl.pt')
            #     config_path = os.path.join(hp.logdir, hp.task, 'config.json')
            #     ckpt = {'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict(),
            #             'epoch': epoch}
            #     torch.save(ckpt, ckpt_path)
        # ----------------------------------------------------
        
        # ----
        ckpt_path = os.path.join(hp.logdir, hp.task, 'ssl.pt')
        # ----
        saved_state = torch.load(ckpt_path,
                    map_location=lambda storage, loc: storage)
        model.load_state_dict(saved_state['model'])

    model = model.to(device)
    if hp.fp16 and 'cuda' in device:
        model = amp.initialize(model, opt_level='O2')

    return model


def tfidf_blocking(pathA, pathB, K=10):
    # read csv
    tableA = []
    tableB = []

    # reader = csv.DictReader(open(pathA))
    # for row in reader:
    #     tableA.append(' '.join(row.values()))

    # reader = csv.DictReader(open(pathB))
    # for row in reader:
    #     tableB.append(' '.join(row.values()))
        
    # 读取txt文件A并将每行的值添加到tableA中
    with open(pathA, 'r') as file:
        lines = file.readlines()
        tableA = [' '.join(line.strip().split()) for line in lines]

    # 读取txt文件B并将每行的值添加到tableB中
    with open(pathB, 'r') as file:
        lines = file.readlines()
        tableB = [' '.join(line.strip().split()) for line in lines]

    corpus = tableA + tableB
    vectorizer = TfidfVectorizer().fit(corpus)

    matA = vectorizer.transform(tableA).toarray()
    matB = vectorizer.transform(tableB).toarray()

    res = blocked_matmul(matA, matB, k=K)
    return res

def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def dump_pairs(out_path,out_fn, entries_a, entries_b, pairs):
    # Dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    outfile_path = os.path.join(out_path,out_fn)
    # txt or json
    if 'txt' in out_fn:
        with open(outfile_path, 'w') as writer:
            for idx_a, idx_b, score in pairs:
                left, right, scr = entries_a[idx_a].strip(), entries_b[idx_b].strip(), str(score)
                # print(left,right,scr)
                writer.write(left+'\t'+right+'\t'+scr+'\n')
    elif 'jsonl' in out_fn:
        with jsonlines.open(outfile_path, mode='w') as writer:
            for idx_a, idx_b, score in pairs:
                # print(idx_a, idx_b, score)
                writer.write([entries_a[idx_a], entries_b[idx_b], str(score)])
    else:
        print('DO NOT SAVE')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="DBLP-ACM")
    parser.add_argument("--task_type", type=str, default="task_type")
    parser.add_argument("--logdir", type=str, default="results/")
    parser.add_argument("--result_logdir", type=str, default="results_blk/")
    # parser.add_argument("--ckpt_path", type=str, default=None) # em_Abt-Buy_da=barlow_twins_id=0_size=300_ssl.pt
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--projector", type=str, default='768')
    parser.add_argument("--lm", type=str, default='roberta')
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--tfidf", dest="tfidf", action="store_true") # if set, apply the baseline blocker
    parser.add_argument("--lm_only", dest="lm_only", action="store_true") # if set, only apply a non-fine-tuned LM
    hp = parser.parse_args()

    # set seed
    seed = hp.run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    # 修改
    # path = input_dir
    # ----------------
    path = '../data/er_magellan/%s' % hp.task

    left_path = os.path.join(path, 'tableA.txt')
    right_path = os.path.join(path, 'tableB.txt')
    # -----------------

    # 使用不同的blocking方法，tableA & tableB --> pairs(dataset)
    # 两种不可兼得，我们希望使用 BT blocking?
    
    if hp.tfidf:
    # if not hp.tfidf:
        
        # tfidf blocking
        # pairs = tfidf_blocking(left_path.replace('.txt', '.csv'),
        #                        right_path.replace('.txt', '.csv'), K=hp.k)
        pairs = tfidf_blocking(left_path,
                               right_path, K=hp.k)
    else:
        # BT blocking
        left_dataset = BTDataset(left_path,
                                 lm=hp.lm,
                                 size=None,
                                 max_len=hp.max_len)

        right_dataset = BTDataset(right_path,
                                  lm=hp.lm,
                                  size=None,
                                  max_len=hp.max_len)

        model = load_model(hp)
        pairs = run_blocking(left_dataset, right_dataset, model, hp)

# get entries
    ent_a = read_txt(left_path)
    ent_b = read_txt(right_path)
    
# get pairs 

#  creat output_file
    
    dump_pairs('output/', 'candidate.jsonl', ent_a, ent_b, pairs)
    dump_pairs('output/', 'candidate.txt', ent_a, ent_b, pairs)

# 评估


    from tensorboardX import SummaryWriter # conda install tensorboardX
    
    writer = SummaryWriter(log_dir=hp.result_logdir)
    blocker_tag = "%s_tfifd=%s_id=%d" % (hp.task, str(hp.tfidf), hp.run_id)
# 如果超参数 hp.k 不存在（为空），则计算所有候选配对的召回率
# 调用 evaluate_pairs 函数计算召回率，并将其存储在变量 recall 中。
# 候选集大小即为配对的数量
    ground_truth, total = read_ground_truth(path)
    
    if hp.k:
        for k in range(1, hp.k+1):
            recall, size = evaluate_pairs(pairs, ground_truth, k=k)
            scalars = {"recall": recall,
                       "new_size": size}
            writer.add_scalars(blocker_tag, scalars, k)
            print('k=%d, recall=%f, new_size=%d' % (k, recall, size))
    else:
        recall = evaluate_pairs(pairs, ground_truth)
        scalars = {"recall": recall,
                   "original_size" : total,
                   "new_size": len(pairs)}
        writer.add_scalars(blocker_tag, scalars, 1)
        print('recall = %f, original_size = %d, new_size = %d' % (recall, total, len(pairs)))


    
    # # dump pairs
    # import pickle
    # pickle.dump(pairs, open('blocking_result.pkl', 'wb'))
    # 将名为 pairs 的对象保存到名为 blocking_result.pkl 的文件
    
    
    # --------------评估---------------
    # with open('blocking_result.pkl', 'rb') as file:
    #     pairs = pickle.load(file)
    #     print(pairs)

        # (627, 1091, 0.712636925726547)
        # 是 (idx1, idx2, confidence)
