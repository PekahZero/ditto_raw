import torch
import random

from torch.utils import data
from transformers import AutoTokenizer

from .augment import Augmenter
from .utils import blocked_matmul

# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'xlnet': 'xlnet-large-cased',
         'bert': 'bert-base-uncased'}

class BTDataset(data.Dataset):
    """Dataset for pre-training"""

    def __init__(self,
                 path,
                 max_len=256,
                 size=None,
                 lm='roberta',
                 da='all'):
        self.tokenizer = AutoTokenizer.from_pretrained(lm_mp[lm])
        self.instances= [] # # 存储数据集中的实例
        self.max_len = max_len # entity 的最大长度限制
        self.size = size # 数据集大小

        for line in open(path):
            self.instances.append(line.strip())

        if size is not None:
            if size > len(self.instances):
                N = size // len(self.instances) + 1
                self.instances = (self.instances * N)[:size] # 若指定了数据集大小，根据情况进行调整 进行重复
            else:
                self.instances = random.sample(self.instances, size)

        self.da = da    # default is random
    # ------------------------------------
        # self.augmenter = Augmenter()
        if da is not None:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None
    # -------------------------------------
        self.ground_truth = None


    def __len__(self):
        """Return the size of the dataset."""
        return len(self.instances)

    def __getitem__(self, idx):
        """Return a tokenized item of the dataset.

        Args:
            idx (int): the index of the item

        Returns:
            List of int: token ID's of the 1st entity
            List of int: token ID's of the 2nd entity
            List of int: token ID's of the two entities combined
            int: the label of the pair (0: unmatch, 1: match)
        """
        # 首先判断有 da 的要求
        if self.da is not None:
            # 返回数据集中的一个经过 tokenization 处理后的项目。 对原来的实体进行编码
            if self.da == 'cutoff':
                # A = B = self.instances[idx]
                A = self.instances[idx]
                # combine with the deletion operator
                B = self.augmenter.augment_sent(A, "del") 
            else:
                A = self.instances[idx]
                B = self.augmenter.augment_sent(A, self.da)
        else:
            A = B = self.instances[idx]
            
        # left
        yA = self.tokenizer.encode(text=A,
                                   max_length=self.max_len,
                                   truncation=True)
        yB = self.tokenizer.encode(text=B,
                                   max_length=self.max_len,
                                   truncation=True)
        return yA, yB

    # 创建标签？
    # 为评估无标签数据集 添加 真实标签
    def create_ground_truth(self, datasets):
        """Add ground truths to the unlabeled set for evaluation.

        Args:
            datasets (List of DMDataset): the (train, valid, test) datasets

        Returns:
            None
        """
        # 为评估无标签数据集添加真实标签
        # 遍历 self.instances 中的每个实例，给每个实例分配一个索引
        #  # 创建一个映射字典，将每个实例映射到一个索引
        
        # self.instance 存储了size=hp.size or 1000 的无标签数据
        mp = {}
        for idx, inst in enumerate(self.instances):
            mp[inst] = idx
            
        
        # 初始化真实标签集合
        self.ground_truth = set([])
        
        for dataset in datasets:
            # 遍历数据集中的每对样本和标签
            for pair, label in zip(dataset.pairs, dataset.labels):
                # 如果标签为1，表示是一对实例
                if int(label) == 1:
                    left, right = pair
                    left = left.strip()
                    right = right.strip()
                    
                    # 检查左右实例是否都在映射中 / 有截取
                    if left in mp and right in mp:
                        left, right = mp[left], mp[right]
                        self.ground_truth.add((left, right))
                        self.ground_truth.add((right, right))
        
        # 打印真实标签集合的长度(所有的数据train+test+valid 在instance之中)
        print(len(self.ground_truth))

    @staticmethod
    def pad(batch):
        """Merge a list of dataset items into a train/test batch

        Args:
            batch (list of tuple): a list of dataset items

        Returns:
            LongTensor: x1 of shape (batch_size, seq_len)
            LongTensor: x2 of shape (batch_size, seq_len).
                        Elements of x1 and x2 are padded to the same length 
                        x1 和 x2 的元素都填充到相同的长度
        """
        #  """将数据集的一批项目合并为训练/测试批次。
        yA, yB = zip(*batch)

        maxlen = max([len(x) for x in yA])
        maxlen = max(maxlen,max([len(x) for x in yB]))

        yA = [xi + [0]*(maxlen - len(xi)) for xi in yA]
        yB = [xi + [0]*(maxlen - len(xi)) for xi in yB]

        return torch.LongTensor(yA), \
               torch.LongTensor(yB)
