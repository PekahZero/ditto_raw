import os
import argparse
import json
import sys
import math
import torch

sys.path.insert(0, "sentence-transformers")

from sentence_transformers.readers import InputExample
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator,LabelAccuracyEvaluator

from torch.utils.data import DataLoader
# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli.py
from tensorboardX import SummaryWriter
from blocker import *

class Reader:
    """A simple reader class for the matching datasets.
    """
    def __init__(self):
        self.guid = 0

    def get_examples(self, fn):
        examples = []
        for line in open(fn):
            sent1, sent2, label = line.strip().split('\t')
            examples.append(InputExample(guid=self.guid,
                texts=[sent1, sent2],
                label=int(label)))
            self.guid += 1
        return examples

def train(hp):
    """Train the advanced blocking model
    Store the trained model in hp.model_fn.

    Args:
        hp (Namespace): the hyperparameters

    Returns:
        None
    """
    # define model
    model_names = { 'roberta': 'roberta-base',
                    'distilbert': 'distilbert-base-uncased',
                    'bert': 'bert-base-uncased',
                    'albert': 'albert-base-v2' }

    # # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(model_names[hp.lm])
    
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# load the training data
    reader = Reader() 
    # example：List: [InputExample]
    # trainset = SentencesDataset(examples=reader.get_examples(hp.train_fn),
    #                             model=model)
    
    train_fn = os.path.join(hp.data_fn,"train.txt")
    valid_fn = os.path.join(hp.data_fn,"valid.txt")
    
    trainset =reader.get_examples(train_fn)
    
    train_dataloader = DataLoader(trainset,
                                  shuffle=True,
                                  batch_size=hp.batch_size)
    
    train_loss = losses.SoftmaxLoss(model=model,
            sentence_embedding_dimension=model\
                    .get_sentence_embedding_dimension(),
            num_labels=2) # lables = [0,1]

    dev_data = reader.get_examples(valid_fn)
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_data,
                                                                     batch_size=hp.batch_size)
    
    warmup_steps = math.ceil(len(train_dataloader) \
            * hp.n_epochs / hp.batch_size * 0.1) #10% of train data for warm-up

    task_tag = hp.data_fn.split('/')[-2] + "_" + hp.data_fn.split('/')[-1]
    
    model_fn = os.path.join(hp.model_fn,task_tag) 
    if os.path.exists(model_fn):
        import shutil
        shutil.rmtree(model_fn)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=hp.n_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=model_fn
        )
    
def test(hp):
    reader = Reader() 
    valid_fn = os.path.join(hp.data_fn,"valid.txt")
    test_samples = reader.get_examples(valid_fn)
    
    # 保存test的得到的数据--> csv文件
    task_tag = hp.data_fn.split('/')[-2] + "_" + hp.data_fn.split('/')[-1]
    
    if not os.path.exists(hp.output_path):
        os.mkdir(hp.output_path)
    
    run_tag = "epoch_%d_k_%d"% (hp.n_epochs,hp.k)
    folder = os.path.join(hp.output_path, run_tag)
    if not os.path.exists(folder):
        os.mkdir(folder)
    save_path = os.path.join(folder, task_tag)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    model_fn = os.path.join(hp.model_fn,task_tag) 
    if os.path.exists(model_fn):
        model = SentenceTransformer(model_fn)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            test_samples, batch_size=hp.batch_size)
        # test_evaluator(model, output_path=save_path)
        
        model.evaluate(test_evaluator, output_path=save_path)
    else:
        print('model not found')
        return 

    
    if hp.blocking:
            
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
            dump_pairs(save_path, 
                        hp.output_fn,
                        entries_a,
                        entries_b,
                        pairs)
        
        # 返回的 recal_score，数据集大小
        recall, new_size = evaluate_blocking(pairs, hp)
        
        if isinstance(recall, list):
            scalars = {}
            for i in range(len(recall)):
                scalars['recall_%d' % i] = recall[i]
                scalars['new_size_%d' % i] = new_size[i]
        else:
            scalars = {'recall': recall,
                        'new_size': new_size}
        # --------------------------------------


        run_tag = "epoch_%d_k_%d"% (hp.n_epochs,hp.k)
        log_dir = os.path.join(hp.logdir, task_tag)
        writer = SummaryWriter(log_dir=log_dir)

        writer.add_scalars(run_tag, scalars, 0)
        print(f"recall={recall}, num_candidates={new_size}")
 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    # ------模型训练--------------
    parser.add_argument("--data_fn", type=str, default="../data/er_magellan/Structured/Beer")
        # train_fn = os.path.join(hp.data_fn,"train.txt")
        # valid_fn = os.path.join(hp.data_fn,"valid.txt")
    # parser.add_argument("--train_fn", type=str, default="../data/er_magellan/Structured/Beer/train.txt")
    # parser.add_argument("--valid_fn", type=str, default="../data/er_magellan/Structured/Beer/valid.txt")
    parser.add_argument("--model_fn", type=str, default="./models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1)
    # parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    
    # 新增对于model的测试
    parser.add_argument("--blocking", dest="blocking", action="store_true")
    
    #------blocker----------------
    parser.add_argument("--input_path", type=str, default="../data/er_magellan/Structured/Beer")
    parser.add_argument("--left_fn", type=str, default='tableA.txt')
    parser.add_argument("--right_fn", type=str, default='tableB.txt')
    parser.add_argument("--output_path", type=str, default='./output/')
    parser.add_argument("--output_fn", type=str, default='candidates.jsonl')
    parser.add_argument("--logdir", type=str, default='./tfevents')
    parser.add_argument("--k", type=int, default=10) # top-k
    parser.add_argument("--threshold", type=float, default=None) # 0.6
    #----------------------------
    
# logdir 无用
# save_model 无用： 一定会保存model到model_fn
    hp = parser.parse_args()

    train(hp)
    
    test(hp) #包含blocking的判断
    
        
        
        
