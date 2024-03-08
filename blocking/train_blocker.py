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
    model_names = {'distilbert': 'distilbert-base-uncased',
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
    
    trainset =reader.get_examples(hp.train_fn)
    
    train_dataloader = DataLoader(trainset,
                                  shuffle=True,
                                  batch_size=hp.batch_size)
    
    train_loss = losses.SoftmaxLoss(model=model,
            sentence_embedding_dimension=model\
                    .get_sentence_embedding_dimension(),
            num_labels=2) # lables = [0,1]

    dev_data = reader.get_examples(hp.valid_fn)
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_data,
                                                                     batch_size=hp.batch_size)
    
    warmup_steps = math.ceil(len(train_dataloader) \
            * hp.n_epochs / hp.batch_size * 0.1) #10% of train data for warm-up

    if os.path.exists(hp.model_fn):
        import shutil
        shutil.rmtree(hp.model_fn)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
        evaluator=dev_evaluator,
        epochs=hp.n_epochs,
        evaluation_steps=1000,
        warmup_steps=warmup_steps,
        output_path=hp.model_fn
        )
    
def test(hp):
    reader = Reader() 
    test_samples = reader.get_examples(hp.valid_fn)
    
    model_path = hp.model_fn
    save_path = os.path.join(model_path,'test')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if os.path.exists(model_path):
        model = SentenceTransformer(model_path)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            test_samples, batch_size=hp.batch_size)
        test_evaluator(model, output_path=save_path)
        
    else:
        print('model not found')

 
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fn", type=str, default="../data/er_magellan/Structured/Beer/train.txt")
    parser.add_argument("--valid_fn", type=str, default="../data/er_magellan/Structured/Beer/valid.txt")
    parser.add_argument("--model_fn", type=str, default="./models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--lm", type=str, default='distilbert')
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--save_model", dest="save_model", action="store_true")
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true")
    
    
    hp = parser.parse_args()

    train(hp)
    
    test(hp)
