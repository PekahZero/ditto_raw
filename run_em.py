import os
import time


lm = 'roberta'
cmd = """CUDA_VISIBLE_DEVICES=2 python train_ditto.py \
--task er_magellan \
--logdir result_em/ \
--finetuning \
--batch_size 32 \
--lr 3e-5 \
--fp16 \
--lm roberta \
--n_epochs 50 \
--dk general \
--da all \
--save_model \
--summarize \
--run_id 0"""\

print(cmd)
os.system(cmd)

        

