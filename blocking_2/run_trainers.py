import os

datasets = """Dirty/DBLP-ACM
Dirty/DBLP-GoogleScholar
Dirty/iTunes-Amazon
Dirty/Walmart-Amazon
Structured/Amazon-Google
Structured/Beer
Structured/DBLP-ACM
Structured/DBLP-GoogleScholar
Structured/Fodors-Zagats
Structured/iTunes-Amazon
Structured/Walmart-Amazon
Textual/Abt-Buy""".split('\n')

# datasets = """Dirty/DBLP-ACM""".split('\n')

for dataset in datasets:
    batch_size, epochs, ssl_epochs = (64, 50, 3)

    cmd = """CUDA_VISIBLE_DEVICES=0 python train_bt.py \
    --task_type er_magellan \
    --task %s \
    --logdir result_blk/ \
    --run_id 0 \
    --batch_size %d \
    --max_len 128 \
    --size 500 \
    --lr 5e-5 \
    --n_epochs %d \
    --lm roberta \
    --ssl_method combined \
    --n_ssl_epochs %d \
    --da cutoff \
    --clustering \
    --num_clusters 90 \
    --bootstrap \
    --multiplier 8 \
    --projector 4096 \
    --blocking \
    --k 20 \
    --save_ckpt \
""" % (dataset, batch_size, epochs, ssl_epochs)

    print(cmd)
    os.system(cmd)
    
