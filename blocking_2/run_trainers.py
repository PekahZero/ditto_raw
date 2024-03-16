import os

# datasets = """Dirty/DBLP-ACM
# Dirty/DBLP-GoogleScholar
# Dirty/iTunes-Amazon
# Dirty/Walmart-Amazon
# Structured/Amazon-Google
# Structured/Beer
# Structured/DBLP-ACM
# Structured/DBLP-GoogleScholar
# Structured/Fodors-Zagats
# Structured/iTunes-Amazon
# Structured/Walmart-Amazon
# Textual/Abt-Buy""".split('\n')

datasets = """Dirty/DBLP-ACM
Dirty/DBLP-GoogleScholar
Dirty/iTunes-Amazon
Dirty/Walmart-Amazon
Dirty/iTunes-Amazon
Structured/Amazon-Google
Structured/Beer
Structured/DBLP-ACM
Structured/DBLP-GoogleScholar
Structured/Fodors-Zagats
Structured/iTunes-Amazon
Structured/Walmart-Amazon
Textual/Abt-Buy""".split('\n')

# batch_size  epoch 

special_datasets = { 'Dirty/iTunes-Amazon': (32, 30, 25) }
# special_datasets = {
#     # 'Structured/Beer': (32, 50),
#     # 'Structured/iTunes-Amazon': (32, 50),
#     # 'Structured/Fodors-Zagats': (32, 50),
#     'Dirty/iTunes-Amazon': (32, 30, 25)}

# for dataset, lm in zip(datasets, lms):
for dataset in datasets:
    if dataset in special_datasets:
        batch_size, epochs, ssl_epochs = special_datasets[dataset]
    else:
        batch_size, epochs, ssl_epochs = (32, 50, 10)

    cmd = """CUDA_VISIBLE_DEVICES=2 python train_bt.py \
    --task_type er_magellan \
    --task %s \
    --logdir result_blk/ \
    --ssl_method combined \
    --bootstrap \
    --clustering \
    --multiplier 10 \
    --blocking \
    --k 20 \
    --batch_size %d \
    --lr 3e-5 \
    --fp16 \
    --lm roberta \
    --n_epochs %d \
    --n_ssl_epochs %d \
    --da all \
    --save_ckpt \
    --run_id 0""" % (dataset, batch_size, epochs, ssl_epochs)


    print(cmd)
    os.system(cmd)
    
