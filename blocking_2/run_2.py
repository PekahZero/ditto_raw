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

datasets = """Dirty/DBLP-ACM""".split('\n')

for dataset in datasets:
    batch_size, epochs, ssl_epochs = (64, 50, 3)
    for id in range(0,3):
        cmd = """CUDA_VISIBLE_DEVICES=0 python train_bt.py \
        --task_type er_magellan \
        --task %s \
        --logdir result_blocking_fine/ \
        --batch_size %d \
        --max_len 128 \
        --size 500 \
        --lr 5e-5 \
        --n_epochs %d \
        --lm distilbert \
        --ssl_method combined \
        --n_ssl_epochs %d \
        --da cutoff \
        --num_clusters 90 \
        --projector 768 \
        --run_id %d \
    """ % (dataset, batch_size, epochs, ssl_epochs, id)

        print(cmd)
        os.system(cmd)
    
