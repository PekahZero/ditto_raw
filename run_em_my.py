import os
import time

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

datasets = """Dirty/iTunes-Amazon
Structured/Beer
Structured/Fodors-Zagats
Structured/iTunes-Amazon
Structured/Walmart-Amazon
Textual/Abt-Buy""".split('\n')

# special_datasets = {
#     'Structured/Beer': (32, 40),
#     'Structured/iTunes-Amazon': (32, 40),
#     'Structured/Fodors-Zagats': (32, 40),
#     'Dirty/iTunes-Amazon': (32, 15)
# }

special_datasets = {
    # 'Structured/Beer': (32, 40),
    'Structured/iTunes-Amazon': (32, 40)
    # 'Structured/Fodors-Zagats': (32, 40),
    # 'Dirty/iTunes-Amazon': (32, 15)
}


# ops = """swap
# swap
# append_col
# del
# swap
# drop_col
# swap
# swap
# append_col
# drop_col
# drop_col
# swap
# del""".split('\n')




lms = ['roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta',
       'roberta', 'roberta', 'roberta', 'roberta', 'roberta', 'roberta']


# for dataset, lm in zip(datasets, lms):
for dataset in datasets:
    if dataset in special_datasets:
        batch_size, epochs = special_datasets[dataset]
        
        lm = 'roberta'
        cmd = """CUDA_VISIBLE_DEVICES=2 python train_ditto.py \
        --task %s \
        --logdir result_em/ \
        --finetuning \
        --batch_size %d \
        --lr 3e-5 \
        --fp16 \
        --lm %s \
        --n_epochs %d \
        --dk general\
        --da all\
        --save_model\
        --summarize\
        --run_id 0""" % (dataset, batch_size, lm, epochs)\

        
        print(cmd)
        os.system(cmd)
    else:
        batch_size, epochs = 32, 40
        

