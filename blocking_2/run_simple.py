import os

datasets = """Dirty/DBLP-ACM
Dirty/DBLP-GoogleScholar
Dirty/iTunes-Amazon
Dirty/Walmart-Amazon
Dirty/iTunes-Amazon""".split('\n')

epochs_num = [(5,5), (10,5),(20,5)]
# for dataset, lm in zip(datasets, lms):
for i in range(0,6):
    epochs, ssl_epochs = epochs_num[i]

    cmd = """CUDA_VISIBLE_DEVICES=2 python train_bt.py \
    --task_type er_magellan \
    --task Dirty/DBLP-ACM \
    --logdir result_%s_%s/ \
    --ssl_method combined \
    --bootstrap \
    --clustering \
    --multiplier 10 \
    --blocking \
    --k 20 \
    --batch_size 32 \
    --lr 3e-5 \
    --fp16 \
    --lm roberta \
    --n_epochs %d \
    --n_ssl_epochs %d \
    --da cutoff \
    --save_ckpt \
    --run_id 0""" % (str(epochs), str(ssl_epochs), epochs, ssl_epochs)


    print(cmd)
    os.system(cmd)
    
