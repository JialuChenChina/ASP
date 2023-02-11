
#!/bin/sh


python train.py \
    --dataset=cora  \
    --SEED=0 \
    --K=70 \
    --lr_p=0.001 \
    --lr_m=0.01 \
    --wd_p=0.0001 \
    --wd_m=0.0 \
    --num_epochs=500 \
    --num_hidden=256 \
    --tau1=1.2 \
    --l1=1.0 \
    --l2=0.6 \
    --metric='jaccard' \
    --optimizer='adam' \
    --drop_edge_rate=0.1 \
    --drop_kg_edge_rate=0.2



