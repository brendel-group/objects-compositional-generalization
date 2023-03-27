How to run SlotAttention training:
```
python main.py --batch_size=64 --weight_decay=0.001 
--lr=0.001 --lr_scheduler_step=100 --warmup=True 
--unsupervised_mode=True --use_consistency_loss=False
--n_slot_latents=16 
--reconstruction_term_weight=1.0
```

How to run SlotAttention with consistency loss training:
```
python main.py --batch_size=128 --weight_decay=0.001 
--lr=0.001 --lr_scheduler_step=50 --warmup=True 
--unsupervised_mode=True --use_consistency_loss=True
--detached_latents=True --extended_consistency_loss=True
--consistency_scheduler=True  --consistency_scheduler_step=200
--n_slot_latents=16 
--reconstruction_term_weight=1.0 --consistency_term_weight=1.0
```

How to run supervised SlotMLPAdditive with consistency loss training:
```
python main.py --batch_size=512 --weight_decay=0.001 
--lr=0.001 --lr_scheduler_step=150 --warmup=True 
--unsupervised_mode=True --use_consistency_loss=True 
--detached_latents=True --extended_consistency_loss=True 
--consistency_scheduler=False 
--n_slot_latents=5 
--reconstruction_term_weight=0.01 --consistency_term_weight=0.05
```

How to run unsupervised SlotMLPAdditive with consistency loss training:
```
python main.py --batch_size=512 --weight_decay=0.001 --warmup=True 
--use_sampled_loss=True --unsupervised_mode=True --n_slot_latents=5 
--reconstruction_term_weight=1.0 --lr_scheduler_step=100 --lr=0.001
--consistency_term_weight=0.2 --detached_latents=False --epochs=3000
```