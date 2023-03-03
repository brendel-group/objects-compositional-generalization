How to run SlotAttention training:
```
python main.py --batch_size=64 --weight_decay=0.001 --warmup=True 
--unsupervised_mode=True --n_slot_latents=16 --reconstruction_term_weight=1.0 
--lr_scheduler_step=100 --lr=0.001
```
How to run SlotMLPAdditive w consistency loss training:
```
python main.py --batch_size=512 --weight_decay=0.001 --warmup=True 
--use_sampled_loss=True --unsupervised_mode=False --n_slot_latents=8 
--reconstruction_term_weight=0.01 --lr_scheduler_step=100 --lr=0.001
```