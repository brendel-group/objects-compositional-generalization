
# Provable Compositional Generalization for Object-Centric Learning [ICLR 2024]
Official code for the paper [Provable Compositional Generalization for Object-Centric Learning](https://arxiv.org/abs/2310.05327).

![Problem Setup](assets/fig3_v6.png)

## Environment Setup
This code was tested for Python 3.10. 

You can start by cloning the repository:

```bash
git clone https://github.com/kotekjedi/object_centric_ood.git
cd object_centric_ood
```

Then, set up your environment by choosing one of the following methods:

<details open>
<summary><strong>Option 1: Installing Dependencies Directly</strong></summary>

```bash
pip install -r requirements.txt
```

</details>

Or, alternatively, you can use Docker:

<details open>
<summary><strong>Option 2: Building a Docker Image</strong></summary>

Build and run a Docker container using the provided Dockerfile:
```bash
docker build -t object_centric_ood .
docker-compose up
```

</details>

## Data Generation

🔗 For understanding how the data looks and to play with the data generation, please refer to the `notebooks/0. Sprite-World Dataset Example.ipynb` notebook.

🔗 For the actual data generation, please refer to the `notebooks/1. Data Generation.ipynb` notebook. The folder used for saving the dataset at this point will be used for training and evaluation.

## Training and Evaluation

### Training
To train the model, run the following command:

```bash
python src/main.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --epochs 400 --use_consistency_loss True
```

For complete details on the parameters, please refer to the `src/main.py` file.

You can find some example commands for training below:

<details open>
<summary><strong>Different Training Setups</strong></summary>

- <details>
  <summary><strong>Training SlotAttention</strong></summary>

  Training vanilla SlotAttention with 2 slots:
  ```bash
  python src/main.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss False
  ```

  Training vanilla SlotAttention with 2 slots and consistency loss:
  ```bash
  python src/main.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss True --consistency_ignite_epoch 150
  ```

  Training SlotAttention with 2 slots, fixed SoftMax and sampling:
  ```bash
  python src/main.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss True --consistency_ignite_epoch 150 --softmax False --sampling False
  ```
</details>

- <details>
  <summary><strong>Training AE Model</strong></summary>

  Training vanilla autoencoder with 2 slots:
  ```bash
  python src/main.py --dataset_path "/path/from/previous/step" --model_name "SlotMLPAdditive" --epochs 300 --num_slots 2 -n_slot_latents 6 --use_consistency_loss False
  ```

  Training vanilla autoencoder with 2 slots and consistency loss:
  ```bash
  python src/main.py --dataset_path "/path/from/previous/step" --model_name "SlotMLPAdditive" --epochs 300 --num_slots 2 -n_slot_latents 6 --use_consistency_loss True --consistency_ignite_epoch 100
  ```

</details>

</details>

### Evaluation

Evaluation can be done using the `evaluate.py` script and closely follows the procedure and metrics used in the training script. The main difference is in calculating the compositional contrast (note: it might cause OOM issues, thus is calculated only for the AE model).

Here is an example command for evaluation:
```bash
python src/evaluation.py --dataset_path "/path/from/previous/step" --model_path "checkpoints/SlotMLPAdditive.pt" --model_name "SlotMLPAdditive" --n_slot_latents 6
```

<details>
<summary><strong>Misc</strong></summary>

🔗 `notebooks/2. Decoder Optimality.ipynb` shows how the experiment described in Appendix B.2 is conducted. Note that the paths to models' checkpoints and exact data splits are omitted, thus it serves more of an illustrative purpose.
</details>

## Citation
If you make use of this code in your own work, please cite our paper:
```bibtex
@inproceedings{
wiedemer2024provable,
title={Provable Compositional Generalization for Object-Centric Learning},
author={Thadd{\"a}us Wiedemer and Jack Brady and Alexander Panfilov and Attila Juhos and Matthias Bethge and Wieland Brendel},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=7VPTUWkiDQ}
}
```
