# Provable Compositional Generalization for Object-Centric Learning [ICLR 2024]
Official code for the paper [Provable Compositional Generalization for Object-Centric Learning](https://arxiv.org/abs/2310.05327).

![Problem Setup](assets/fig3_v6.png)

## Environment Setup
This code was tested with Python 3.10. Start by cloning the repository:

```bash
git clone https://github.com/your-repository-url-here
cd your-repository-directory
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

🔗 For understanding how the data is look like and play with the data generation, please refer to the `notebooks/Sprite-World Dataset Example.ipynb` notebook.

🔗 For the actual data generation, please refer to the `notebooks/Data Generation.ipynb` notebook. The folder used for saving the dataset at this point would be used for training and evaluation.


## Training and Evaluation

### Training
To train the model, run the following command:

```bash
python train.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --epochs 400 --use_consistency_loss True
```

For complete details on the parameters, please refer to the `main.py` file.

You can find some example commands for training below:

<details open>
<summary><strong>Different training setups</strong></summary>

   - <details>
      <summary><strong>Training SlotAttention:</strong></summary>

      Training vanilla SlotAttention with 2 slots:
      ```bash
      python train.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss False
      ```

      Training vanilla SlotAttention with 2 slots and consistency loss:
      ```bash
      python train.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss True --consistency_ignite_epoch 150
      ```

      Training SlotAttention with 2 slots, fixed SoftMax and sampling:
      ```bash
      python train.py --dataset_path "/path/from/previous/step" --model_name "SlotAttention" --num_slots 2 --use_consistency_loss True --consistency_ignite_epoch 150 --softmax False --sampling False
      ```
   </details>

   - <details>
      <summary><strong>Training AE model:</strong></summary>

      Training vanilla autoencoder with 2 slots:
      ```bash
      python train.py --dataset_path "/path/from/previous/step" --model_name "SlotMLPAdditive" --epochs 300 --num_slots 2 -n_slot_latents 6 --use_consistency_loss False
      ```

      Training vanilla autoencoder with 2 slots and consistency loss:
      ```bash
      python train.py --dataset_path "/path/from/previous/step" --model_name "SlotMLPAdditive" --epochs 300 --num_slots 2 -n_slot_latents 6 --use_consistency_loss True --consistency_ignite_epoch 100
      ```


      </details>

</details>

### Evaluation