# Copyright 2023 Roland S. Zimmermann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import os

import evaluation
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inferred-latents", type=str, required=True)
    parser.add_argument("--ground-truth-latents", type=str, required=True)
    parser.add_argument("--n-slots", type=int, required=False, default=-1)
    parser.add_argument("--categorical-dimensions", type=int, required=False, nargs="+")
    parser.add_argument("--evaluation-frequency", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.inferred_latents.endswith(".npz"):
        output_file = args.inferred_latents[:-4] + "_scores.npy"
    else:
        output_file = args.inferred_latents + ".scores.npy"

    if os.path.exists(output_file):
        print(f"Output file (`{output_file}`) exists already. Exiting.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(device))

    inferred_latents_data = np.load(args.inferred_latents, allow_pickle=True).item()
    ground_truth_latents_data = np.load(args.ground_truth_latents, allow_pickle=True)

    max_epochs = len(inferred_latents_data["inf_z"])
    print(f"Detected maximum number of epochs: {max_epochs}.")

    scores = {}

    for epoch in range(0, max_epochs, args.evaluation_frequency):
        print("Epoch: {}".format(epoch))

        inferred_latents = inferred_latents_data["inf_z"][epoch]

        if not (
            isinstance(ground_truth_latents_data, np.ndarray)
            and ground_truth_latents_data.ndim == 2
        ):
            ground_truth_latents = ground_truth_latents_data.item()["gt_z"][epoch]
        else:
            ground_truth_latents = ground_truth_latents_data

        print("Loaded data.")
        print("Inferred latents shape: {}".format(inferred_latents.shape))
        print("Ground-truth latents shape: {}".format(ground_truth_latents.shape))

        if args.n_slots != -1:
            inferred_latents = inferred_latents.reshape(
                len(inferred_latents), args.n_slots, -1
            )
            ground_truth_latents = ground_truth_latents.reshape(
                (len(ground_truth_latents), args.n_slots, -1)
            )

            print("Reshaped latents.")
            print("Inferred latents shape: {}".format(inferred_latents.shape))
            print("Ground-truth latents shape: {}".format(ground_truth_latents.shape))

        categorical_dimensions = args.categorical_dimensions
        print("Categorical dimensions: {}".format(categorical_dimensions))
        if len(categorical_dimensions) > 0:
            if max(categorical_dimensions) > inferred_latents.shape[-1]:
                raise ValueError("Invalid categorical dimension.")

        z = torch.Tensor(ground_truth_latents).to(device)
        z_pred = torch.Tensor(inferred_latents).to(device)

        current_scores = evaluation.evaluate_model(
            z,
            z_pred,
            categorical_dimensions,
            max_training_epochs=100,
            model_depth=5,
            train_val_test_split=(0.7, 0.1, 0.2),
            verbose=2,
            standard_scale=True,
            z_mask_values=0,
        )

        print("Scores:\n{}".format(current_scores))

        scores[epoch] = current_scores

    np.save(output_file, scores)


if __name__ == "__main__":
    main()
