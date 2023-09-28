import argparse

from src import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the training and testing.")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use. Either 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "SlotMLPAdditive",
            "SlotAttention",
        ],
        default="SlotAttention",
        help="Model to use. One of the models defined in base_models.py.",
    )
    parser.add_argument(
        "--dataset_name",
        choices=["dsprites", "kubric"],
        default="dsprites",
        help="Dataset to use. All datasets are pre-generated and stored in data folder.",
    )
    parser.add_argument(
        "--use_consistency_loss",
        choices=[True, False],
        default=True,
        help="Whether to use consistency loss.",
    )
    parser.add_argument(
        "--extended_consistency_loss",
        choices=[True, False],
        default=False,
        help="Whether to use extended consistency loss.",
    )
    parser.add_argument(
        "--unsupervised_mode",
        choices=[True, False],
        default=True,
        help="Turns model to Autoencoder mode (no slots loss).",
    )
    parser.add_argument(
        "--epochs", type=int, default=400, help="Number of epochs to train for."
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use.")
    parser.add_argument(
        "--lr", type=float, default=0.0004, help="Learning rate to use."
    )
    parser.add_argument(
        "--lr_scheduler_step",
        type=int,
        default=50,
        help="How often to decrease learning rate.",
    )
    parser.add_argument(
        "--reconstruction_term_weight",
        type=float,
        default=1.0,
        help="Weight for reconstruction term in total loss.",
    )
    parser.add_argument(
        "--consistency_term_weight",
        type=float,
        default=1.0,
        help="Weight for consistency term in consistency loss.",
    )
    parser.add_argument(
        "--consistency_encoder_term_weight",
        type=float,
        default=1.0,
        help="Weight for consistency encoder loss in consistency loss.",
    )
    parser.add_argument(
        "--consistency_decoder_term_weight",
        type=float,
        default=1.0,
        help="Weight for consistency decoder loss in consistency loss.",
    )
    parser.add_argument(
        "--consistency_ignite_epoch",
        type=int,
        default=150,
        help="Epoch to start consistency loss.",
    )
    parser.add_argument(
        "--n_samples_train",
        type=int,
        default=100000,
        help="Number of samples in training dataset.",
    )
    parser.add_argument(
        "--n_samples_truncate",
        type=int,
        default=None,
        help="Number of samples to truncate training dataset to.",
    )
    parser.add_argument(
        "--n_samples_test",
        type=int,
        default=5000,
        help="Number of samples in testing dataset (ID and OOD).",
    )
    parser.add_argument(
        "--n_slots", type=int, default=2, help="Number of slots, i.e. objects in scene."
    )
    parser.add_argument(
        "--n_slot_latents",
        type=int,
        default=16,
        help="Number of latents per slot. GT is 5.",
    )
    parser.add_argument(
        "--no_overlap",
        choices=[True, False],
        default=True,
        help="Whether to allow overlapping figures.",
    )

    parser.add_argument(
        "--sample_mode_train",
        type=str,
        default="diagonal",
        help="Sampling mode for training dataset.",
    )

    parser.add_argument(
        "--sample_mode_test_id",
        type=str,
        default="diagonal",
        help="Sampling mode for ID testing dataset.",
    )

    parser.add_argument(
        "--sample_mode_test_ood",
        type=str,
        default="no_overlap_off_diagonal",
        help="Sampling mode for OOD testing dataset.",
    )

    parser.add_argument(
        "--delta",
        type=float,
        default=0.125,
        help="Delta for 'diagonal' and 'off_diagonal' dataset.",
    )

    parser.add_argument("--seed", type=int, default=2023, help="Random seed to use.")

    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load.",
    )

    args = parser.parse_args()

    print(args)

    args.save_name = f"temp"  # < -- change this to specify the name of the model
    train.run(**vars(args))
