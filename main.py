import argparse

from src import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--model_name",
        choices=[
            "SlotMLPAdditive",
            "SlotMLPEncoder",
            "SlotMLPMonolithic",
            "SlotMLPAdditiveDecoder",
            "SlotMLPMonolithicDecoder",
            "SlotAttention",
        ],
        default="SlotMLPAdditive",
    )
    parser.add_argument("--use_consistency_loss", choices=[True, False], default=True)
    parser.add_argument(
        "--extended_consistency_loss", choices=[True, False], default=True
    )
    parser.add_argument("--unsupervised_mode", choices=[True, False], default=True)
    parser.add_argument("--detached_latents", choices=[True, False], default=True)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup", choices=[True, False], default=True)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--lr_scheduler_step", type=int, default=150)
    parser.add_argument("--reduction", choices=["mean", "sum"], default="sum")
    parser.add_argument("--reconstruction_term_weight", type=float, default=1.0)
    parser.add_argument("--consistency_term_weight", type=float, default=1.0)
    parser.add_argument("--consistency_scheduler", choices=[True, False], default=True)
    parser.add_argument("--consistency_scheduler_step", type=int, default=4000)
    parser.add_argument("--n_samples_train", type=int, default=100000)
    parser.add_argument("--n_samples_test", type=int, default=5000)
    parser.add_argument("--n_slots", type=int, default=2)
    parser.add_argument("--n_slot_latents", type=int, default=5)
    parser.add_argument("--no_overlap", choices=[True, False], default=True)
    parser.add_argument("--sample_mode_train", type=str, default="diagonal")
    parser.add_argument("--sample_mode_test_id", type=str, default="diagonal")
    parser.add_argument("--sample_mode_test_ood", type=str, default="off_diagonal")
    parser.add_argument("--delta", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=2023)

    args = parser.parse_args()
    print(args)

    train.run(**vars(args))
