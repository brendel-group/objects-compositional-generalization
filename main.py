from src import train

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--model_name",
        choices=["SlotMLPAdditive", "SlotMLPEncoder", "SlotMLPMonolithic"],
        default="SlotMLPAdditive",
    )
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.15)
    parser.add_argument("--reduction", choices=["mean", "sum"], default="sum")
    parser.add_argument("--n_samples_train", type=int, default=50000)
    parser.add_argument("--n_samples_test", type=int, default=1000)
    parser.add_argument("--n_slots", type=int, default=2)
    parser.add_argument("--n_slot_latents", type=int, default=8)
    parser.add_argument("--no_overlap", choices=[True, False], default=True)
    parser.add_argument("--sample_mode_train", type=str, default="diagonal")
    parser.add_argument("--sample_mode_test_id", type=str, default="diagonal")
    parser.add_argument("--sample_mode_test_ood", type=str, default="off_diagonal")
    parser.add_argument("--delta", type=float, default=0.125)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2023)

    args = parser.parse_args()
    print(args)

    train.run(**vars(args))
