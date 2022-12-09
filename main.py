from src import train

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model", choices=["SlotMLPAdditive", "SlotMLPEncoder", "SlotMLPMonolithic"], default="SlotMLPAdditive")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n_samples_train", type=int, default=10000)
    parser.add_argument("--n_samples_test", type=int, default=1000)
    parser.add_argument("--n_slots", type=int, default=2)
    parser.add_argument("--n_slot_latents", type=int, default=8)
    parser.add_argument("--no_overlap", action="store_true")
    parser.add_argument("--sample_mode_train", type=str, default="diagonal")
    parser.add_argument("--sample_mode_test_id", type=str, default="diagonal")
    parser.add_argument("--sample_mode_test_ood", type=str, default="off_diagonal")
    parser.add_argument("--delta", type=float, default=0.125)
    parser.add_argument("--in_channels", type=int, default=3)

    train.run(**vars(parser.parse_args()))