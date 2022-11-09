import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rc("axes", labelsize=15)


def plot_random_dataset_images(dataset, rows=2, cols=3):
    """Showing random images from the dataset."""
    fig, ax = plt.subplots(rows, cols, sharex="col", sharey="row")

    for row in range(rows):
        for col in range(cols):
            ax[row, col].imshow(dataset[np.random.choice(range(len(dataset)))][0])
    plt.tight_layout()

    plt.show()


def plot_latents_scatter(dataset):
    """Scatter plot for latents to check whete sampling works correctly."""
    df = pd.DataFrame(
        dataset[:][1].reshape(-1, 8),
        columns=list(dataset.cfg.get_latents_metadata().keys()),
    )

    sns.pairplot(df, hue="shape", diag_kind="hist", corner=True, palette="tab10")


def plot_slots_scatter(dataset, n_slots, delta=0):
    slot_i, slot_j = np.random.choice(range(n_slots), size=[2], replace=False)
    if slot_i < slot_j:
        slot_i, slot_j = slot_j, slot_i

    metadata = dataset.cfg.get_latents_metadata()
    rows = dataset.cfg.get_total_latent_dim // 3
    if dataset.cfg.get_total_latent_dim % 3 > 0:
        rows += 1
    fig, ax = plt.subplots(rows, 3, figsize=(10, 10))
    fig.suptitle(f"delta={delta}")
    fig.tight_layout()

    for i, latent in enumerate(metadata):
        ax_i = i // 3
        ax_j = i % 3
        x = dataset[:][1][:, slot_i, i]  # i-th latent of all slot_i-th slots
        y = dataset[:][1][:, slot_j, i]  # j-th latent of all slot_j-th slots

        if metadata[latent][0] == "categorical":
            print(
                f"For latent '{latent}' {(x == y).sum()} pairs out of {len(x)} are the same."
            )

            _min = min(min(x), min(y))
            _max = max(max(x), max(y))
        else:
            _min = dataset.cfg[latent].min
            _max = dataset.cfg[latent].max

        _offset = (_max - _min) * 0.05
        if latent == "x":
            _offset = (_max - _min) * 0.7

        ax[ax_i, ax_j].scatter(
            x, y, s=25, c=x, cmap="viridis", edgecolors="black", linewidth=0.2
        )


        ax[ax_i, ax_j].set_xlim(_min  - _offset, _max + _offset)
        ax[ax_i, ax_j].set_ylim(_min  - _offset, _max + _offset)
        ax[ax_i, ax_j].set_xlabel(r"$z_{" + f"{slot_i}" + "_{" + f"{latent}" + "}}$")
        ax[ax_i, ax_j].set_ylabel(r"$z_{" + f"{slot_j}" + "_{" + f"{latent}" + "}}$")
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()


def plot_slots_heatmap(dataset, n_slots, delta=0):
    slot_i, slot_j = np.random.choice(range(n_slots), size=[2], replace=False)
    metadata = dataset.cfg.get_latents_metadata()
    for i, latent in enumerate(metadata):
        x = dataset[:][1][:, slot_i, i]  # i-th latent of all slot_i-th slots
        y = dataset[:][1][:, slot_j, i]  # j-th latent of all slot_j-th slots
        df = pd.DataFrame(
            np.hstack([x.numpy().reshape(-1, 1), y.numpy().reshape(-1, 1)]),
            columns=["x", "y"],
        )
        df.x = df.x.apply(lambda x: round(x, 2))
        df.y = df.y.apply(lambda x: round(x, 2))
        df["counts"] = 1
        df = df.groupby(["x", "y"]).sum().reset_index()
        if df.counts.max() < 5:
            continue
        df = df.pivot("x", "y", "counts")

        if metadata[latent][0] == "categorical":
            continue
        sns.heatmap(df).invert_yaxis()
        plt.xlabel(r"$z_{" + f"{slot_i}" + "_{" + f"{latent}" + "}}$")
        plt.ylabel(r"$z_{" + f"{slot_j}" + "_{" + f"{latent}" + "}}$")
        plt.show()
