from __future__ import annotations
import matplotlib.pyplot as plt
from typing import Sequence, Optional

__all__ = [
    "plot_training_curves",
    'plot_all_metrics_vs_negatives'
]

def plot_training_curves(
    train_loss: Sequence[float],
    rmse:       Sequence[float],
    mae:        Sequence[float],
    *,
    title_size: int = 11,
    label_size: int = 10,
    tick_size:  int = 9,
    colors: tuple[str, str] = ("royalblue", "firebrick"),
    save_path: str | None = None,
):
    epochs = range(1, len(train_loss) + 1)

    fig, (ax_loss, ax_rmse) = plt.subplots(1, 2, figsize=(11, 4), dpi=110, sharex=True, gridspec_kw=dict(wspace=0.28))

    ax_loss.plot(epochs, train_loss, marker="o", lw=2, color=colors[0], label="Train Loss")

    title = "Evolution of Training Loss"

    ax_loss.set_title(title, fontsize=title_size)
    ax_loss.set_xlabel("Epochs", fontsize=label_size)
    ax_loss.set_ylabel("Loss",   fontsize=label_size)
    ax_loss.tick_params(axis="both", labelsize=tick_size)
    ax_loss.grid(alpha=.3)
    ax_loss.legend(fontsize=tick_size)

    ax_rmse.plot(epochs, rmse, marker="o", lw=2, color=colors[0], label="RMSE")
    ax_rmse.set_title("Evolution of RMSE and MAE per Epoch", fontsize=title_size)
    ax_rmse.set_xlabel("Epochs", fontsize=label_size)
    ax_rmse.set_ylabel("RMSE",   fontsize=label_size, color=colors[0])
    ax_rmse.tick_params(axis="y", labelcolor=colors[0], labelsize=tick_size)
    ax_rmse.tick_params(axis="x", labelsize=tick_size)
    ax_rmse.grid(alpha=.3)

    ax_mae = ax_rmse.twinx()
    ax_mae.plot(epochs, mae, marker="o", lw=2,
                color=colors[1], label="MAE")
    ax_mae.set_ylabel("MAE", fontsize=label_size, color=colors[1])
    ax_mae.tick_params(axis="y", labelcolor=colors[1], labelsize=tick_size)

    lines, labels = ax_rmse.get_legend_handles_labels()
    lines2, labels2 = ax_mae.get_legend_handles_labels()
    ax_rmse.legend(lines + lines2, labels + labels2, fontsize=tick_size, loc="upper right")

    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure ➟ {save_path}")
        
    else:
        plt.show()
        
def plot_all_metrics_vs_negatives(
    negatives: list[int],
    hr_list:      list[float],
    ndcg_list:    list[float],
    acc_list:     list[float],
    tol_acc_list: list[float],
    *,
    K: int = 10,
    title_size: int = 11,
    label_size: int = 10,
    tick_size: int = 9,
    colors: tuple[str, str, str, str] = (
        "royalblue",    # HR
        "firebrick",    # NDCG
        "seagreen",     # Accuracy
        "darkorange",   # ±1‑star acc
    ),
    save_path: str | None = None,
):

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4), dpi=110, sharex=True,
        gridspec_kw=dict(wspace=0.3)
    )

    ax1.plot(negatives, hr_list,    marker="o", lw=2, color=colors[0], label=f"HR@{K}")
    ax1.plot(negatives, ndcg_list,  marker="s", lw=2, color=colors[1], label=f"NDCG@{K}")
    ax1.set_title(f"Ranking Metrics (K={K})", fontsize=title_size)
    ax1.set_xlabel("Number of negatives", fontsize=label_size)
    ax1.set_ylabel("Metric value", fontsize=label_size)
    ax1.tick_params(axis="both", labelsize=tick_size)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=tick_size)

    ax2.plot(negatives, acc_list,    marker="o", lw=2, color=colors[2], label="Accuracy")
    ax2.plot(negatives, tol_acc_list,marker="s", lw=2, color=colors[3], label="±1‑star Acc")
    ax2.set_title("Explicit‑Rating Accuracy", fontsize=title_size)
    ax2.set_xlabel("Number of negatives", fontsize=label_size)
    ax2.set_ylabel("Accuracy", fontsize=label_size)
    ax2.tick_params(axis="both", labelsize=tick_size)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=tick_size)

    ax1.set_xticks(negatives)
    ax2.set_xticks(negatives)

    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved figure ➟  {save_path}")
    else:
        plt.show()