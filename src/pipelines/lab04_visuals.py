"""Small plotting helpers for the Lab04 teaching notebook."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from src.models.classical.wavelet_hash import hamming_distance


def show_grid(
    images: Sequence[np.ndarray], titles: Sequence[str], ncols: int = 3
) -> None:
    """Display grayscale/coefficient or RGB images in a compact notebook grid."""
    if not images or len(images) != len(titles) or ncols < 1:
        raise ValueError("Images, titles and a positive column count are required.")
    rows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        rows, ncols, figsize=(4 * ncols, 3.5 * rows), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image, title in zip(axes.ravel(), images, titles, strict=False):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_pair_examples(
    pairs: Sequence[Mapping[str, str]], images: Mapping[str, np.ndarray]
) -> list[Mapping[str, str]]:
    """Display one pair per theme/label and return those example records."""
    selected = {}
    for pair in pairs:
        selected.setdefault((pair["theme"], pair["label"]), pair)
    examples = list(selected.values())
    show_grid(
        [images[p[key]] for p in examples for key in ("left_id", "right_id")],
        [
            f"{p['theme']} | nhãn {p['label']} | {side}"
            for p in examples
            for side in ("trái", "phải")
        ],
        ncols=4,
    )
    return examples


def plot_hash_steps(image: np.ndarray, steps: Mapping[str, tuple]) -> None:
    """Show isolated RGB channels and the LL/resampled LL/bits from each channel."""
    views = []
    for channel in range(3):
        view = np.zeros_like(image)
        view[:, :, channel] = image[:, :, channel]
        views.append(view)
    show_grid([image] + views, ["Ảnh RGB"] + [f"Kênh {n}" for n in "RGB"], ncols=4)
    show_grid(
        [steps[n][i] for n in "RGB" for i in (1, 2, 3)],
        [f"{n}: {title}" for n in "RGB" for title in ("LL", "LL 8×8", "Bit 8×8")],
        ncols=3,
    )


def plot_channel_hamming(
    pairs: Sequence[Mapping[str, str]], hashes: Mapping[str, np.ndarray], bits: int
) -> None:
    """Show R/G/B contributions to Hamming distance for example pairs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(pairs))
    channel_bits = bits**2
    for channel, name in enumerate("RGB"):
        values = [
            hamming_distance(
                hashes[p["left_id"]][
                    channel * channel_bits : (channel + 1) * channel_bits
                ],
                hashes[p["right_id"]][
                    channel * channel_bits : (channel + 1) * channel_bits
                ],
            )
            for p in pairs
        ]
        ax.bar(x + (channel - 1) * 0.22, values, width=0.22, label=name)
    ax.set_xticks(x, [f"{p['theme']}\nnhãn {p['label']}" for p in pairs])
    ax.set(
        ylabel=f"Bit khác nhau / {channel_bits} bit kênh",
        title="Đóng góp Hamming của R, G, B trong 4 cặp minh họa",
    )
    ax.legend(title="Kênh")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_hamming_matrix(
    matrix: np.ndarray, sources: Sequence[Mapping[str, str]], bits: int
) -> None:
    """Display the pairwise RGB Hamming matrix with source labels."""
    fig, ax = plt.subplots(figsize=(10, 8))
    plot = ax.imshow(matrix, vmin=0, vmax=3 * bits**2, cmap="viridis")
    ax.set_xticks(
        range(len(sources)), [str(i + 1) for i in range(len(sources))], fontsize=7
    )
    ax.set_yticks(
        range(len(sources)),
        [f"{i + 1}: {s['category']} {s['source_id']}" for i, s in enumerate(sources)],
        fontsize=7,
    )
    ax.set_title("Hamming giữa các ảnh nguồn — số trục là ID trong hình")
    fig.colorbar(plot, ax=ax, label="Số bit khác nhau")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_robustness(
    robustness: Mapping[str, Sequence[int]], source_count: int, bits: int
) -> None:
    """Compare source-to-variant distances across transformations."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot(list(robustness.values()), tick_labels=list(robustness))
    ax.set_ylabel(f"Hamming với ảnh nguồn ({bits} bit)")
    ax.set_title(f"Độ bền trên {source_count} ảnh calibration")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_threshold_sweep(
    labels: np.ndarray, distances: np.ndarray, threshold: float
) -> None:
    """Visualize calibration Recall, Specificity and Youden J versus threshold."""
    from src.pipelines.lab04_evaluation import score_pairs

    candidates = np.unique(distances)
    scores = [score_pairs(labels, distances, float(t)) for t in candidates]
    fig, ax = plt.subplots(figsize=(9, 4))
    for key, name in [("recall", "Recall"), ("specificity", "Specificity")]:
        ax.plot(candidates, [v[key] for v in scores], marker=".", label=name)
    ax.plot(
        candidates,
        [v["recall"] + v["specificity"] - 1 for v in scores],
        marker=".",
        label="Youden J",
    )
    ax.axvline(threshold, color="black", linestyle="--", label=f"τ={threshold:g} chọn")
    ax.set(
        xlabel="Ngưỡng Hamming RGB / 192 bit",
        ylabel="Tỷ lệ",
        ylim=(-0.05, 1.05),
        title="Chọn ngưỡng trên calibration",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_evaluation(
    labels: Mapping[str, np.ndarray],
    distances: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    threshold: float,
    hash_bits: int,
) -> None:
    """Show ROC, evaluation distances and confusion matrix at a frozen threshold."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split in labels:
        fpr, tpr, _ = roc_curve(labels[split], -distances[split])
        axes[0].plot(fpr, tpr, label=f"{split}: AUC={metrics[split]['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set(xlabel="FPR", ylabel="TPR (Recall)", title="ROC")
    axes[0].legend(fontsize=8)
    for label, name in [(0, "Khác nguồn"), (1, "Cùng nguồn")]:
        axes[1].hist(
            distances["evaluation"][labels["evaluation"] == label],
            bins=np.arange(-0.5, hash_bits + 1.5, 4),
            alpha=0.65,
            label=name,
        )
    axes[1].axvline(threshold, color="red", linestyle="--", label=f"τ={threshold:g}")
    axes[1].set(
        xlabel="Hamming", ylabel="Số cặp", title="Evaluation: phân bố khoảng cách"
    )
    axes[1].legend(fontsize=8)
    cm = confusion_matrix(
        labels["evaluation"], distances["evaluation"] <= threshold, labels=[0, 1]
    )
    axes[2].imshow(cm, cmap="Blues")
    for (y, x), value in np.ndenumerate(cm):
        axes[2].text(x, y, str(value), ha="center", va="center")
    axes[2].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xlabel="Dự đoán",
        ylabel="Nhãn thật",
        title="Confusion matrix: 0=khác, 1=cùng",
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_error_pair(
    left: np.ndarray,
    right: np.ndarray,
    left_hash: np.ndarray,
    right_hash: np.ndarray,
    label: int,
    threshold: float,
    bits: int,
) -> None:
    """Show a mistaken pair, RGB pixel change and three bit-mismatch maps."""
    changed = (left_hash != right_hash).reshape(3, bits, bits)
    difference = np.clip(
        np.abs(left.astype(np.int16) - right.astype(np.int16)) * 3, 0, 255
    ).astype(np.uint8)
    distance = hamming_distance(left_hash, right_hash)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, image, title in zip(
        axes[0],
        [left, right, difference],
        ["Ảnh gốc", "Bản tăng sáng", "|ΔRGB| ×3 để dễ nhìn"],
        strict=True,
    ):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    for channel, name in enumerate("RGB"):
        axes[1, channel].imshow(
            changed[channel], cmap="Reds", vmin=0, vmax=1, interpolation="nearest"
        )
        axes[1, channel].set_title(
            f"Kênh {name}: {changed[channel].sum()}/{bits**2} bit đổi"
        )
        axes[1, channel].axis("off")
    predicted = int(distance <= threshold)
    verdict = "FN" if label == 1 else "FP"
    relation = ">" if verdict == "FN" else "≤"
    fig.suptitle(
        f"{verdict}: nhãn thật {label}, dự đoán {predicted}; d={distance} {relation} τ={threshold:g}"
    )
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_method_summary(method_results: Sequence[Mapping[str, object]]) -> None:
    """Compare held-out Recall and Specificity for each method."""
    fig, ax = plt.subplots(figsize=(11, 4))
    names = [str(r["name"]) for r in method_results]
    x = np.arange(len(names))
    ax.bar(
        x - 0.18,
        [r["evaluation"]["recall"] for r in method_results],
        width=0.36,
        label="Recall evaluation",
    )
    ax.bar(
        x + 0.18,
        [r["evaluation"]["specificity"] for r in method_results],
        width=0.36,
        label="Specificity evaluation",
    )
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set(
        ylim=(0, 1.05),
        ylabel="Tỷ lệ",
        title="Khảo sát Wavelet — ngưỡng chọn riêng trên calibration",
    )
    ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_retrieval(
    query_rgb: np.ndarray,
    candidates: Sequence[Mapping[str, str]],
    results: Sequence[tuple[int, int]],
    images: Mapping[str, np.ndarray],
    threshold: float,
) -> None:
    """Display the RGB query and ranked images with threshold decisions."""
    titles = ["Ảnh truy vấn"]
    pictures = [query_rgb]
    for rank, (index, distance) in enumerate(results, start=1):
        candidate = candidates[index]
        pictures.append(images[candidate["source_id"]])
        decision = "đủ gần" if distance <= threshold else "vượt ngưỡng"
        titles.append(f"Top {rank}: {candidate['category']}\nd={distance} — {decision}")
    show_grid(pictures, titles, ncols=len(pictures))
