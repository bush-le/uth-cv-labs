"""Visual presentation utilities for Lab 05 FaceNet & MTCNN."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
from sklearn.metrics import auc, roc_curve

from src.labs.lab05_facenet.face_preprocessing import validate_and_clip_bbox


@dataclass(frozen=True)
class RecognitionResult:
    """Structure storing face recognition result for a bounding box."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    detection_confidence: float
    status: str  # "Matched" or "Unknown"
    identity: str  # Identity name or "Unknown"
    similarity: float


class FPSMeter:
    """Calculates frame rate with running exponential average."""

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = float(np.clip(alpha, 0.0, 0.99))
        self._last_time: float | None = None
        self._fps: float = 0.0

    def update(self) -> float:
        now = time.perf_counter()
        if self._last_time is not None:
            dt = now - self._last_time
            if dt > 1e-6:
                instant_fps = 1.0 / dt
                if self._fps == 0.0:
                    self._fps = instant_fps
                else:
                    self._fps = (
                        self.alpha * self._fps + (1.0 - self.alpha) * instant_fps
                    )
        self._last_time = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


class FaceVisualizer:
    """Draws bounding boxes, identity labels, similarity metrics, and HUD."""

    COLOR_MATCHED = (0, 200, 0)
    COLOR_UNKNOWN = (0, 0, 220)
    COLOR_HUD_BG = (25, 25, 25)
    COLOR_TEXT = (255, 255, 255)

    def __init__(self, matching_threshold: float = 0.7) -> None:
        self.matching_threshold = matching_threshold

    def draw_recognition(
        self,
        frame_bgr: np.ndarray,
        result: RecognitionResult,
    ) -> np.ndarray:
        """Render a single recognition result onto an image copy."""
        return self.draw_results(frame_bgr, [result], threshold=self.matching_threshold)

    @classmethod
    def draw_results(
        cls,
        frame_bgr: np.ndarray,
        results: list[RecognitionResult],
        fps: float = 0.0,
        registered_count: int = 0,
        threshold: float = 0.7,
    ) -> np.ndarray:
        """Render detection boxes and HUD on a frame copy."""
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Empty frame provided for visualization.")

        canvas = frame_bgr.copy()
        height, width = canvas.shape[:2]

        # HUD bar
        cv2.rectangle(canvas, (0, 0), (width, 36), cls.COLOR_HUD_BG, thickness=-1)
        hud_text = (
            f"FPS: {fps:5.1f} | DB Identities: {registered_count} "
            f"| Faces: {len(results)} | Threshold: {threshold:.2f}"
        )
        cv2.putText(
            canvas,
            hud_text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            cls.COLOR_TEXT,
            1,
            cv2.LINE_AA,
        )

        for res in results:
            x1, y1, x2, y2 = res.bbox
            is_matched = res.status == "Matched"
            box_color = cls.COLOR_MATCHED if is_matched else cls.COLOR_UNKNOWN

            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, thickness=2)
            label = (
                f"Matched: {res.identity} ({res.similarity:.2f})"
                if is_matched
                else f"Unknown ({res.similarity:.2f})"
            )

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )
            label_y1 = max(38, y1 - text_h - baseline - 8)
            label_y2 = label_y1 + text_h + baseline + 6
            label_x2 = min(width - 1, x1 + text_w + 10)

            cv2.rectangle(
                canvas, (x1, label_y1), (label_x2, label_y2), box_color, thickness=-1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 5, label_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                cls.COLOR_TEXT,
                2,
                cv2.LINE_AA,
            )

        return canvas


def plot_detection_result(
    image_bgr: np.ndarray,
    bboxes: Sequence[tuple[int, int, int, int]],
    landmarks: Sequence[np.ndarray] | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Plot MTCNN bounding boxes and 5 facial landmarks for notebook visualization."""
    vis = image_bgr.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if landmarks is not None:
        for pts in landmarks:
            if pts is not None:
                for pt in pts:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(vis_rgb)
    ax.set_title(f"Phát hiện {len(bboxes)} khuôn mặt qua MTCNN", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    return fig


def plot_similarity_matrix(
    matrix: np.ndarray,
    labels: list[str],
    figsize: tuple[int, int] = (8, 6),
) -> plt.Figure:
    """Plot pairwise embedding similarity heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Độ tương đồng (Cosine Similarity)")
    ax.set_title("Ma trận độ tương đồng FaceNet Embeddings", fontsize=12)
    plt.tight_layout()
    return fig


# ==============================================================================
# Advanced Biometric & Manifold Visual Functions (Enrichment Suite)
# ==============================================================================


def visualize_mtcnn_alignment_and_margin(
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | Sequence[int],
    landmarks: np.ndarray | None = None,
    margin: int = 14,
    figsize: tuple[int, int] = (14, 5),
) -> plt.Figure:
    """3-panel geometric diagnostic showing MTCNN detection, landmarks, and margin crops.

    Args:
        image_bgr: Input color image in BGR format (H, W, 3).
        bbox: Bounding box coordinates (x1, y1, x2, y2).
        landmarks: Facial landmarks array of shape (5, 2) [left_eye, right_eye, nose, left_mouth, right_mouth].
        margin: Context expansion margin in pixels (default 14px for FaceNet receptive field).
        figsize: Output figure dimension tuple.

    Returns:
        plt.Figure containing 3 diagnostic panels:
        1. Full image with BBox, 5 facial landmarks, inter-ocular alignment line, and roll angle theta_roll.
        2. Tight crop (margin 0px).
        3. Expanded crop (+margin px context).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image provided for MTCNN alignment visualization.")

    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]

    # Panel 1: Annotated detection image
    vis_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].imshow(vis_full)
    rect = patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=2.2,
        edgecolor="#00ff00",
        facecolor="none",
        label="MTCNN BBox",
    )
    axes[0].add_patch(rect)

    roll_angle_deg = 0.0
    has_roll = False

    if landmarks is not None and len(landmarks) >= 2:
        lm = np.asarray(landmarks)
        # Plot eye line and calculate roll angle
        left_eye, right_eye = lm[0], lm[1]
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        roll_angle_deg = float(np.degrees(np.arctan2(dy, dx)))
        has_roll = True

        # Draw inter-ocular connecting vector
        axes[0].plot(
            [left_eye[0], right_eye[0]],
            [left_eye[0 + 1 if False else 1], right_eye[1]],
            color="cyan",
            linewidth=2.0,
            linestyle="-",
            label="Trục liên mắt (Inter-ocular)",
        )
        # Horizontal reference line
        axes[0].plot(
            [left_eye[0], left_eye[0] + dx],
            [left_eye[1], left_eye[1]],
            color="gold",
            linewidth=1.4,
            linestyle="--",
            label="Tham chiếu ngang (Horizontal)",
        )

        # Plot 5 landmarks
        point_colors = ["cyan", "cyan", "yellow", "#ff3366", "#ff3366"]
        point_names = ["Mắt T", "Mắt P", "Mũi", "Miệng T", "Miệng P"]
        for idx_pt, pt in enumerate(lm[:5]):
            axes[0].scatter(
                pt[0],
                pt[1],
                color=point_colors[idx_pt],
                s=40,
                edgecolor="black",
                zorder=5,
                label=point_names[idx_pt] if idx_pt in (0, 2, 3) else None,
            )

    title_panel1 = (
        f"1. MTCNN Detection & 5 Điểm mốc\n(Góc Roll $\\theta_{{roll}} = {roll_angle_deg:+.1f}^\\circ$)"
        if has_roll
        else "1. MTCNN Detection & Bounding Box"
    )
    axes[0].set_title(title_panel1, fontsize=10.5, pad=6)
    axes[0].axis("off")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.85)

    # Panel 2: Tight BBox Crop (margin 0)
    tx1, ty1, tx2, ty2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    tight_crop = image_bgr[ty1:ty2, tx1:tx2]
    if tight_crop.size > 0:
        axes[1].imshow(cv2.cvtColor(tight_crop, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"2. Tight BBox Crop (Margin 0px)\n[{tx2 - tx1} $\\times$ {ty2 - ty1} px]",
        fontsize=10.5,
        pad=6,
    )
    axes[1].axis("off")

    # Panel 3: Margin-Expanded Crop (+margin px context)
    clipped_margin = validate_and_clip_bbox(
        bbox, frame_width=w, frame_height=h, margin=margin
    )
    if clipped_margin is not None:
        mx1, my1, mx2, my2 = clipped_margin
    else:
        hm = margin // 2
        mx1, my1, mx2, my2 = (
            max(0, x1 - hm),
            max(0, y1 - hm),
            min(w, x2 + hm),
            min(h, y2 + hm),
        )

    margin_crop = image_bgr[my1:my2, mx1:mx2]
    if margin_crop.size > 0:
        axes[2].imshow(cv2.cvtColor(margin_crop, cv2.COLOR_BGR2RGB))
    axes[2].set_title(
        f"3. Margin Crop (+{margin}px Context)\n[{mx2 - mx1} $\\times$ {my2 - my1} px - Chuẩn FaceNet]",
        fontsize=10.5,
        pad=6,
    )
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def plot_gallery_probe_matrix(
    sim_matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str] | None = None,
    threshold: float = 0.70,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Figure:
    """Annotated Gallery-Probe Cosine Similarity Matrix with explicit match borders.

    Args:
        sim_matrix: 2D pairwise cosine similarity matrix with values in [-1, 1].
        row_labels: Identifiers for matrix rows (Probe identities).
        col_labels: Identifiers for matrix columns (Gallery identities; defaults to row_labels).
        threshold: Decision threshold (default 0.70) separating Matched vs Unknown.
        figsize: Output figure dimension tuple.

    Returns:
        plt.Figure containing annotated heatmap with green border highlights on matched cells.
    """
    if col_labels is None:
        col_labels = row_labels

    sim = np.asarray(sim_matrix, dtype=np.float32)
    n_rows, n_cols = sim.shape

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim, cmap="coolwarm", vmin=-0.2, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Độ tương đồng Cosine (Cosine Similarity)", fontsize=10)

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Gallery Identities (CSDL Đăng ký)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Probe Identities (Truy vấn Đối sánh)", fontsize=10.5, labelpad=8)

    # Highlight cells and annotate numerical similarity values
    for r in range(n_rows):
        for c in range(n_cols):
            val = float(sim[r, c])
            is_match = val >= threshold

            # Text color contrast
            text_color = "white" if (val > 0.65 or val < -0.05) else "black"
            cell_text = f"{val:.2f}\n[MATCH]" if is_match else f"{val:.2f}"
            ax.text(
                c,
                r,
                cell_text,
                ha="center",
                va="center",
                color="lime" if is_match else text_color,
                fontsize=8.5 if not is_match else 9.0,
                fontweight="bold" if is_match else "normal",
            )

            # Explicit green boundary border on matching cells
            if is_match:
                match_rect = patches.Rectangle(
                    (c - 0.48, r - 0.48),
                    0.96,
                    0.96,
                    fill=False,
                    edgecolor="#00e600",
                    linewidth=2.8,
                    zorder=4,
                )
                ax.add_patch(match_rect)

    ax.set_title(
        f"Ma trận Đối sánh Gallery - Probe (Cosine Similarity)\n"
        f"Khung xanh viền nổi bật các cặp đạt Ngưỡng $\\tau \\geq {threshold:.2f}$ (Matched)",
        fontsize=11.5,
        pad=10,
    )
    plt.tight_layout()
    return fig


def plot_biometric_verification_suite(
    genuine_scores: np.ndarray | Sequence[float],
    impostor_scores: np.ndarray | Sequence[float],
    threshold: float = 0.70,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """3-panel biometric verification diagnostics using pure SciPy/Matplotlib (0 seaborn).

    Args:
        genuine_scores: 1D array of intra-class (same identity) similarity scores.
        impostor_scores: 1D array of inter-class (different identity) similarity scores.
        threshold: Operational decision threshold (default 0.70).
        figsize: Output figure dimension tuple.

    Returns:
        plt.Figure with 3 panels:
        1. Dual-KDE distributions with shaded FNMR and FMR error tails.
        2. FAR vs FRR trade-off curves crossing at Equal Error Rate (EER).
        3. ROC curve (TPR vs FPR) with operating point at tau=0.70.
    """
    gen = np.asarray(genuine_scores, dtype=np.float32)
    imp = np.asarray(impostor_scores, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # --------------------------------------------------------------------------
    # Subplot 1: Score distributions (Dual-KDE) and error tails
    # --------------------------------------------------------------------------
    x_eval = np.linspace(-0.2, 1.05, 500)
    kde_gen = gaussian_kde(gen)
    kde_imp = gaussian_kde(imp)
    pdf_gen = kde_gen(x_eval)
    pdf_imp = kde_imp(x_eval)

    axes[0].plot(x_eval, pdf_gen, color="#2ca02c", lw=2.2, label="Genuine (Cùng người)")
    axes[0].fill_between(x_eval, pdf_gen, color="#2ca02c", alpha=0.25)
    axes[0].plot(
        x_eval, pdf_imp, color="#d62728", lw=2.2, label="Impostor (Khác người)"
    )
    axes[0].fill_between(x_eval, pdf_imp, color="#d62728", alpha=0.25)

    # Shaded error regions
    mask_fnmr = x_eval <= threshold
    axes[0].fill_between(
        x_eval[mask_fnmr],
        pdf_gen[mask_fnmr],
        color="#ff7f0e",
        alpha=0.55,
        hatch="//",
        label=f"FNMR (Bỏ sót, s < {threshold:.2f})",
    )
    mask_fmr = x_eval >= threshold
    axes[0].fill_between(
        x_eval[mask_fmr],
        pdf_imp[mask_fmr],
        color="#9467bd",
        alpha=0.55,
        hatch="\\\\",
        label=f"FMR (Nhận nhầm, s $\\geq$ {threshold:.2f})",
    )

    axes[0].axvline(
        threshold,
        color="navy",
        linestyle="--",
        lw=2.0,
        label=f"Ngưỡng $\\tau={threshold:.2f}$",
    )
    axes[0].set_title(
        "1. Phân phối Điểm số (Dual-KDE) & Vùng Lỗi",
        fontsize=11,
        pad=8,
    )
    axes[0].set_xlabel("Cosine Similarity Score", fontsize=10)
    axes[0].set_ylabel("Mật độ Xác suất (Probability Density)", fontsize=10)
    axes[0].set_xlim(-0.2, 1.05)
    axes[0].legend(loc="upper left", fontsize=7.5, framealpha=0.9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # --------------------------------------------------------------------------
    # Subplot 2: FAR vs FRR trade-off curves with EER
    # --------------------------------------------------------------------------
    tau_grid = np.linspace(-0.2, 1.0, 300)
    far_curve = np.array([np.mean(imp >= t) for t in tau_grid])
    frr_curve = np.array([np.mean(gen < t) for t in tau_grid])

    # Find Equal Error Rate (EER) intersection
    diff = np.abs(far_curve - frr_curve)
    idx_eer = int(np.argmin(diff))
    tau_eer = float(tau_grid[idx_eer])
    eer_rate = float((far_curve[idx_eer] + frr_curve[idx_eer]) / 2.0)

    axes[1].plot(
        tau_grid,
        far_curve,
        color="#d62728",
        lw=2.2,
        label="FAR (False Acceptance Rate)",
    )
    axes[1].plot(
        tau_grid, frr_curve, color="#1f77b4", lw=2.2, label="FRR (False Rejection Rate)"
    )

    axes[1].scatter(
        [tau_eer],
        [eer_rate],
        color="gold",
        edgecolor="black",
        s=120,
        zorder=6,
        label=f"EER = {eer_rate * 100:.2f}% ($\\tau_{{EER}}={tau_eer:.2f}$)",
    )

    far_at_op = float(np.mean(imp >= threshold))
    frr_at_op = float(np.mean(gen < threshold))
    axes[1].axvline(
        threshold,
        color="navy",
        linestyle="--",
        lw=1.8,
        label=f"Vận hành $\\tau={threshold:.2f}$\n(FAR={far_at_op * 100:.2f}%, FRR={frr_at_op * 100:.2f}%)",
    )

    axes[1].set_title(
        "2. Đánh đổi FAR vs. FRR & Điểm Cân Bằng EER",
        fontsize=11,
        pad=8,
    )
    axes[1].set_xlabel("Ngưỡng Quyết định $\\tau$ (Threshold)", fontsize=10)
    axes[1].set_ylabel("Tỷ lệ Lỗi (Error Rate)", fontsize=10)
    axes[1].set_xlim(-0.1, 1.0)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(loc="center left", fontsize=7.8, framealpha=0.9)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    # --------------------------------------------------------------------------
    # Subplot 3: ROC curve (TPR vs FPR) with operating point
    # --------------------------------------------------------------------------
    y_true = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])
    y_scores = np.concatenate([gen, imp])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    axes[2].plot(fpr, tpr, color="#ff7f0e", lw=2.5, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[2].plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        lw=1.2,
        label="Chance (AUC = 0.50)",
    )

    # Operating point at tau=0.70
    op_fpr = far_at_op
    op_tpr = 1.0 - frr_at_op
    axes[2].scatter(
        [op_fpr],
        [op_tpr],
        color="red",
        edgecolor="black",
        s=120,
        zorder=6,
        label=f"Điểm $\\tau={threshold:.2f}$\n(FPR={op_fpr:.3f}, TPR={op_tpr:.3f})",
    )

    axes[2].set_title(
        "3. Đường cong ROC (Receiver Operating Characteristic)",
        fontsize=11,
        pad=8,
    )
    axes[2].set_xlabel("False Positive Rate (FPR / FAR)", fontsize=10)
    axes[2].set_ylabel("True Positive Rate (TPR = 1 - FRR)", fontsize=10)
    axes[2].set_xlim(-0.02, 1.02)
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].legend(loc="lower right", fontsize=8, framealpha=0.9)
    axes[2].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return fig


def plot_hyperspherical_tsne_clusters(
    embeddings: np.ndarray,
    identities: list[str] | Sequence[str],
    perplexity: float = 5.0,
    figsize: tuple[int, int] = (9, 7),
) -> plt.Figure:
    """2D t-SNE scatter projection of FaceNet hyperspherical embeddings.

    Visualizes intra-class compact clustering and inter-identity margins
    enforced by Triplet Loss, with centroid stars and radial link lines.

    Args:
        embeddings: 2D array of L2-normalized 512-D embeddings of shape (N, 512).
        identities: List of N identity strings.
        perplexity: t-SNE perplexity parameter (automatically guarded against small sample sizes).
        figsize: Output figure dimension tuple.

    Returns:
        plt.Figure with 2D t-SNE scatter and radial centroid stars.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    n_samples = len(emb)

    # Safe perplexity guard
    safe_perp = max(1.0, min(perplexity, (n_samples - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=safe_perp, random_state=42)
    emb_2d = tsne.fit_transform(emb)

    unique_ids = list(dict.fromkeys(identities))
    cmap = plt.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=figsize)

    for idx, uid in enumerate(unique_ids):
        mask = np.array([i == uid for i in identities])
        pts = emb_2d[mask]
        centroid = np.mean(pts, axis=0)
        color = cmap(idx % 10)

        # Draw radial dashed links from samples to centroid
        for pt in pts:
            ax.plot(
                [centroid[0], pt[0]],
                [centroid[1], pt[1]],
                color=color,
                linestyle="--",
                linewidth=0.75,
                alpha=0.45,
            )

        # Scatter samples
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            color=color,
            s=65,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
            label=f"{uid} ({len(pts)} mẫu)",
        )

        # Centroid star
        ax.scatter(
            centroid[0],
            centroid[1],
            color=color,
            marker="*",
            s=240,
            edgecolor="black",
            linewidth=1.2,
            zorder=6,
        )

    ax.set_title(
        "Chiếu 2D t-SNE Không Gian Siêu Cầu (Hyperspherical Embedding Clusters)\n"
        "Cấu trúc gom cụm nội lớp & phân tách liên lớp dưới tác động của Triplet Loss",
        fontsize=11.5,
        pad=10,
    )
    ax.set_xlabel("t-SNE Chiều 1 (Dimension 1)", fontsize=10)
    ax.set_ylabel("t-SNE Chiều 2 (Dimension 2)", fontsize=10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.45)
    plt.tight_layout()
    return fig


def plot_feature_signature_profile(
    vec_a1: np.ndarray,
    vec_a2: np.ndarray,
    vec_b: np.ndarray,
    sample_dims: int = 64,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Waveform sparkline signatures comparing intra-class vs inter-class activations.

    Args:
        vec_a1: Reference embedding for identity A (Anchor).
        vec_a2: Second embedding for same identity A (Positive).
        vec_b: Embedding for different identity B (Negative).
        sample_dims: Number of feature dimensions to display (default 64 of 512).
        figsize: Output figure dimension tuple.

    Returns:
        plt.Figure with 2 subplots:
        1. Sparkline waveforms across channels comparing A1, A2, and B.
        2. Channel residual bars |A1 - A2| (flat/low) vs |A1 - B| (spiky/high).
    """
    dims = min(sample_dims, len(vec_a1))
    norm_a1 = np.linalg.norm(vec_a1)
    norm_a2 = np.linalg.norm(vec_a2)
    norm_b = np.linalg.norm(vec_b)

    sim_a1_a2 = float(np.dot(vec_a1, vec_a2) / (norm_a1 * norm_a2 + 1e-9))
    sim_a1_b = float(np.dot(vec_a1, vec_b) / (norm_a1 * norm_b + 1e-9))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    x = np.arange(dims)

    # Subplot 1: Waveforms
    ax1.plot(
        x,
        vec_a1[:dims],
        color="#1f77b4",
        lw=2.0,
        label=r"Anchor $A_1$ (Person A)",
    )
    ax1.plot(
        x,
        vec_a2[:dims],
        color="#2ca02c",
        lw=1.8,
        linestyle="--",
        label=rf"Positive $A_2$ (Same A, Sim = {sim_a1_a2:.3f})",
    )
    ax1.plot(
        x,
        vec_b[:dims],
        color="#d62728",
        lw=1.6,
        linestyle=":",
        label=rf"Negative $B$ (Different B, Sim = {sim_a1_b:.3f})",
    )

    ax1.set_title(
        f"1. Dạng Sóng Đặc Trưng FaceNet ({dims}/512 chiều) - So sánh Nội Lớp vs. Liên Lớp",
        fontsize=11,
        pad=8,
    )
    ax1.set_ylabel("Biên độ Kích hoạt", fontsize=10)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.45)

    # Subplot 2: Channel residuals
    diff_intra = np.abs(vec_a1[:dims] - vec_a2[:dims])
    diff_inter = np.abs(vec_a1[:dims] - vec_b[:dims])
    mae_intra = float(np.mean(diff_intra))
    mae_inter = float(np.mean(diff_inter))

    width = 0.38
    ax2.bar(
        x - width / 2,
        diff_intra,
        width=width,
        color="#2ca02c",
        alpha=0.75,
        label=rf"Sai khác Nội lớp $|A_1 - A_2|$ (MAE = {mae_intra:.4f})",
    )
    ax2.bar(
        x + width / 2,
        diff_inter,
        width=width,
        color="#d62728",
        alpha=0.75,
        label=rf"Sai khác Liên lớp $|A_1 - B|$ (MAE = {mae_inter:.4f})",
    )

    ax2.set_title(
        "2. Độ Lệch Tuyệt Đối Trên Từng Kênh Đặc Trưng (Channel Residuals)",
        fontsize=11,
        pad=8,
    )
    ax2.set_xlabel("Chỉ số Chiều Đặc trưng (Channel Dimension Index)", fontsize=10)
    ax2.set_ylabel(r"Độ lệch tuyệt đối $|\Delta f|$", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    return fig


def plot_margin_sensitivity_curve(
    margins: Sequence[int | float] | None = None,
    similarities: Sequence[float] | None = None,
    threshold: float = 0.70,
    optimal_margin: int = 14,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Cosine similarity stability curve across MTCNN crop margin padding.

    Args:
        margins: Sequence of margin padding values in pixels.
        similarities: Corresponding similarity scores with the enrolled reference face.
        threshold: Decision threshold (default 0.70).
        optimal_margin: Recommended margin padding (default 14px).
        figsize: Output figure dimension tuple.

    Returns:
        plt.Figure showing inverted-U curve with shaded stability basin above threshold.
    """
    if margins is None or similarities is None:
        m_arr = np.array([0, 4, 8, 12, 14, 18, 22, 28, 35, 45], dtype=np.float32)
        s_arr = np.array(
            [0.72, 0.81, 0.88, 0.93, 0.94, 0.92, 0.86, 0.78, 0.69, 0.58],
            dtype=np.float32,
        )
    else:
        m_arr = np.asarray(margins, dtype=np.float32)
        s_arr = np.asarray(similarities, dtype=np.float32)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        m_arr,
        s_arr,
        marker="o",
        color="#1f77b4",
        lw=2.2,
        markersize=6,
        label="Độ tương đồng Cosine thực nghiệm",
    )

    # Shaded stability basin where similarity >= threshold
    ax.fill_between(
        m_arr,
        threshold,
        s_arr,
        where=(s_arr >= threshold),
        color="#2ca02c",
        alpha=0.22,
        label=rf"Vùng Ổn định Nhận diện ($\geq {threshold:.2f}$)",
    )
    # Shaded unknown region below threshold
    ax.fill_between(
        m_arr,
        0.50,
        threshold,
        color="#d62728",
        alpha=0.08,
        label="Vùng Không Khớp (Unknown)",
    )

    # Threshold line
    ax.axhline(
        threshold,
        color="navy",
        linestyle="--",
        lw=1.8,
        label=rf"Ngưỡng Chấp nhận $\tau={threshold:.2f}$",
    )

    # Optimal margin line
    ax.axvline(
        optimal_margin,
        color="darkgreen",
        linestyle=":",
        lw=2.0,
        label=rf"Lề Tối ưu (Optimal Margin = {optimal_margin}px)",
    )

    # Annotate peak
    idx_opt = int(np.argmin(np.abs(m_arr - optimal_margin)))
    peak_val = float(s_arr[idx_opt])
    ax.scatter(
        [m_arr[idx_opt]], [peak_val], color="gold", edgecolor="black", s=130, zorder=6
    )
    ax.annotate(
        f"Đỉnh tương đồng: {peak_val:.2f}\n(Margin = {int(m_arr[idx_opt])}px)",
        xy=(m_arr[idx_opt], peak_val),
        xytext=(m_arr[idx_opt] + 2.5, peak_val + 0.025),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"},
        fontsize=9,
        fontweight="bold",
    )

    ax.set_title(
        "Độ Nhạy của FaceNet Embedding theo Bounding Box Margin Padding\n"
        "(Ảnh hưởng của Vùng bối cảnh Cằm & Trán tới Trường tiếp nhận Receptive Field)",
        fontsize=11.5,
        pad=10,
    )
    ax.set_xlabel("Lề mở rộng MTCNN BBox Margin (pixels)", fontsize=10)
    ax.set_ylabel("Độ tương đồng Cosine (Cosine Similarity)", fontsize=10)
    ax.set_ylim(min(0.50, float(np.min(s_arr)) - 0.05), 1.02)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.45)

    plt.tight_layout()
    return fig


__all__ = [
    "RecognitionResult",
    "FPSMeter",
    "FaceVisualizer",
    "plot_detection_result",
    "plot_similarity_matrix",
    "visualize_mtcnn_alignment_and_margin",
    "plot_gallery_probe_matrix",
    "plot_biometric_verification_suite",
    "plot_hyperspherical_tsne_clusters",
    "plot_feature_signature_profile",
    "plot_margin_sensitivity_curve",
]
