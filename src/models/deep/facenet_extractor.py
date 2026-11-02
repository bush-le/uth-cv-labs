"""Module trích xuất vector đặc trưng khuôn mặt (512-D Embedding) bằng mạng FaceNet."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger("lab05.facenet")


class FaceNetExtractor:
    """Trình trích xuất đặc trưng khuôn mặt sử dụng kiến trúc FaceNet (InceptionResnetV1)."""

    SUPPORTED_PRETRAINED = ("vggface2", "casia-webface")

    def __init__(
        self,
        pretrained_dataset: str = "vggface2",
        checkpoint_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = "experiments/checkpoints/lab05",
        l2_normalize: bool = True,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.l2_normalize = l2_normalize
        self.pretrained_dataset = pretrained_dataset

        if checkpoint_dir is not None:
            ckpt_dir_path = Path(checkpoint_dir)
            ckpt_dir_path.mkdir(parents=True, exist_ok=True)
            os.environ["TORCH_HOME"] = str(ckpt_dir_path.resolve())

        if checkpoint_path is not None:
            ckpt_file = Path(checkpoint_path)
            if not ckpt_file.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file trọng số FaceNet tại: {ckpt_file}"
                )
            self.model = InceptionResnetV1(pretrained=None, classify=False)
            state_dict = torch.load(ckpt_file, map_location=self.device, weights_only=True)
            # Loại bỏ trọng số tầng phân loại logits nếu có trong state_dict gốc
            filtered_state = {
                k: v for k, v in state_dict.items() if not k.startswith("logits.")
            }
            self.model.load_state_dict(filtered_state, strict=False)
            logger.info("Đã nạp trọng số FaceNet cục bộ từ: %s", ckpt_file)
        else:
            if pretrained_dataset not in self.SUPPORTED_PRETRAINED:
                raise ValueError(
                    f"Pretrained dataset '{pretrained_dataset}' không hợp lệ. "
                    f"Chỉ hỗ trợ: {self.SUPPORTED_PRETRAINED}"
                )
            self.model = InceptionResnetV1(
                pretrained=pretrained_dataset,
                classify=False,
            )
            logger.info(
                "Đã khởi tạo FaceNet (InceptionResnetV1) pretrained='%s' trên %s.",
                pretrained_dataset,
                self.device,
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, face_tensor: torch.Tensor) -> np.ndarray:
        """Trích xuất face embedding từ tensor khuôn mặt.

        Args:
            face_tensor: Tensor đầu vào có shape (B, 3, 160, 160) hoặc (3, 160, 160).

        Returns:
            Mảng NumPy 1 chiều (512,) nếu B=1, hoặc (B, 512) nếu B>1, kiểu float32.
        """
        if face_tensor.ndim == 3:
            face_tensor = face_tensor.unsqueeze(0)
        if face_tensor.ndim != 4 or face_tensor.shape[1] != 3:
            raise ValueError(
                f"Đầu vào của FaceNet phải có dạng (B, 3, H, W), nhận được: {tuple(face_tensor.shape)}"
            )

        inputs = face_tensor.to(self.device, dtype=torch.float32)
        embeddings = self.model(inputs)

        if self.l2_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        output_np = embeddings.cpu().numpy().astype(np.float32)
        if output_np.shape[0] == 1:
            return output_np[0]
        return output_np
