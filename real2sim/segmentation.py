"""Instance segmentation with SAM 3 (text-prompted, open vocabulary)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class Detection:
    mask: np.ndarray  # (H, W) bool
    box: np.ndarray  # [x0, y0, x1, y1] pixels
    score: float
    label: str


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    return float(inter) / max(float(np.logical_or(a, b).sum()), 1.0)


def mask_containment(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the smaller mask covered by the other."""
    inter = np.logical_and(a, b).sum()
    return float(inter) / max(float(min(a.sum(), b.sum())), 1.0)


class Sam3Segmenter:
    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.4, checkpoint_path: str | None = None):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
        )
        self.processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)

    @torch.inference_mode()
    def __call__(
        self,
        rgb: np.ndarray,
        prompts: list[str],
        min_area_frac: float = 0.0005,
        max_area_frac: float = 0.25,
        dedupe_iou: float = 0.6,
    ) -> list[Detection]:
        h, w = rgb.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(rgb))
            raw: list[Detection] = []
            for prompt in prompts:
                self.processor.reset_all_prompts(state)
                out = self.processor.set_text_prompt(prompt=prompt, state=state)
                masks = out["masks"].squeeze(1).cpu().numpy()
                boxes = out["boxes"].float().cpu().numpy()
                scores = out["scores"].float().cpu().numpy()
                raw += [Detection(m, b, float(s), prompt) for m, b, s in zip(masks, boxes, scores)]

        # keep highest-scoring instance among overlapping detections (across prompts too)
        raw.sort(key=lambda d: -d.score)
        kept: list[Detection] = []
        for det in raw:
            area = det.mask.sum() / (h * w)
            if not (min_area_frac <= area <= max_area_frac):
                continue
            if any(mask_iou(det.mask, k.mask) > dedupe_iou or mask_containment(det.mask, k.mask) > 0.9 for k in kept):
                continue
            kept.append(det)
        return kept


def overlay(rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Visualization: colored masks + index labels."""
    import cv2

    vis = rgb.copy().astype(np.float32)
    rng = np.random.default_rng(0)
    for i, det in enumerate(detections):
        color = rng.integers(60, 255, 3).astype(np.float32)
        vis[det.mask] = 0.45 * vis[det.mask] + 0.55 * color
        x0, y0 = det.box[:2].astype(int)
        cv2.putText(vis, f"{i}:{det.label} {det.score:.2f}", (x0, max(y0 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1, cv2.LINE_AA)
    return vis.clip(0, 255).astype(np.uint8)
