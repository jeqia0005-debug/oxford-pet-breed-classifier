"""Grad-CAM explanations for a trained checkpoint (Member 3).

Wraps ``pytorch-grad-cam`` (jacobgil/pytorch-grad-cam) rather than
reimplementing the gradient hooks from scratch. Works for both
``CustomCNN`` and ``MobileNetV2Classifier`` via
:func:`pet_classifier.evaluation.checkpoints.gradcam_target_layer`.

Install with: ``uv add grad-cam`` (package name is ``grad-cam``, imported
as ``pytorch_grad_cam``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pet_classifier.data.transforms import build_eval_transform, denormalize
from pet_classifier.evaluation.checkpoints import gradcam_target_layer


class GradCAMExplainer:
    """Generates Grad-CAM heatmaps overlaid on the original (denormalized) image."""

    def __init__(self, model: torch.nn.Module, device: torch.device | str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.transform = build_eval_transform()
        target_layer = gradcam_target_layer(model)
        # use_cuda kwarg is deprecated in newer versions; device is inferred
        # from the model's parameters instead.
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])

    def explain_tensor(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Run Grad-CAM on a single already-normalized ``(C, H, W)`` tensor.

        If ``target_class`` is None, explains the model's own top prediction.
        Returns ``(overlay_rgb_uint8_as_float, predicted_class, confidence)``.
        """
        batch = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)[0]
        predicted = int(probs.argmax())
        confidence = float(probs[predicted])

        # The frozen MobileNetV2 backbone has requires_grad=False on every
        # feature-block parameter, so activations produced there have
        # requires_grad=False too and pytorch-grad-cam's backward hook never
        # fires. Marking the *input* as requiring grad makes the whole graph
        # (including frozen layers) track gradients for this one forward
        # pass, without touching any parameter's requires_grad flag.
        batch.requires_grad_(True)

        targets = [ClassifierOutputTarget(target_class or predicted)]
        grayscale_cam = self.cam(input_tensor=batch, targets=targets)[0]  # (H, W)

        rgb_image = denormalize(image_tensor).permute(1, 2, 0).cpu().numpy()
        overlay = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

        return overlay, predicted, confidence

    def explain_image_file(
        self,
        image_path: str | Path,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Convenience wrapper: load a PIL image from disk, preprocess, explain.

        This is the entry point the Streamlit app should call for an
        uploaded image.
        """
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        return self.explain_tensor(tensor, target_class=target_class)

    def explain_pil_image(
        self,
        image: Image.Image,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, float]:
        """Same as :meth:`explain_image_file` but for an in-memory PIL image
        (e.g. from ``st.file_uploader`` in the Streamlit app)."""
        tensor = self.transform(image.convert("RGB"))
        return self.explain_tensor(tensor, target_class=target_class)
