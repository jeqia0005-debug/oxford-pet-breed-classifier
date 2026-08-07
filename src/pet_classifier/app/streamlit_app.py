"""Streamlit web demo (Member 3).

Upload a cat/dog photo and see:
* the predicted breed with a confidence score
* the top-3 candidate breeds
* a Grad-CAM heatmap showing which image regions drove the prediction

Run with:
    uv run streamlit run src/pet_classifier/app/streamlit_app.py

By default this loads the frozen MobileNetV2 checkpoint (the strongest
model per ``reports/member2_model_comparison.md``). Point
``CHECKPOINT_PATH`` at a different ``.pth`` file (and set ``ARCHITECTURE``
accordingly) to demo the custom CNN or a fine-tuned checkpoint instead.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from pet_classifier.config import CLASS_NAMES, ROOT_DIR
from pet_classifier.evaluation.checkpoints import load_checkpoint
from pet_classifier.evaluation.gradcam import GradCAMExplainer
from pet_classifier.training.trainer import get_device

# ---------------------------------------------------------------------------
# Configuration — change these two lines to demo a different checkpoint.
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "mobilenet_frozen.pth"
ARCHITECTURE = "mobilenet_v2"  # or "custom_cnn"

TOP_K = 3


@st.cache_resource
def load_model_and_explainer(checkpoint_path: str, architecture: str):
    """Load once per Streamlit session; cached across reruns/uploads."""
    device = get_device()
    model, checkpoint = load_checkpoint(
        checkpoint_path, architecture=architecture, device=device
    )
    class_names = checkpoint.get("class_names") or list(CLASS_NAMES)
    explainer = GradCAMExplainer(model, device=device)
    return model, explainer, class_names, device, checkpoint


def main() -> None:
    st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐾", layout="centered")
    st.title("🐾 Oxford-IIIT Pet Breed Classifier")
    st.caption(
        "Upload a photo of a cat or dog to predict its breed (37 classes) "
        "and see a Grad-CAM heatmap of the regions the model focused on."
    )

    if not Path(CHECKPOINT_PATH).exists():
        st.error(
            f"Checkpoint not found at `{CHECKPOINT_PATH}`.\n\n"
            "Train a model first (see the Quickstart in the README), or "
            "edit `CHECKPOINT_PATH` / `ARCHITECTURE` at the top of this "
            "file to point at a checkpoint you already have."
        )
        return

    with st.spinner("Loading model..."):
        model, explainer, class_names, device, checkpoint = load_model_and_explainer(
            str(CHECKPOINT_PATH), ARCHITECTURE
        )

    with st.sidebar:
        st.subheader("Model info")
        st.write(f"**Architecture:** {ARCHITECTURE}")
        st.write(f"**Device:** {device}")
        st.write(f"**Checkpoint epoch:** {checkpoint.get('epoch')}")
        val_metrics = checkpoint.get("val_metrics", {})
        if val_metrics:
            st.write(f"**Val accuracy:** {val_metrics.get('acc', float('nan')):.2%}")
            st.write(f"**Val top-3:** {val_metrics.get('topk', float('nan')):.2%}")

    uploaded_file = st.file_uploader(
        "Upload a pet photo", type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file is None:
        st.info("Upload a JPG/PNG image to get a prediction.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    col_original, col_gradcam = st.columns(2)
    with col_original:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with st.spinner("Running inference and Grad-CAM..."):
        # Top-K predictions.
        tensor = explainer.transform(image)
        with torch.no_grad():
            probs = torch.softmax(model(tensor.unsqueeze(0).to(device)), dim=1)[0]
        topk_probs, topk_indices = torch.topk(probs, k=TOP_K)

        # Grad-CAM for the top-1 prediction.
        overlay, predicted_idx, confidence = explainer.explain_pil_image(image)

    with col_gradcam:
        st.subheader("Grad-CAM")
        st.image(overlay, use_container_width=True)
        st.caption("Highlighted regions influenced the prediction most.")

    st.subheader(f"Prediction: {class_names[predicted_idx]}")
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    st.subheader(f"Top-{TOP_K} breeds")
    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        st.write(f"**{class_names[idx]}** — {prob:.1%}")
        st.progress(prob)


if __name__ == "__main__":
    main()
