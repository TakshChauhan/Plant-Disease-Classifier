"""Interactive Gradio demo: upload a leaf photo, get a prediction + Grad-CAM overlay.

Usage:
    python app/demo.py --checkpoint results/best_model.pt
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from dataset import build_transforms  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from model import get_last_conv_layer, get_model  # noqa: E402
from utils import get_device  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default="results/best_model.pt")
parser.add_argument("--share", action="store_true")
args = parser.parse_args()

device = get_device()
ckpt = torch.load(args.checkpoint, map_location=device)
class_names = ckpt["class_names"]
model_name = ckpt["model_name"]

model, _ = get_model(model_name, num_classes=len(class_names))
model.load_state_dict(ckpt["model_state"])
model.to(device).eval()

target_layer = get_last_conv_layer(model, model_name)
gradcam = GradCAM(model, target_layer)
_, eval_tfms = build_transforms()


def predict(image):
    if image is None:
        return {}, None

    input_tensor = eval_tfms(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top5 = torch.topk(probs, k=5)
    confidences = {class_names[i]: float(p) for p, i in zip(top5.values, top5.indices)}

    # Grad-CAM needs gradients, so run it separately from the no_grad inference pass above
    input_tensor_grad = eval_tfms(image.convert("RGB")).unsqueeze(0).to(device)
    top_class = int(top5.indices[0])
    cam, _ = gradcam(input_tensor_grad, target_class=top_class)
    overlay = overlay_heatmap(image, cam)

    return confidences, overlay


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Leaf photo"),
    outputs=[
    gr.JSON(label="Prediction"),
    gr.Image(label="Grad-CAM"),
],
    
    title="🌿 Plant Disease Classifier",
    description=(
        f"Model: {model_name} | {len(class_names)} classes. "
        "Upload a crop leaf photo to get a disease prediction and a Grad-CAM heatmap "
        "showing which regions influenced the decision."
    ),
)
import os

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )