"""Interactive Gradio demo: upload a leaf photo, get a prediction + Grad-CAM overlay.

Usage:
    python app/demo.py                       # uses results/best_model.pt
    python app/demo.py --checkpoint path/to/best_model.pt --device cpu
"""
import argparse
import logging
import os
import sys
import threading
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageOps

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from dataset import build_transforms, eval_crop_box, preprocessing_from_checkpoint  # noqa: E402
from gradcam import GradCAM, overlay_heatmap  # noqa: E402
from model import build_model_from_checkpoint, get_last_conv_layer, load_checkpoint  # noqa: E402
from utils import get_device, pretty_class_name  # noqa: E402

logger = logging.getLogger("plant-demo")

DEFAULT_CHECKPOINT = REPO_ROOT / "results" / "best_model.pt"
TOP_K = 5
MIN_IMAGE_SIDE = 64  # px; smaller uploads are almost certainly not usable leaf photos
DISPLAY_MAX_SIDE = 512
# Display heuristics only. They are NOT calibrated and NOT an out-of-distribution detector.
LOW_TOP1_PROB = 0.60
LOW_MARGIN = 0.20


class Predictor:
    """Holds the model, its preprocessing, and a Grad-CAM explainer for one checkpoint."""

    def __init__(self, checkpoint_path, device):
        ckpt = load_checkpoint(str(checkpoint_path), map_location="cpu")
        self.model_name = ckpt["model_name"]
        self.class_names = ckpt["class_names"]
        self.preprocessing = preprocessing_from_checkpoint(ckpt)
        self.device = device
        self.model = build_model_from_checkpoint(ckpt, device)

        pp = self.preprocessing
        _, self.transform = build_transforms(pp["img_size"], pp["resize_size"], pp["mean"], pp["std"])
        self.gradcam = GradCAM(self.model, get_last_conv_layer(self.model, self.model_name))
        self._lock = threading.Lock()  # GradCAM keeps per-call state on the instance

    @property
    def crops(self):
        return sorted({pretty_class_name(c).split(" — ")[0] for c in self.class_names})

    def predict(self, image: Image.Image, top_k: int = TOP_K) -> dict:
        image = ImageOps.exif_transpose(image).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with self._lock:
            # One forward pass gives both the logits and the Grad-CAM for the top class
            cam, _ = self.gradcam(x)
            probs = torch.softmax(self.gradcam.logits[0].float(), dim=0).cpu()

        values, indices = torch.topk(probs, k=min(top_k, len(probs)))
        top = [(self.class_names[i], p) for p, i in zip(values.tolist(), indices.tolist())]

        # Grad-CAM is in the coordinates of the center crop the model saw, so overlay it there
        pp = self.preprocessing
        box = tuple(round(v) for v in eval_crop_box(image.width, image.height, pp["resize_size"], pp["img_size"]))
        model_view = image.crop(box)
        model_view.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))

        return {
            "top": top,
            "cam": cam,
            "crop_box": box,
            "context": draw_crop_box(image, box),
            "overlay": overlay_heatmap(model_view, cam),
        }


def draw_crop_box(image: Image.Image, box) -> Image.Image:
    """Downscaled copy of the upload with the analysed region outlined."""
    preview = image.copy()
    preview.thumbnail((DISPLAY_MAX_SIDE, DISPLAY_MAX_SIDE))
    scale = preview.width / image.width
    left, top, right, bottom = (v * scale for v in box)
    width = max(2, round(min(preview.size) / 120))
    ImageDraw.Draw(preview).rectangle((left, top, right - 1, bottom - 1), outline=(255, 214, 0), width=width)
    return preview


def summarize(top) -> str:
    (best_name, best_p), second_p = top[0], (top[1][1] if len(top) > 1 else 0.0)
    lines = [f"### {pretty_class_name(best_name)}",
             f"Top model probability: **{best_p:.1%}** (next best: {second_p:.1%})"]
    if best_p < LOW_TOP1_PROB or best_p - second_p < LOW_MARGIN:
        lines.append("\n⚠️ **Inconclusive** — the model is split between several classes. "
                     "Try a sharper, well-lit photo of a single leaf, or consult an expert.")
    lines.append("\n_Model probabilities are not calibrated confidence. A high value does not mean "
                 "the diagnosis is correct, and the model returns one of its 38 classes even for "
                 "photos of other plants or non-plant objects._")
    return "\n".join(lines)


HEADER = """
# 🌿 Plant Disease Classifier
Upload a photo of a **single crop leaf** to get the model's top-5 classes and a Grad-CAM heatmap
of the image regions that most influenced the top prediction.

> **Experimental research demo — not a diagnostic tool.** The model was trained and validated only on
> PlantVillage-style images (one detached leaf, plain background, controlled lighting). Its accuracy on
> field photos has **not** been measured. Do not make treatment or crop-management decisions from it;
> confirm any suspected disease with a qualified agronomist or plant-pathology lab.
"""

EXPLAINER = """
**Prediction.** The scores are softmax outputs of a {model} network. They sum to 100% across the 38
known classes, so they say how the model ranks *those classes* — not how likely the answer is to be right.
Softmax scores from networks like this are often over-confident, especially on unfamiliar images.

**Grad-CAM.** The model only sees the **central square** of your photo (outlined in yellow) after resizing
the shorter side. The heatmap is drawn on exactly that region. Red areas pushed the score of the top class up
the most. The map has a 7×7 resolution, so it is coarse; it shows where the model looked, not whether it
looked for the right reason. Highlighted background or leaf edges are a warning sign.

**What is out of scope.** Unsupported crops, multiple leaves, whole plants, fruit, stems, pests, nutrient
deficiencies, and non-plant images. The demo cannot reliably detect these cases.
"""


def analyze_image(predictor: Predictor, image):
    """Gradio callback body: validates the upload and returns (summary, label dict, context, overlay)."""
    if image is None:
        raise gr.Error("Please upload a leaf photo first.")
    if min(image.size) < MIN_IMAGE_SIDE:
        raise gr.Error(f"Image is too small ({image.width}×{image.height}px). "
                       f"Please upload a photo at least {MIN_IMAGE_SIDE}px on each side.")
    try:
        result = predictor.predict(image)
    except Exception:
        logger.exception("Inference failed")
        raise gr.Error("Sorry, this image could not be analysed. Please try a different JPG or PNG photo.")
    label = {pretty_class_name(name): p for name, p in result["top"]}
    return summarize(result["top"]), label, result["context"], result["overlay"]


def build_demo(predictor: Predictor) -> gr.Blocks:
    tips = (
        "**Tips for a usable photo**\n"
        "- One leaf, filling most of the frame and **centered** (edges get cropped)\n"
        "- Even lighting, in focus, minimal background\n"
        f"- Supported crops: {', '.join(predictor.crops)}\n"
        "- JPG/PNG/WebP, up to 10 MB"
    )

    def analyze(image):
        return analyze_image(predictor, image)

    with gr.Blocks(title="Plant Disease Classifier", theme=gr.themes.Soft(primary_hue="green"),
                   analytics_enabled=False) as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(type="pil", label="Leaf photo", sources=["upload", "clipboard"])
                run = gr.Button("Analyse leaf", variant="primary")
                gr.Markdown(tips)
            with gr.Column(scale=1):
                summary = gr.Markdown("_Upload a photo and press **Analyse leaf**._")
                probs = gr.Label(num_top_classes=TOP_K, label="Top-5 model probabilities (not calibrated)")
        with gr.Row():
            context = gr.Image(label="Region analysed by the model (yellow box)", interactive=False)
            overlay = gr.Image(label="Grad-CAM on the analysed region", interactive=False)
        with gr.Accordion("How to read these results", open=False):
            gr.Markdown(EXPLAINER.format(model=predictor.model_name))
        gr.Markdown(f"<small>Model: {predictor.model_name} · {len(predictor.class_names)} classes · "
                    f"device: {predictor.device}</small>")

        outputs = [summary, probs, context, overlay]
        run.click(analyze, inputs=image_in, outputs=outputs, api_name="analyze")
        image_in.upload(analyze, inputs=image_in, outputs=outputs, api_name=False)  # same fn; one API endpoint
    return demo


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=os.environ.get("CHECKPOINT_PATH", str(DEFAULT_CHECKPOINT)))
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    # Hosting platforms such as Render set PORT and need 0.0.0.0; locally we stay on localhost.
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME")
                        or ("0.0.0.0" if "PORT" in os.environ else "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    return parser.parse_args(argv)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint}. Pass --checkpoint or set CHECKPOINT_PATH.")
    try:
        predictor = Predictor(checkpoint, get_device(args.device))
    except Exception as exc:
        sys.exit(f"Could not load checkpoint {checkpoint}: {exc}")
    logger.info("Loaded %s (%d classes) on %s", predictor.model_name, len(predictor.class_names), predictor.device)

    demo = build_demo(predictor)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_file_size="10mb"
    )


if __name__ == "__main__":
    main()
