import logging
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = "."
MAX_SEQUENCE_LENGTH = 128

EXAMPLE_COMPLAINTS = [
    "They keep calling me multiple times a day about a debt I already paid off.",
    "My credit report shows a late payment that was never actually late.",
    "The bank charged me an overdraft fee even though I had sufficient funds.",
    "My mortgage servicer keeps sending me conflicting payment amounts.",
    "Someone used my credit card without my permission and the company won't refund me.",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("complaint-classifier")


class ComplaintClassifier:
    """Wraps the tokenizer + model and exposes a simple predict() method."""

    def __init__(self, model_path: str):
        required_file = Path(model_path) / "config.json"
        if not required_file.exists():
            raise FileNotFoundError(
                f"Couldn't find 'config.json' under '{model_path}'. "
                "Make sure config.json, model.safetensors, tokenizer.json, and "
                "tokenizer_config.json are in that folder."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model from %s onto %s ...", model_path, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label
        self.num_classes = len(self.id2label)
        logger.info("Model ready — %d categories: %s", self.num_classes, list(self.id2label.values()))

    @torch.no_grad()
    def predict(self, text: str) -> list[tuple[str, float]]:
        """Returns [(category, probability), ...] sorted by probability, descending."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        ranked = sorted(
            ((self.id2label[i], float(p)) for i, p in enumerate(probs)),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked


classifier = ComplaintClassifier(MODEL_PATH)


def classify_complaint(complaint_text: str):
    if not complaint_text or not complaint_text.strip():
        return "*Enter a complaint above to see a prediction.*", None

    predictions = classifier.predict(complaint_text.strip())
    top_label, top_confidence = predictions[0]

    result_markdown = (
        f"### {top_label.replace('_', ' ').title()}\n"
        f"Confidence: **{top_confidence:.1%}**"
    )
    confidence_by_category = {label: prob for label, prob in predictions}
    return result_markdown, confidence_by_category



CUSTOM_CSS = """
#header { text-align: center; margin-bottom: 0.5rem; }
#subheader { text-align: center; color: var(--body-text-color-subdued); margin-bottom: 1.5rem; }
.gradio-container { max-width: 1000px !important; margin: auto; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=CUSTOM_CSS, title="Consumer Complaint Classifier") as demo:

    gr.Markdown("# 🏦 Consumer Complaint Classifier", elem_id="header")
    gr.Markdown(
        "Paste a customer complaint narrative and a fine-tuned DistilBERT model will predict "
        "its category, trained on real CFPB consumer complaint data.",
        elem_id="subheader",
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            complaint_input = gr.Textbox(
                lines=9,
                label="Complaint Narrative",
                placeholder="e.g. They keep calling me multiple times a day about a debt I already paid off...",
                autofocus=True,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(components=[complaint_input], value="Clear")
                submit_btn = gr.Button("Classify", variant="primary")

            gr.Examples(
                examples=EXAMPLE_COMPLAINTS,
                inputs=complaint_input,
                label="Try an example",
            )

        with gr.Column(scale=1):
            gr.Markdown("#### Prediction")
            result_output = gr.Markdown()
            gr.Markdown("#### Confidence by category")
            confidence_chart = gr.Label(num_top_classes=classifier.num_classes, show_label=False)

    gr.Markdown(
        f"<sub>Model: DistilBERT (fine-tuned) &nbsp;|&nbsp; "
        f"Categories: {classifier.num_classes} &nbsp;|&nbsp; "
        f"Device: {classifier.device}</sub>"
    )

    submit_btn.click(fn=classify_complaint, inputs=complaint_input, outputs=[result_output, confidence_chart])
    complaint_input.submit(fn=classify_complaint, inputs=complaint_input, outputs=[result_output, confidence_chart])


if __name__ == "__main__":
    demo.launch()