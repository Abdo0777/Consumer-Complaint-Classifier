# Consumer Complaint Classifier

An end-to-end NLP pipeline that classifies customer complaint narratives into one of five categories, comparing four different model architectures and deploying the best-performing one as an interactive web app.

## Overview

Financial institutions receive large volumes of customer complaints that need to be routed to the correct department. This project automates that routing by training a text classification model on real consumer complaint narratives, reducing manual triage effort.

**Categories predicted:**
- Credit Card
- Credit Reporting
- Debt Collection
- Mortgages and Loans
- Retail Banking

## Dataset

[Consumer Complaints Dataset for NLP](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp) (Kaggle) — ~162K labeled consumer complaint narratives, cleaned to ~87K rows after deduplication and class balancing.

## Pipeline

1. **Data cleaning** — removed placeholder rows, nulls, and duplicate narratives
2. **Class imbalance handling** — capped dominant classes, applied class weights during training
3. **Text preprocessing** — lowercasing, punctuation/number removal, stopword removal, lemmatization
4. **Tokenization & padding** — Keras `Tokenizer` with a 10,000-word vocabulary, sequences padded/truncated to 200 tokens
5. **Model training & comparison** — four architectures trained and evaluated on a held-out test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| SimpleRNN | 0.795 | 0.801 | 0.795 | 0.796 |
| LSTM | 0.836 | 0.837 | 0.836 | 0.836 |
| GRU | 0.843 | 0.844 | 0.843 | 0.843 |
| **DistilBERT (fine-tuned)** | **0.867** | **0.867** | **0.867** | **0.867** |

The fine-tuned DistilBERT transformer performed best, benefiting from pretrained language understanding that the from-scratch recurrent models don't have.

6. **Deployment** — best model served through a Gradio web app for live classification of new complaints

## Project Structure

```
consumer-complaint-classifier/
├── app.py                    # Gradio web app
├── config.json                # Model configuration + label mappings
├── model.safetensors           # Fine-tuned DistilBERT weights (tracked via Git LFS)
├── tokenizer.json
├── tokenizer_config.json
├── requirements.txt
└── README.md
```

## Running Locally

```bash
git clone https://github.com/Abdo0777/Consumer-Complaint-Classifier.git
cd Consumer-Complaint-Classifier

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
python app.py
```

Open the printed local URL (typically `http://127.0.0.1:7860`) in your browser.

## Example

**Input:**
> "They keep calling me multiple times a day about a debt I already paid off."

**Output:**
> Debt Collection — 97.7% confidence

## Tech Stack

- **TensorFlow / Keras** — SimpleRNN, LSTM, GRU models built from scratch
- **HuggingFace Transformers** — fine-tuned DistilBERT
- **Gradio** — interactive web deployment
- **scikit-learn** — evaluation metrics and preprocessing utilities
- **NLTK** — stopword removal and lemmatization

## Notes

- Model weights are tracked with **Git LFS** due to file size (~255MB). Make sure `git-lfs` is installed before cloning: `git lfs install`
