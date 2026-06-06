# 🎬 BERT Sentiment Analysis — IMDb Fine-Tuning

Fine-tune `bert-base-uncased` on the [IMDb dataset](https://huggingface.co/datasets/stanfordnlp/imdb) for binary sentiment classification (positive / negative). Achieves **~90% accuracy** after a single epoch.

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | **90.4%** |
| F1-Score | **0.901** |
| Eval Loss | 0.288 |
| Training Epochs | 1 |
| Train Samples | 2,000 |
| Eval Samples | 500 |

---

## 🗂️ Project Structure

```
bert-sentiment/
├── bert_finetuning.ipynb   # Main training notebook
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone & install

```bash
git clone https://github.com/your-username/bert-sentiment.git
cd bert-sentiment
pip install -r requirements.txt
```

### 2. Run the notebook

```bash
jupyter notebook bert_finetuning.ipynb
```

Or run end-to-end without a notebook UI:

```bash
jupyter nbconvert --to script bert_finetuning.ipynb
python bert_finetuning.py
```

---

## ⚙️ Configuration

Key hyperparameters (in `TrainingArguments`):

| Parameter | Value |
|---|---|
| Learning Rate | `2e-5` |
| Batch Size (train/eval) | `8` |
| Epochs | `1` |
| Weight Decay | `0.01` |
| Eval Strategy | `epoch` |

---

## 🧠 Model

- **Base model:** [`bert-base-uncased`](https://huggingface.co/bert-base-uncased)
- **Task head:** Linear classifier on `[CLS]` token (`num_labels=2`)
- **Tokenizer:** `BertTokenizer` with `max_length` padding and truncation

---

## 📋 Requirements

See `requirements.txt`. Core dependencies:

- `transformers >= 4.40`
- `datasets >= 2.0`
- `torch >= 2.0`
- `scikit-learn`

---

## 📌 Notes

- Training and eval use random subsets (2K / 500 samples) for fast iteration. Remove `.select()` calls to train on the full dataset (~25K samples).
- No GPU is required, but training on CPU will be significantly slower. The full dataset with 3 epochs takes ~2–3 hours on a T4 GPU.
- The `classifier` head weights are randomly initialized (expected warning at load time) and learned from scratch during fine-tuning.

---

## 📄 License

MIT
