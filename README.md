# Banking77 Intent Classification - LoRA Fine-Tuning 

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Parameter-efficient fine-tuning of DistilBERT using LoRA**  
> Achieves 96% of full fine-tune performance with only 1.92% trainable parameters

##  Project Overview

Fine-tuned DistilBERT on the Banking77 dataset (77 intent classes) using Low-Rank Adaptation (LoRA) to reduce training costs while maintaining high accuracy for banking chatbot intent classification.

**Key Results:**
- **88.25% test accuracy** (vs 91.92% full fine-tune)
- **1.31M trainable params** (1.92% of total 68.3M)
- **5MB checkpoint** (vs 268MB full model)
- **~15ms inference latency** (production-ready)

---

##  Quick Start

### Installation
```bash
pip install transformers==4.57.1 datasets==4.2.0 evaluate==0.4.6 peft==0.17.1 accelerate==1.11.0
```

### Run Training
```bash
# Open notebook
jupyter notebook Test_A.ipynb

# Or use Colab
# Upload Test_A.ipynb → Runtime → Run all
```

### Load Fine-Tuned Model
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Load base + LoRA weights
base = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=77
)
model = PeftModel.from_pretrained(base, "./results/lora_peft")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Inference
text = "I lost my debit card"
inputs = tokenizer(text, return_tensors="pt")
pred_idx = model(**inputs).logits.argmax().item()
print(f"Intent: {label_names[pred_idx]}")
```

---

## 📊 Results Summary

| Method | Val Acc | Test Acc | Trainable % | Training Time |
|--------|---------|----------|-------------|---------------|
| Full Fine-Tune | 91.11% | 91.92% | 100% | ~2.7 hrs |
| **LoRA (Best)** | **88.31%** | **88.25%** | **1.92%** | **~3.0 hrs** |
| LoRA + Gradual Unfreeze | 85.31% | 86.04% | 1.92% | ~3.8 hrs |

### Hyperparameter Optimization

| Config | lr | r | α | Val Acc | Test Acc |
|--------|-----|---|---|---------|----------|
| **1 (Best)** | **1e-4** | **8** | **32** | **88.31%** | **88.25%** |
| 2 | 5e-5 | 16 | 16 | 78.42% | 77.95% |
| 3 | 1e-5 | 8 | 16 | 40.86% | 37.73% |

**Finding:** LoRA requires higher learning rates (1e-4) than full fine-tuning (5e-5) due to sparse parameter updates.

---

## 🎬 Demo

![Learning Curves](learning_curves.png)

**Video Walkthrough:** [Link to video or embed if <100MB]

---

##📁 Repository Structure
```
.
├── Test_A.ipynb              # Main training notebook
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── technical_report.pdf      # Detailed methodology & results
├── learning_curves.png       # Training visualization

```

---

## 🛠️ Technical Details

**Model Architecture:**
- Base: `distilbert-base-uncased` (66M params)
- LoRA config: r=8, α=32, dropout=0.05
- Target modules: `q_lin`, `k_lin`, `v_lin`, `out_lin`, `ffn.lin1`, `ffn.lin2`

**Training Setup:**
- Platform: Google Colab (T4 GPU)
- Framework: PyTorch + Hugging Face Transformers
- Training time: ~3 hours for 4 epochs

**Dataset:**
- Banking77: 13,083 queries, 77 intent classes
- Split: 9,002 train / 1,001 val / 3,080 test

---

## 📈 Performance Analysis

**Strengths:**
-  High accuracy on common banking intents
-  Fast inference (<20ms per query)
-  Memory-efficient (fits on single T4 GPU)

**Known Limitations:**
- Confuses semantically similar intents (e.g., "transfer_into_account" vs "receiving_money")
- Struggles with very short queries (<5 tokens)
- 3.67% accuracy gap vs full fine-tune

**Suggested Improvements:**
1. Data augmentation (back-translation for confused pairs)
2. Entity normalization (`<AMOUNT>`, `<DATE>` placeholders)
3. Increase max_length to 128 for queries with metadata

---

##  Contributing

This is a course project, but feedback is welcome! Open an issue if you spot bugs or have suggestions.

---

##  License

MIT License - see LICENSE file for details.

---


- Banking77 dataset by Casanueva et al.
- Hugging Face for Transformers & PEFT libraries
- LoRA paper by Hu et al. (2021)
