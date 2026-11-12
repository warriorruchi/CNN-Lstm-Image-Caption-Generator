# 🖼️ Image Captioning with CNN-LSTM

> A hybrid deep learning system for automatic image captioning using ResNet18 encoder and LSTM decoder with Word2Vec embeddings.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0%2B-orange)](https://gradio.app/)

---

## 📋 Overview

This project implements an **image captioning system** that generates natural language descriptions for images. It combines computer vision and NLP techniques using a hybrid architecture:

- **Encoder**: ResNet18 (pretrained on ImageNet) for visual feature extraction
- **Decoder**: LSTM with Word2Vec embeddings for caption generation
- **Baseline**: Comparison with BLIP (State-of-the-Art) model

---

## 🏗️ Architecture

```
Image (224×224) → ResNet18 (Frozen) → Image Features (512D) → LSTM Decoder → Caption
                           ↓
                   Word2Vec (Frozen) → Word Embeddings (300D)
```

## 📊 Dataset

- **Flickr8k**: 8,091 images × 5 captions each
- **Augmentation**: Horizontal flipping (2×) → 16,182 samples
- **Split**: 80% Train | 10% Val | 10% Test

---

## 🚀 Features

- ✅ Short caption (20 tokens) & Long caption (40 tokens) models
- ✅ Pretrained ResNet18 & Word2Vec embeddings
- ✅ Interactive Gradio web interface
- ✅ Comparison with BLIP (SOTA) baseline
- ✅ BLEU-1 evaluation metric

### Model Performance

| Model | Max Length | BLEU-1 |
|-------|------------|--------|
| Short Caption | 20 | **0.2309** |
| Long Caption | 40 | **0.2151** |
| BLIP (SOTA) | 50 | **0.3345** |

---

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning

# Install dependencies
pip install -r requirements.txt

# Download required data
# - Flickr8k dataset → images/ directory
# - GoogleNews Word2Vec → project root
```

---

## 💻 Usage

### Training

```bash
jupyter notebook data_creation.ipynb      # Preprocess dataset
jupyter notebook hybrid.ipynb             # Train short model
jupyter notebook hybrid_long.ipynb        # Train long model
```

### Inference

```bash
python app.py                             # Launch web interface
```

Upload an image to compare captions from all three models simultaneously.

---

## 📁 Project Structure

```
├── data_creation.ipynb          # Dataset preprocessing
├── hybrid.ipynb                 # Short model training
├── hybrid_long.ipynb            # Long model training  
├── app.py                       # Gradio web app
├── report_file.pdf              # Complete report
└── requirements.txt             # Dependencies
```

---

## 📈 Results

### Sample Captions

| Image | Ground Truth | Our Model | BLIP (SOTA) |
|-------|--------------|-----------|-------------|
| 🐕 | "Dog running on beach" | ✅ "dog running in snow" | ✅ "dog on beach with ball" |
| 🎾 | "Tennis player hitting ball" | ❌ "man in basketball" | ✅ "man playing tennis" |

### Training Progress

```
Short Model: Train Loss 4.41 → 2.69 | Val Loss 4.22 → 3.85
Long Model:  Train Loss 4.97 → 3.46 | Val Loss 4.74 → 4.52
```

---

## ⚠️ Limitations

- No spatial attention mechanism
- Exposure bias causing repetition in long captions
- Small dataset (Flickr8k) limits generalization
- BLEU-1 metric ignores grammar/semantic quality

---

## 🔮 Future Work

- Add spatial attention (Show, Attend and Tell)
- Upgrade to Vision Transformers
- Implement beam search decoding
- Train on larger dataset (MS-COCO)
- Use better metrics (CIDEr, SPICE, BERTScore)

---

## 👥 Contributors

| Name | Roll No. | Contributions |
|------|----------|---------------|
| **Prashant Kumar Mishra** | 2447021 | Architecture design, Long model training, BLIP evaluation, Gradio interface |
| **Nikita Mewada** | 2447027 | Data preprocessing, Short model training, Loss visualization, Documentation |

---

## 📚 References

- [Show and Tell](https://arxiv.org/abs/1411.4555) - Vinyals et al. (2015)
- [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) - Xu et al. (2015)
- [BLIP](https://arxiv.org/abs/2201.12086) - Li et al. (2022)
- [Flickr8k Dataset](https://www.jair.org/index.php/jair/article/view/10833)

---

## 📄 License

MIT License

---

**Course**: Natural Language Processing (MC470405)  
**Institution**: National Institute of Technology Patna  
**Semester**: MCA (AI & IoT) 4th Sem

<div align="center">
  Made with ❤️ for NLP Course Project
</div>****
