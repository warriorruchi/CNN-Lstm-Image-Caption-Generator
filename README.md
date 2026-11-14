
# 🖼️ Image Captioning with CNN-LSTM

> A Hybrid Deep Learning System for Automatic Image Caption Generation using ResNet18, LSTM Decoder and Word2Vec Embeddings.

**Maintained and Modified by: Ruchi Upadhyay**

---

## 📋 Overview

This project implements an Image Captioning System that generates natural language descriptions for images using Deep Learning.

It combines Computer Vision and Natural Language Processing through a hybrid architecture.

- **Encoder:** ResNet18 for visual feature extraction
- **Decoder:** LSTM for caption generation
- **Word Embeddings:** Word2Vec
- **Dataset:** Flickr8k

---

## 🏗️ Architecture

```text
Image (224×224)
       ↓
ResNet18 Encoder
       ↓
Image Features (512D)
       ↓
LSTM Decoder
       ↓
Generated Caption
```

---

## 📊 Dataset

- Flickr8k Dataset
- 8,091 original images
- 40,460 image captions
- Horizontal flipping for data augmentation
- 80% Training, 10% Validation, 10% Testing

---

## 🚀 Features

- Short Caption Generation (20 tokens)
- Long Caption Generation (40 tokens)
- Pretrained ResNet18 Encoder
- Word2Vec Embeddings
- Gradio Web Interface
- BLEU-1 Evaluation

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- NLTK
- Pandas
- NumPy
- Gradio
- Jupyter Notebook

---

## 📁 Project Structure

```text
CNN-Lstm-Image-Caption-Generator/
│
├── data_creation.ipynb
├── hybrid.ipynb
├── hybrid_long.ipynb
├── app.py
├── captions.txt
├── Flickr8k.token.txt
├── images/
├── README.md
└── report_file.pdf
```

---

## 🛠️ Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/warriorruchi/CNN-Lstm-Image-Caption-Generator.git

cd CNN-Lstm-Image-Caption-Generator
```

### Step 2: Install Dependencies

```bash
pip install torch torchvision
pip install pandas numpy nltk pillow gradio
```

### Step 3: Prepare Dataset

- Download Flickr8k images.
- Prepare the captions file.
- Download GoogleNews Word2Vec embeddings.

---

## 💻 Usage

### Training

```bash
jupyter notebook data_creation.ipynb
jupyter notebook hybrid.ipynb
jupyter notebook hybrid_long.ipynb
```

### Run Application

```bash
python app.py
```

---

## 📈 Evaluation

The project uses BLEU-1 as an evaluation metric to measure the quality of generated captions.

Actual model performance will be updated after successful training and evaluation.

---

## ⚠️ Limitations

- No spatial attention mechanism.
- Long captions may contain repetition.
- Flickr8k is a relatively small dataset.
- BLEU-1 does not fully measure semantic quality.

---

## 🔮 Future Improvements

- Implement spatial attention.
- Explore Vision Transformers.
- Implement beam search decoding.
- Train using larger datasets.
- Evaluate using CIDEr and SPICE.

---

## 👩‍💻 Project Maintainer

**Ruchi Upadhyay**

GitHub: [@warriorruchi](https://github.com/warriorruchi)

Project Repository: [CNN-LSTM Image Caption Generator](https://github.com/warriorruchi/CNN-Lstm-Image-Caption-Generator)

---

## 🙌 Original Project Acknowledgement

This project is adapted from the original work of:

- Ruchi Upadhyay
- Riddhima Pant

Original contributions are acknowledged.

---

## 📚 References

- [Show and Tell (2015)](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell (2015)](https://arxiv.org/abs/1502.03044)
- [BLIP (2022)](https://arxiv.org/abs/2201.12086)
- [Flickr8k Dataset](https://www.jair.org/index.php/jair/article/view/10833)

---

**Made with ❤️ using Deep Learning and NLP.**
