# -*- coding: utf-8 -*-
"""app.py - Simple Image Captioning Interface (CPU/GPU Compatible)"""

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr
from PIL import Image
import numpy as np
import nltk
import re
import os

nltk.download('punkt', quiet=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load Short Model
short_checkpoint = torch.load('image_captioning_model.pt', map_location=torch.device('cpu'), weights_only=False)

class ShortModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.image_projection = nn.Linear(in_features, hidden_dim)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.3)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, vocab_size)
        )
    
    def generate(self, images, word_to_idx, max_len=20):
        self.eval()
        with torch.no_grad():
            batch_size = images.size(0)
            image_features = self.encoder(images)
            image_features = self.image_projection(image_features)
            h0 = image_features.unsqueeze(0).repeat(2, 1, 1)
            c0 = torch.zeros_like(h0)
            generated = []
            current = self.word_embedding(torch.full((batch_size, 1), word_to_idx['<start>']).to(images.device))
            for _ in range(max_len):
                lstm_out, (h0, c0) = self.lstm(current, (h0, c0))
                out = self.output_layer(lstm_out.squeeze(1))
                next_word = torch.argmax(out, dim=-1, keepdim=True)
                generated.append(next_word)
                current = self.word_embedding(next_word)
                if (next_word == word_to_idx['<end>']).all():
                    break
            return torch.cat(generated, dim=1)

short_model = ShortModel(short_checkpoint['vocab_size'])
short_model.load_state_dict(short_checkpoint['model_state_dict'])
short_model = short_model.to(device)
short_model.eval()
short_word_to_idx = short_checkpoint['word_to_idx']
short_idx_to_word = short_checkpoint['idx_to_word']
print(f"✅ Short model loaded")

# Load Long Model
long_checkpoint = torch.load('image_captioning_model_long.pt', map_location=torch.device('cpu'), weights_only=False)

class LongModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        for param in self.encoder.parameters():
            param.requires_grad = False
        in_features = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.image_projection = nn.Linear(in_features, hidden_dim)
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.3)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, vocab_size)
        )
    
    def generate(self, images, word_to_idx, max_len=35):
        self.eval()
        with torch.no_grad():
            batch_size = images.size(0)
            image_features = self.encoder(images)
            image_features = self.image_projection(image_features)
            h0 = image_features.unsqueeze(0).repeat(3, 1, 1)
            c0 = torch.zeros_like(h0)
            generated = []
            current = self.word_embedding(torch.full((batch_size, 1), word_to_idx['<start>']).to(images.device))
            for _ in range(max_len):
                lstm_out, (h0, c0) = self.lstm(current, (h0, c0))
                out = self.output_layer(lstm_out.squeeze(1))
                next_word = torch.argmax(out, dim=-1, keepdim=True)
                generated.append(next_word)
                current = self.word_embedding(next_word)
                if (next_word == word_to_idx['<end>']).all():
                    break
            return torch.cat(generated, dim=1)

long_model = LongModel(long_checkpoint['vocab_size'])
long_model.load_state_dict(long_checkpoint['model_state_dict'])
long_model = long_model.to(device)
long_model.eval()
long_word_to_idx = long_checkpoint['word_to_idx']
long_idx_to_word = long_checkpoint['idx_to_word']
print(f"✅ Long model loaded")

# Load BLIP Model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = blip_model.to(device)
blip_model.eval()
print(f"✅ BLIP model loaded")


def indices_to_text(indices, idx_to_word):
    words = []
    for idx in indices[0]:
        if idx.item() == idx_to_word.get('<end>', 2):
            break
        if idx.item() not in [0, 1, 2, 3]:
            words.append(idx_to_word[idx.item()])
    return ' '.join(words)

def transform_image(img):
    """Transform PIL image to model input tensor"""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0).to(device)


def generate_captions(image):
    if image is None:
        return "Please upload an image first", "Please upload an image first", "Please upload an image first"
    
    # Transform image
    img_tensor = transform_image(image)
    
    # Generate Short Caption
    with torch.no_grad():
        short_indices = short_model.generate(img_tensor, short_word_to_idx, max_len=20)
    short_caption = indices_to_text(short_indices, short_idx_to_word)
    
    # Generate Long Caption
    with torch.no_grad():
        long_indices = long_model.generate(img_tensor, long_word_to_idx, max_len=35)
    long_caption = indices_to_text(long_indices, long_idx_to_word)
    
    # Generate BLIP Caption
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50)
    blip_caption = blip_processor.decode(out[0], skip_special_tokens=True)
    
    return short_caption, long_caption, blip_caption

# ============================================
# GRADIO INTERFACE
# ============================================

with gr.Blocks(theme='soft') as demo:
    gr.Markdown("""
    # 🖼️ Image Captioning System
    ### Compare captions from Custom Short Model, Custom Long Model, and BLIP (State-of-the-Art)
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📤 Upload Image (Drag & Drop)")
            with gr.Row():
                generate_btn = gr.Button("🎯 Generate Captions", variant="primary")
                clear_btn = gr.Button("🔄 Start Over", variant="secondary")
        
        with gr.Column():
            gr.Markdown("### 📝 Generated Captions")
            short_output = gr.Textbox(label="🔵 Custom Short Model", lines=2, interactive=False)
            long_output = gr.Textbox(label="🟠 Custom Long Model", lines=2, interactive=False)
            blip_output = gr.Textbox(label="🟢 BLIP (SOTA)", lines=2, interactive=False)
    
    # Wire up buttons
    generate_btn.click(
        generate_captions, 
        inputs=[image_input], 
        outputs=[short_output, long_output, blip_output]
    )
    
    clear_btn.click(
        lambda: [None, "", "", ""],
        inputs=[],
        outputs=[image_input, short_output, long_output, blip_output]
    )
    
    gr.Markdown("""
    ---
    **Instructions:**
    1. Drag & drop or click to upload an image
    2. Click "Generate Captions" to see predictions from all three models
    3. Click "Start Over" to clear and upload a new image
    """)

if __name__ == "__main__":
    demo.launch(share=True)