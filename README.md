# 🎨 Image Generation with GANs (DCGAN)

## 📌 Overview
This project demonstrates image generation using a Deep Convolutional GAN (DCGAN) trained on the CelebA dataset. The generator learns to synthesize realistic human face images.

## 🛠️ Tools & Tech
- Python
- PyTorch
- DCGAN architecture
- CelebA Dataset

## 🔍 Key Features
- Realistic image synthesis
- Custom training loop with generator and discriminator
- Result visualizations every few epochs

## 📊 Results
- High-quality face generation after training
- Discriminator and generator loss tracking

## 🚀 How to Run
```bash
pip install torch torchvision matplotlib
python train_dcgan.py
```

## 📁 Folder Structure
- `dcgan.py`: Generator and Discriminator models
- `train_dcgan.py`: Training loop
- `results/`: Generated images from training
