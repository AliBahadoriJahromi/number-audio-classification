# 🔊 Audio Classification Using CNN & MelSpectrogram

This project implements a **Convolutional Neural Network (CNN)** for classifying audio files using **Mel Spectrograms** as input features. The model is built with **PyTorch** and **Torchaudio** and is designed for tasks like classifying spoken digits, commands, or environmental sounds.

---
## 🎧 Workflow

1. **Audio Loading**
   - `AudioDataset.py` handles loading and labeling `.wav` files from directories.

2. **Feature Extraction**
   - Each audio file is transformed into a **Mel Spectrogram** using `torchaudio.transforms.MelSpectrogram`.

3. **Model Architecture**
   - `MODEL.py` contains a CNN that takes Mel Spectrograms as input and performs classification.

4. **Training & Evaluation**
   - Use `train.py` to train the model.

---
### Install dependencies

```bash
pip install -r requirements.txt
```
