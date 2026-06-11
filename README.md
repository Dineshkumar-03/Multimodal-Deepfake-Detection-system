# Multimodal Deepfake Detection System

A deep learning system that detects AI-generated or manipulated content across **audio, video, and image** modalities using a combined CNN + LSTM architecture.

---

## Overview

With the rise of synthetic media, verifying the authenticity of digital content has become critical. This project tackles that problem by building a multimodal detection pipeline that analyzes visual, temporal, and audio signals together — rather than treating each modality in isolation.

**Key capabilities:**
- Detects manipulated faces in images and video frames
- Identifies audio-visual inconsistencies in video content
- Classifies content as real or fake with confidence scoring

---

## Architecture

```
Input (Image / Video / Audio)
        │
        ├── Visual stream  →  CNN  (spatial feature extraction)
        │
        └── Temporal/Audio stream  →  LSTM  (sequence analysis)
                │
         Feature Fusion Layer
                │
         Fully Connected + Sigmoid
                │
         Real / Fake Classification
```

- **CNN** extracts spatial features from frames (face regions, compression artifacts, blending boundaries)
- **LSTM** models temporal inconsistencies across frame sequences and audio signals
- **Fusion layer** combines both streams for final classification

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.x |
| Deep Learning | CNN, LSTM |
| Data Processing | NumPy, Pandas |
| Preprocessing | OpenCV |
| Model Training | Scikit-learn |

---

## Results

> Approximate results on test set:

| Metric | Score |
|--------|-------|
| Accuracy | ~90%+ |
| Precision | ~88% |
| Recall | ~87% |

*Results may vary depending on dataset and training configuration.*

---

## Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Dineshkumar-03/multimodal-deepfake-detection.git
cd multimodal-deepfake-detection

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Run detection on a single file:**

```bash
python detect.py --input path/to/file.mp4
```

**Run on an image:**

```bash
python detect.py --input path/to/image.jpg --mode image
```

**Output:** The model returns a `Real` / `Fake` label along with a confidence score.

---

## Project Structure

```
multimodal-deepfake-detection/
│
├── models/             # Trained model weights
├── preprocessing/      # Data cleaning and augmentation scripts
├── detect.py           # Main inference script
├── train.py            # Model training script
├── requirements.txt
└── README.md
```

---

## Author

**Dinesh Kumar K**
[LinkedIn](https://linkedin.com/in/dinesh-kumar-k-600073360) · [GitHub](https://github.com/Dineshkumar-03)
