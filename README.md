# SignBridge

> Real-time sign language captioning for video calls using hand landmark detection and deep learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com/)

## ⚠️ Important: Training Required

**This is a research framework, not a pre-trained application.** To use SignBridge for inference, you must:

1. Obtain or prepare a sign language dataset (e.g., WLASL)
2. Preprocess the data using our tools
3. Train an LSTM or custom model
4. Deploy the trained model to the backend

We provide **complete end-to-end training scripts** and support multiple data sources. See [Model Training](#model-training) section.

## Overview

SignBridge is an open-source framework for real-time sign language recognition in video calls. It combines:

- **Hand detection**: MediaPipe hand landmark extraction (21 joints, 3D coordinates)
- **Sequence modeling**: PyTorch LSTM classifier with temporal awareness
- **Live captioning**: Chrome extension overlay for Google Meet, Zoom, and other platforms
- **Training pipeline**: Complete data preprocessing and model training utilities

## Status & Roadmap

### ✅ Completed
- FastAPI backend with hand landmark extraction
- Chrome extension (Manifest V3) for video call integration
- Full training pipeline (dataset prep, model training, inference)
- Docker support for deployment
- Comprehensive documentation

### 📋 In Progress / Planned
- [ ] Pre-trained models on common sign languages (ASL, BSL, LSF)
- [ ] Multi-hand support
- [ ] Fingerspelling recognition (letter-level classification)
- [ ] Transformer-based models for improved accuracy
- [ ] Mobile extensions (Firefox, Safari)
- [ ] Real-time video demo script

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Nightxelf/SignBridge.git
cd SignBridge

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### 3. Load the Extension

```bash
cd extension
npm install
npm run build
```

Then:
- Open `chrome://extensions`
- Enable "Developer mode"
- Click "Load unpacked" → select `extension/dist/`

### 4. Train a Model ⭐ **REQUIRED**

**Option A: Demo mode (no external data)**
```bash
python training/run_training.py --demo --epochs 1 --max-items 4
```

**Option B: WLASL dataset**
```bash
python training/run_training.py \
  --json training/dataset/metadata/WLASL_v0.3.json \
  --raw-root training/dataset/raw \
  --out-root training/dataset/processed \
  --download \
  --epochs 8 \
  --batch-size 16
```

**Option C: Pre-extracted landmarks (5GB+ ZIP)**
```bash
# Extract preprocessed dataset
unzip wlasl_preprocessed.zip -d training/dataset/processed/

# Train directly
python training/train.py \
  --data-dir training/dataset/processed \
  --epochs 8 \
  --batch-size 16
```

After training, the model checkpoint will be saved to `backend/checkpoints/latest.pt` and used automatically by the backend.

## Project Structure

```
SignBridge/
├── backend/                    # FastAPI inference server
│   ├── app.py                  # Main FastAPI application
│   ├── inference.py            # Model loading & prediction logic
│   ├── mediapipe_handler.py    # Hand landmark extraction
│   ├── models/
│   │   └── lstm_model.py       # LSTM architecture
│   ├── routes/
│   │   └── predict.py          # API endpoints
│   └── checkpoints/            # Trained model directory (created after training)
│
├── training/                   # Model training pipeline
│   ├── train.py                # Main training loop
│   ├── wlasl_prepare.py        # WLASL dataset preprocessing
│   ├── dataset.py              # PyTorch dataset class
│   ├── run_training.py         # End-to-end orchestrator
│   └── dataset/
│       ├── metadata/           # Dataset JSON manifests
│       ├── raw/                # Raw video files (created after download)
│       └── processed/          # Preprocessed landmarks (.npz files)
│
├── extension/                  # Chrome extension (React + Vite)
│   ├── manifest.json           # Manifest V3
│   ├── src/
│   │   ├── App.jsx             # React popup UI
│   │   ├── content/
│   │   │   └── contentScript.js # Video call integration
│   │   └── background.js       # Service worker
│   └── dist/                   # Built extension (load unpacked into Chrome)
│
├── docs/                       # Architecture & design documentation
├── Dockerfile
├── requirements.txt
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## API Reference

### GET `/api/health`
Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

### POST `/api/predict`
Classify from raw landmark sequence.

**Request:**
```json
{
  "landmarks": [[x0, y0, z0, ..., x20, y20, z20]]
}
```

**Response:**
```json
{
  "prediction": "hello",
  "confidence": 0.87
}
```

### POST `/api/extract`
Extract landmarks from image and classify in one call.

**Request:** Multipart form with image file parameter `file`

**Response:**
```json
{
  "prediction": "thank_you",
  "confidence": 0.92,
  "bbox": [0.1, 0.2, 0.9, 0.8]
}
```

## Model Training

### Dataset Options

**1. WLASL (Word-Level American Sign Language)**
- ~2000+ words with video samples
- Automatically downloadable via yt-dlp
- Well-structured JSON metadata

```bash
python training/run_training.py \
  --json training/dataset/metadata/WLASL_v0.3.json \
  --download \
  --max-items 512
```

**2. Pre-extracted Landmarks**
- Fastest option (no video processing)
- Support `.npz` files with normalized landmarks
- Ideal for large-scale training

```bash
python training/train.py --data-dir training/dataset/processed --max-samples 2000
```

**3. Custom Datasets**
- Prepare videos in any format
- Use `MediaPipeHandler` for landmark extraction
- Save as `.npz` with "landmarks" key

**4. Demo Mode** (for testing pipeline)
- Generates synthetic data
- No external dependencies
- ~4 samples, runs in <1 minute

### Training Configuration

```bash
python training/run_training.py \
  --json <path to dataset JSON> \
  --raw-root <video directory> \
  --out-root <processed directory> \
  --epochs 8 \
  --batch-size 16 \
  --max-samples 512 \
  --download          # Automatically download videos from URLs
  --demo              # Generate synthetic data instead
```

**Parameters:**
- `--epochs`: Training epochs (default: 8)
- `--batch-size`: Batch size (default: 16)
- `--max-samples`: Limit training samples (default: 512)
- `--download`: Download videos from metadata URLs
- `--demo`: Use synthetic data instead of real videos

### Training Output

After training, the pipeline produces:

```
training/dataset/processed/
├── label_map.json         # Class ID → name mapping
├── label_names.json       # Name → ID mapping
├── mapping.json           # Video file → class ID
└── <video_id>.npz         # Landmark sequences
```

And saves model to:
```
backend/checkpoints/latest.pt  # Best model checkpoint
```

## Architecture

### Backend (FastAPI)

- Lightweight, stateless inference server
- Hand landmark extraction via MediaPipe (fallback to skin detection if unavailable)
- CORS-enabled for cross-origin requests
- CPU and GPU support

### Extension (React + Vite)

- Manifest V3 content script for video call pages
- Real-time frame capture and API requests
- Status overlay with confidence scores
- Persistent backend URL configuration

### ML Pipeline

**Feature Extraction:**
- MediaPipe extracts 21 hand joints (x, y, z per joint) = 63 features
- Normalized to [0, 1] range
- Sequence length: variable (padded to maximum)

**Model Architecture:**
- Input: Variable-length landmark sequences
- 2-layer LSTM (128 hidden units)
- Output: Softmax classification over vocab size
- Loss: Cross-entropy
- Optimizer: Adam

**Inference:**
- Single forward pass through LSTM
- Returns predicted class and confidence

## Configuration

Set environment variables:

```bash
export SIGNBRIDGE_MODEL_PATH=backend/checkpoints/latest.pt
export SIGNBRIDGE_LABEL_MAP=backend/label_map.json
export SIGNBRIDGE_BACKEND_URL=http://localhost:8000
```

## Docker Deployment

```bash
# Build
docker build -t signbridge:latest .

# Run
docker run -p 8000:8000 signbridge:latest
```

The Dockerfile includes all dependencies and runs the FastAPI server on port 8000.

## Performance Notes

**Training Time (reference):**
- Demo mode: <1 minute (4 samples, 1 epoch)
- 100 samples, 5 epochs: ~5 minutes (CPU)
- 512 samples, 8 epochs: ~2 hours (CPU) / ~20 minutes (GPU)

**Inference:**
- Landmark extraction: 20-50ms per frame (CPU)
- Model forward pass: 1-5ms per frame (CPU)
- Overall: ~30-50ms per frame end-to-end

**Hardware Requirements:**
- CPU: Intel i5+, AMD Ryzen 5+ for reasonable speed
- GPU: NVIDIA GTX 1060+ recommended for training
- RAM: 8GB minimum, 16GB+ recommended

## Known Limitations

- Single-hand detection only (multi-hand support in progress)
- Requires well-lit environment for accurate landmark extraction
- Model accuracy heavily dependent on training dataset coverage and quality
- Chrome/Chromium extension only (Firefox/Safari support planned)
- Inference disabled by default until model is trained

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup
- Code style guidelines
- Testing requirements
- Areas for contribution

**High-priority areas:**
- Transformer-based models
- Multi-hand support
- Fingerspelling recognition
- Pre-trained model weights
- Performance optimization

## License

MIT License © 2024 SignBridge Contributors. See [LICENSE](LICENSE) file.

## Citation

If you use SignBridge in research or projects, please cite:

```bibtex
@software{signbridge2024,
  title={SignBridge: Real-time Sign Language Captioning Framework},
  author={SignBridge Contributors},
  year={2024},
  url={https://github.com/Nightxelf/SignBridge},
  license={MIT}
}
```

## Acknowledgments

- [WLASL Dataset](https://github.com/dxli94/WLASL) - Word-Level American Sign Language
- [MediaPipe](https://mediapipe.dev/) - Hand landmark detection
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Vite](https://vitejs.dev/) - Build tool

## Support & Community

- **Issues**: [GitHub Issues](https://github.com/Nightxelf/SignBridge/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Nightxelf/SignBridge/discussions)
- **Email**: Create an issue and tag with `question` label

---

**Made with ❤️ for accessibility and inclusion.**
