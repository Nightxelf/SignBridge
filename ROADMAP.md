# SignBridge Roadmap

This document outlines the planned development for SignBridge.

## Current Version (v0.1 Beta)

✅ **Completed:**
- FastAPI backend with hand landmark extraction
- Chrome extension (Manifest V3) for video call integration
- Complete training pipeline (dataset prep, LSTM training, inference)
- Docker deployment support
- Documentation and contribution guidelines

## Short Term (v0.2 - Q3 2024)

- [ ] Pre-trained models on WLASL (100-500 words)
- [ ] Improved hand detection with fallback modes
- [ ] Real-time video demo script
- [ ] Benchmark script for hardware-specific performance
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Unit test coverage >80%

## Medium Term (v0.3-0.4 - Q4 2024)

- [ ] **Transformer-based models** (replace LSTM for better accuracy)
- [ ] **Multi-hand support** (left + right hand simultaneous recognition)
- [ ] **Fingerspelling recognition** (letter-level classification)
- [ ] Improved MediaPipe hand detection with confidence filtering
- [ ] Dataset annotation tool (web UI for manual corrections)
- [ ] Support for additional sign languages (BSL, LSF, DGS)

## Long Term (v1.0 - 2025)

- [ ] **Firefox and Safari extension** support
- [ ] **Web demo** (no extension required, browser-based)
- [ ] Mobile app (React Native or Flutter)
- [ ] Real-time pose + hand landmark combination
- [ ] Contextual sentence recognition (not just isolated signs)
- [ ] GPU optimization with ONNX export
- [ ] Pre-trained models published on Hugging Face

## Research Directions

- [ ] Zero-shot learning for unseen sign language
- [ ] Cross-language transfer learning
- [ ] Continuous sign language recognition (vs. isolated signs)
- [ ] Facial expression and mouthing integration
- [ ] End-to-end video transformer models

## Community Contributions Needed

We actively seek contributions in these areas:

1. **Model Development**
   - Transformer implementations
   - Alternative architectures (TCN, 3D CNN)
   - Knowledge distillation for edge devices

2. **Data & Annotation**
   - WLASL dataset improvements
   - Additional sign language datasets
   - Annotation tools

3. **Extension & Integration**
   - Firefox/Safari support
   - Better video call platform integration
   - Accessibility features

4. **Performance & Deployment**
   - GPU optimization
   - ONNX/TensorRT export
   - Edge device support

5. **Documentation & Examples**
   - Tutorial videos
   - Deployment guides
   - Custom dataset how-tos

## Version Timeline

- **v0.1** (Current): MVP - basic functionality, single-hand, LSTM
- **v0.2**: Polish, pre-trained models, CI/CD
- **v0.3**: Multi-hand, Transformer, advanced features
- **v0.4**: Fingerspelling, improved accuracy
- **v1.0**: Stable release, multi-platform, production-ready

## Priorities

### 🔴 Critical
- Model accuracy on diverse sign languages
- Robustness (handle edge cases, poor lighting)
- Reproducibility and documentation

### 🟡 Important
- Extension compatibility (Firefox, Safari)
- Performance optimization
- User experience improvements

### 🟢 Nice to Have
- Advanced architectures
- Multiple output formats
- Analytics and monitoring

## Feedback & Discussion

We welcome community input on this roadmap. Please:
- Open GitHub issues to propose features
- Start discussions for design decisions
- Submit PRs to accelerate development

---

Last updated: June 2024
