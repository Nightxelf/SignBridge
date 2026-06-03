# Contributing to SignBridge

Thank you for your interest in contributing to SignBridge! This document provides guidelines and instructions for contributing.

## How to Contribute

### Reporting Issues

- **Bugs**: Open an issue with reproduction steps, expected vs actual behavior, and environment details (OS, Python version, etc.)
- **Feature requests**: Describe the feature, use case, and potential implementation approach
- **Documentation**: Report typos, unclear explanations, or missing docs

### Code Contributions

1. **Fork** the repository and create a branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

3. **Make your changes**:
   - Follow PEP 8 style guidelines
   - Add tests for new functionality
   - Update docstrings and README as needed

4. **Format and lint**:
   ```bash
   black backend/ training/ extension/src/
   flake8 backend/ training/
   pytest
   ```

5. **Commit with clear messages**:
   ```bash
   git commit -m "feat: add fingerspelling support"
   git commit -m "fix: handle missing landmarks in preprocessing"
   git commit -m "docs: clarify WLASL dataset setup"
   ```

6. **Push and create a Pull Request** with a description of changes.

## Development Setup

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
uvicorn app:app --reload
```

### Extension Development

```bash
cd extension
npm install
npm run dev    # Dev server at localhost:5173
npm run build  # Production build
```

### Training Pipeline

Test the training pipeline locally:

```bash
# Demo mode (no external data)
python training/run_training.py --demo --epochs 1 --max-items 4

# With WLASL dataset
python training/run_training.py --json training/dataset/metadata/WLASL_v0.3.json --download --epochs 2 --max-items 32
```

## Code Style

- **Python**: PEP 8 via `black` (line length: 100)
- **JavaScript/JSX**: Prettier defaults
- **Docstrings**: Google-style format
- **Type hints**: Encouraged for new code

Example:
```python
def predict_from_landmarks(landmarks_sequence: List[List[float]]) -> Tuple[str, float]:
    """Predict sign language class from normalized landmark sequence.
    
    Args:
        landmarks_sequence: List of frames, each with 63 float values (21 joints × 3 coords).
    
    Returns:
        Tuple of (predicted_class, confidence_score).
    
    Raises:
        ValueError: If landmarks_sequence is empty or malformed.
    """
    # implementation
```

## Testing

Write tests for new features:

```bash
pytest backend/tests/ training/tests/
pytest --cov=backend --cov=training
```

Test checklist:
- [ ] Unit tests for utilities and helpers
- [ ] Integration tests for pipeline (data → train → inference)
- [ ] E2E test for extension (backend + UI interaction)

## Areas for Contribution

### High Priority
- [ ] Transformer-based model (replace LSTM)
- [ ] Multi-hand support
- [ ] GPU optimization
- [ ] Dataset annotation tool

### Medium Priority
- [ ] Fingerspelling recognition (letter-level)
- [ ] Real-time video demo script
- [ ] Comprehensive logging and monitoring
- [ ] CI/CD pipeline (GitHub Actions)

### Lower Priority
- [ ] Mobile extension (Firefox, Safari)
- [ ] Web demo (no extension required)
- [ ] Internationalization (non-ASL sign languages)
- [ ] Performance benchmarking

## Documentation

Help improve docs:
- Add examples to API reference
- Create tutorials for common tasks
- Document known limitations
- Clarify unclear sections

## Review Process

1. **Automated checks**: CI pipeline runs tests, linting, and type checks
2. **Code review**: Maintainers review for correctness, clarity, and fit with project goals
3. **Feedback**: Iterate based on review comments
4. **Merge**: Approved PRs are merged to main

Typical review time: 3-7 days

## License

By contributing, you agree that your code will be licensed under the MIT License (same as the project).

## Questions?

- Open an issue with the `question` label
- Check existing issues and discussions
- See the main README for architecture overview

---

Thank you for making SignBridge better! 🤝
