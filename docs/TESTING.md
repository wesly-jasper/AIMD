# AIMD Testing & Verification Guide

## Test Suite Execution

All unit and integration tests run with `pytest`:

```bash
# Run complete test suite (110 passed)
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run integration pipeline tests
pytest tests/integration/ -v
```

## Test Breakdown

- `test_trufor_runner.py`: PyTorch module inference, ONNX inference, missing file errors.
- `test_trufor_detector.py`: TruFor schema integration and region mapping.
- `test_embedding_provider.py`: OpenCLIP 512-dim embedding generation, L2 normalization, baseline color histogram.
- `test_face_detector.py`: MTCNN deep learning face detector and Haar cascade forensics.
- `test_audio_detector.py`: Baseline acoustic spectral features and NeuralSpeechDetector.
- `test_object_detector.py`: YOLOv8 object detector bounding box mapping and baseline placeholder.
- `test_internet_search_provider.py`: Google Cloud Vision and TinEye connector mocks and UNAVAILABLE handling.
- `test_analysis_pipeline.py`: Full end-to-end multi-stage pipeline execution for images, videos, and audio.
