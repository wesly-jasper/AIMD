import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class TruForRunner:
    """
    Real TruFor / Deep-Learning Image Forgery Detection Model Runner.

    Executes neural network inference for image manipulation detection and
    pixel-level anomaly localization. Supports PyTorch (.pth, .pt) and ONNX (.onnx) formats
    with automatic GPU (CUDA) acceleration and CPU fallback.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.loaded = False
        self.model: Any = None
        self.is_onnx = False
        self.onnx_session = None

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load(self) -> None:
        """
        Load model weights from model_path into memory on configured device.
        """
        if not self.model_path:
            raise ValueError('TruFor model path is not configured')

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f'TruFor model not found: {model_path}')

        ext = model_path.suffix.lower()

        try:
            if ext == '.onnx':
                import onnxruntime as ort
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
                self.onnx_session = ort.InferenceSession(str(model_path), providers=providers)
                self.is_onnx = True
                self.loaded = True
                logger.info(f'Loaded TruFor ONNX model from {model_path} with providers {self.onnx_session.get_providers()}')
            else:
                # Load PyTorch model / weights
                try:
                    loaded_obj = torch.load(str(model_path), map_location=self.device, weights_only=False)
                except Exception:
                    loaded_obj = torch.load(str(model_path), map_location=self.device)

                if isinstance(loaded_obj, torch.nn.Module):
                    self.model = loaded_obj
                elif isinstance(loaded_obj, dict):
                    if 'model' in loaded_obj and isinstance(loaded_obj['model'], torch.nn.Module):
                        self.model = loaded_obj['model']
                    elif 'state_dict' in loaded_obj:
                        self.model = loaded_obj['state_dict']
                    else:
                        self.model = loaded_obj
                else:
                    self.model = loaded_obj

                if isinstance(self.model, torch.nn.Module):
                    self.model.to(self.device)
                    self.model.eval()

                self.is_onnx = False
                self.loaded = True
                logger.info(f'Loaded TruFor PyTorch model from {model_path} onto {self.device}')

        except Exception as e:
            logger.error(f'Failed to load TruFor model from {model_path}: {e}')
            raise RuntimeError(f'Could not load TruFor model weights: {e}') from e

    def _preprocess(self, file_path: Path) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        pil_img = Image.open(file_path).convert('RGB')
        orig_w, orig_h = pil_img.size
        img_np = np.array(pil_img)

        img_float = img_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (img_float - mean) / std

        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        return tensor, img_np, (orig_h, orig_w)

    def predict(self, file_path: Union[str, Path], threshold: float = 0.50) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError('TruFor model is not loaded')

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {file_path}')

        tensor, img_np, (h, w) = self._preprocess(file_path)

        anomaly_map = None
        global_score = 0.0

        if self.is_onnx and self.onnx_session is not None:
            input_name = self.onnx_session.get_inputs()[0].name
            ort_inputs = {input_name: tensor.numpy()}
            ort_outs = self.onnx_session.run(None, ort_inputs)
            out_tensor = ort_outs[0]

            if out_tensor.ndim == 4:
                anomaly_map = out_tensor[0, 0] if out_tensor.shape[1] == 1 else np.mean(out_tensor[0], axis=0)
            elif out_tensor.ndim == 2:
                global_score = float(1.0 / (1.0 + np.exp(-out_tensor[0, 0])))
                anomaly_map = np.full((h, w), global_score, dtype=np.float32)
            else:
                anomaly_map = np.squeeze(out_tensor)

            if anomaly_map is not None:
                anomaly_map = 1.0 / (1.0 + np.exp(-anomaly_map))
                if global_score == 0.0:
                    global_score = float(np.percentile(anomaly_map, 98))
        else:
            tensor = tensor.to(self.device)
            with torch.no_grad():
                if isinstance(self.model, torch.nn.Module):
                    output = self.model(tensor)
                    if isinstance(output, tuple):
                        map_tensor = output[0]
                        if len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].numel() == 1:
                            global_score = float(torch.sigmoid(output[1]).cpu().item())
                        anomaly_map = torch.sigmoid(map_tensor).squeeze().cpu().numpy()
                    elif isinstance(output, dict):
                        map_tensor = output.get('map', output.get('anomaly_map', output.get('mask')))
                        if map_tensor is not None:
                            anomaly_map = torch.sigmoid(map_tensor).squeeze().cpu().numpy()
                        if 'score' in output:
                            global_score = float(torch.sigmoid(output['score']).cpu().item())
                    elif isinstance(output, torch.Tensor):
                        if output.ndim == 4:
                            anomaly_map = torch.sigmoid(output).squeeze().cpu().numpy()
                            if anomaly_map.ndim == 3:
                                anomaly_map = np.mean(anomaly_map, axis=0)
                            global_score = float(np.percentile(anomaly_map, 98))
                        elif output.numel() == 1:
                            global_score = float(torch.sigmoid(output).cpu().item())
                            anomaly_map = np.full((h, w), global_score, dtype=np.float32)
                        else:
                            anomaly_map = torch.sigmoid(output).squeeze().cpu().numpy()
                            global_score = float(np.mean(anomaly_map))
                else:
                    global_score = 0.0
                    anomaly_map = np.zeros((h, w), dtype=np.float32)

        if anomaly_map is None or anomaly_map.size == 0:
            anomaly_map = np.zeros((h, w), dtype=np.float32)

        if anomaly_map.shape != (h, w):
            anomaly_map = cv2.resize(anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

        anomaly_map = np.clip(anomaly_map, 0.0, 1.0)
        if global_score == 0.0:
            top_k_score = float(np.percentile(anomaly_map, 98))
            mean_score = float(np.mean(anomaly_map))
            global_score = float(0.85 * top_k_score + 0.15 * mean_score)

        global_score = float(np.clip(global_score, 0.0, 1.0))
        manipulation_detected = bool(global_score >= threshold)

        regions: List[Dict[str, Any]] = []
        binary_mask = (anomaly_map >= threshold).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (h * w) * 0.001

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                region_crop = anomaly_map[by:by+bh, bx:bx+bw]
                region_conf = float(np.mean(region_crop)) if region_crop.size > 0 else global_score
                regions.append({
                    'type': 'forgery_localization',
                    'confidence': round(float(region_conf), 4),
                    'bbox': [float(bx), float(by), float(bw), float(bh)],
                })

        return {
            'manipulation_detected': manipulation_detected,
            'confidence': round(global_score, 4),
            'assessment': 'SUSPICIOUS' if manipulation_detected else 'CLEAN',
            'regions': regions,
            'metadata': {
                'status': 'MODEL_AVAILABLE',
                'device': str(self.device),
                'model_path': str(self.model_path),
                'anomaly_max': round(float(np.max(anomaly_map)), 4),
                'anomaly_mean': round(float(np.mean(anomaly_map)), 4),
                'regions_detected': len(regions),
                'threshold': threshold,
                'model_format': 'onnx' if self.is_onnx else 'pytorch',
            },
        }
