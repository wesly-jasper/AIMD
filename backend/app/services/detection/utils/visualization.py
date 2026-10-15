"""
Visualization Utilities for Manipulation Localization.

Generates heatmap overlays, region annotations, and comparison visualizations
for forensic analysis reports.
"""
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.schemas.detection import DetectionRegion

logger = logging.getLogger(__name__)


# Color maps for different region types
REGION_COLORS = {
    "copy_move_region": (0, 255, 255),      # Yellow
    "manipulation_mask": (255, 0, 255),     # Magenta
    "face_region": (0, 255, 0),             # Green
    "forensic_grid": (255, 165, 0),         # Orange
    "ela_region": (255, 0, 0),              # Red
    "fft_region": (0, 0, 255),              # Blue
    "noise_region": (128, 0, 128),          # Purple
    "tracked_region": (0, 128, 255),        # Light Blue
    "splice_region": (255, 128, 0),         # Dark Orange
    "deepfake_region": (255, 0, 128),       # Pink
}

REGION_LABELS = {
    "copy_move_region": "Copy-Move",
    "manipulation_mask": "Manipulation",
    "face_region": "Face Region",
    "forensic_grid": "ELA/FFT Grid",
    "ela_region": "ELA Anomaly",
    "fft_region": "FFT Anomaly",
    "noise_region": "Noise Anomaly",
    "tracked_region": "Tracked Region",
    "splice_region": "Splice",
    "deepfake_region": "Deepfake",
}


def generate_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
    threshold: float = 0.0,
) -> np.ndarray:
    """
    Generate heatmap overlay on image.

    Args:
        image: Input image (BGR)
        heatmap: Heatmap values [0, 1]
        alpha: Overlay transparency
        colormap: OpenCV colormap
        threshold: Minimum heatmap value to display

    Returns:
        Image with heatmap overlay
    """
    h, w = image.shape[:2]

    # Resize heatmap if needed
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

    # Apply threshold
    heatmap_thresh = heatmap.copy()
    heatmap_thresh[heatmap_thresh < threshold] = 0

    # Normalize to [0, 255]
    heatmap_norm = (heatmap_thresh * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)

    # Create mask for non-zero heatmap
    mask = heatmap_thresh > 0

    # Blend
    overlay = image.copy()
    overlay[mask] = cv2.addWeighted(
        image[mask], 1 - alpha, heatmap_colored[mask], alpha, 0
    )

    return overlay


def draw_detection_regions(
    image: np.ndarray,
    regions: List[DetectionRegion],
    show_labels: bool = True,
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> np.ndarray:
    """
    Draw detection regions on image.

    Args:
        image: Input image (BGR)
        regions: List of DetectionRegion
        show_labels: Show region type labels
        show_confidence: Show confidence scores
        line_thickness: Bounding box line thickness
        font_scale: Font scale for labels

    Returns:
        Annotated image
    """
    annotated = image.copy()

    for region in regions:
        x, y, w, h = [int(v) for v in region.bbox]
        color = REGION_COLORS.get(region.type, (255, 255, 255))
        label = REGION_LABELS.get(region.type, region.type)

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, line_thickness)

        # Prepare label text
        if show_labels or show_confidence:
            text_parts = []
            if show_labels:
                text_parts.append(label)
            if show_confidence:
                text_parts.append(f"{region.confidence:.2f}")
            text = " ".join(text_parts)

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(
                annotated,
                (x, y - text_h - 4),
                (x + text_w + 4, y),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                text,
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    return annotated


def create_region_comparison(
    original: np.ndarray,
    regions_by_detector: Dict[str, List[DetectionRegion]],
    heatmap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create side-by-side comparison: original vs each detector's regions.

    Args:
        original: Original image
        regions_by_detector: Dict of detector_name -> regions
        heatmap: Optional combined heatmap

    Returns:
        Comparison grid image
    """
    h, w = original.shape[:2]
    n_detectors = len(regions_by_detector)
    n_cols = min(3, n_detectors + (1 if heatmap is not None else 0))
    n_rows = (n_detectors + (1 if heatmap is not None else 0) + n_cols - 1) // n_cols

    cell_w, cell_h = w, h
    grid = np.zeros((n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)

    # First cell: original
    grid[0:cell_h, 0:cell_w] = original

    col = 1
    row = 0

    # Heatmap cell
    if heatmap is not None:
        hm_overlay = generate_heatmap_overlay(original, heatmap)
        grid[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = hm_overlay
        col += 1
        if col >= n_cols:
            col = 0
            row += 1

    # Detector cells
    for det_name, regions in regions_by_detector.items():
        det_img = draw_detection_regions(original, regions)
        # Add detector name
        cv2.putText(
            det_img,
            det_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        grid[row*cell_h:(row+1)*cell_h, col*cell_w:(col+1)*cell_w] = det_img
        col += 1
        if col >= n_cols:
            col = 0
            row += 1

    return grid


def generate_forensic_report_image(
    image_path: str,
    regions: List[DetectionRegion],
    heatmap: Optional[np.ndarray] = None,
    metadata: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Generate a comprehensive forensic visualization image.

    Args:
        image_path: Path to original image
        regions: All detection regions
        heatmap: Optional combined heatmap
        metadata: Optional metadata dict
        output_path: Optional path to save

    Returns:
        Report image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    h, w = img.shape[:2]

    # Create canvas with space for legend and metadata
    legend_h = 120
    meta_h = 150 if metadata else 0
    canvas = np.ones((h + legend_h + meta_h, w, 3), dtype=np.uint8) * 240
    canvas[:h, :w] = img

    # Draw all regions
    annotated = draw_detection_regions(canvas[:h, :w], regions)
    canvas[:h, :w] = annotated

    # Draw legend
    legend_y = h + 10
    x_pos = 10
    for region_type, color in REGION_COLORS.items():
        # Check if this type exists in regions
        if any(r.type == region_type for r in regions):
            cv2.rectangle(canvas, (x_pos, legend_y), (x_pos + 20, legend_y + 20), color, -1)
            cv2.rectangle(canvas, (x_pos, legend_y), (x_pos + 20, legend_y + 20), (0, 0, 0), 1)
            label = REGION_LABELS.get(region_type, region_type)
            cv2.putText(
                canvas, label, (x_pos + 25, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
            )
            x_pos += 150
            if x_pos > w - 150:
                x_pos = 10
                legend_y += 30

    # Draw metadata
    if metadata:
        meta_y = h + legend_h + 10
        cv2.putText(canvas, "Analysis Metadata:", (10, meta_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y = meta_y + 20
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                value = str(value)[:80]
            text = f"{key}: {value}"
            cv2.putText(canvas, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
            y += 18

    if output_path:
        cv2.imwrite(output_path, canvas)

    return canvas


def create_video_timeline_visualization(
    frame_regions: Dict[int, List[DetectionRegion]],
    video_info: Dict[str, Any],
    output_path: Optional[str] = None,
    width: int = 1200,
    height: int = 400,
) -> np.ndarray:
    """
    Create timeline visualization of detections across video frames.

    Args:
        frame_regions: Dict of frame_idx -> regions
        video_info: Video metadata (fps, duration, frame_count)
        output_path: Optional save path
        width: Output width
        height: Output height

    Returns:
        Timeline visualization
    """
    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    fps = video_info.get("fps", 30)
    frame_count = video_info.get("frame_count", 0)
    duration = frame_count / fps if fps > 0 else 0

    # Time axis
    margin = 50
    timeline_y = height - margin
    timeline_h = 60

    # Draw time axis
    cv2.line(canvas, (margin, timeline_y), (width - margin, timeline_y), (0, 0, 0), 2)

    # Time markers
    for t in range(0, int(duration) + 1, max(1, int(duration) // 10)):
        x = margin + int((t / duration) * (width - 2 * margin)) if duration > 0 else margin
        cv2.line(canvas, (x, timeline_y - 5), (x, timeline_y + 5), (0, 0, 0), 1)
        cv2.putText(canvas, f"{t}s", (x - 10, timeline_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Plot detections by type
    region_types = set()
    for regions in frame_regions.values():
        for r in regions:
            region_types.add(r.type)

    type_colors = {rt: REGION_COLORS.get(rt, (128, 128, 128)) for rt in region_types}
    type_y = {rt: margin + i * 30 for i, rt in enumerate(sorted(region_types))}

    for frame_idx, regions in frame_regions.items():
        if frame_count == 0:
            continue
        x = margin + int((frame_idx / frame_count) * (width - 2 * margin))

        for region in regions:
            y = type_y.get(region.type, margin)
            conf = region.confidence
            # Dot size based on confidence
            radius = max(2, int(conf * 8))
            color = type_colors.get(region.type, (128, 128, 128))
            cv2.circle(canvas, (x, y), radius, color, -1)

    # Legend
    for i, (rt, color) in enumerate(type_colors.items()):
        y = margin + i * 30 - 5
        cv2.circle(canvas, (width - margin + 10, y), 6, color, -1)
        cv2.putText(canvas, REGION_LABELS.get(rt, rt), (width - margin + 20, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    if output_path:
        cv2.imwrite(output_path, canvas)

    return canvas


def blend_heatmaps(heatmaps: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """Blend multiple heatmaps with optional weights."""
    if not heatmaps:
        return np.array([])

    if weights is None:
        weights = [1.0] * len(heatmaps)

    # Resize all to same size
    target_shape = heatmaps[0].shape[:2]
    resized = []
    for hm in heatmaps:
        if hm.shape[:2] != target_shape:
            hm = cv2.resize(hm, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        resized.append(hm)

    # Weighted blend
    blended = np.zeros_like(resized[0], dtype=np.float32)
    for hm, w in zip(resized, weights):
        blended += hm * w
    blended /= sum(weights)

    return blended


def export_regions_geojson(regions: List[DetectionRegion], image_size: Tuple[int, int]) -> Dict:
    """Export regions as GeoJSON for web visualization."""
    features = []
    for i, region in enumerate(regions):
        x, y, w, h = region.bbox
        # Convert to polygon (rectangle)
        coords = [[
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
            [x, y]
        ]]

        features.append({
            "type": "Feature",
            "id": i,
            "properties": {
                "type": region.type,
                "confidence": region.confidence,
                "label": REGION_LABELS.get(region.type, region.type),
                **region.extra
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": coords
            }
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "bbox": [0, 0, image_size[0], image_size[1]]
    }