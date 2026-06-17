"""
Real-Time Eye-Blink to Morse Code System
=========================================
MediaPipe FaceMesh + YOLOv26-cls + Streamlit

This system converts eye blinks into Morse code and decoded text using a webcam.
It combines MediaPipe FaceMesh for geometric eye analysis and YOLOv26-cls for
deep learning-based eye state classification with hybrid confidence scoring.

Date: January 2026
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MpImage
from transformers import AutoTokenizer, EncoderDecoderModel
from ultralytics import YOLO
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
from enum import Enum
import time
import threading
from abc import ABC, abstractmethod
import urllib.request
import os


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# MediaPipe FaceMesh eye landmark indices
# Left eye landmarks (from user's perspective, right side of image)
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
# Right eye landmarks (from user's perspective, left side of image)
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]

# Extended eye region for cropping (includes eyebrow area)
LEFT_EYE_REGION = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_REGION = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Morse code dictionary
MORSE_CODE_DICT = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9',
    '-----': '0', '.-.-.-': '.', '--..--': ',', '..--..': '?',
    '.----.': "'", '-.-.--': '!', '-..-.': '/', '-.--.': '(',
    '-.--.-': ')', '.-...': '&', '---...': ':', '-.-.-.': ';',
    '-...-': '=', '.-.-.': '+', '-....-': '-', '..--.-': '_',
    '.-..-.': '"', '...-..-': '$', '.--.-.': '@', '...---...': 'SOS'
}

# Bump this when pipeline behavior changes to rebuild session system safely.
SYSTEM_LOGIC_VERSION = "2026-04-10-nlp-raw-v2"


# =============================================================================
# DATA CLASSES & ENUMS
# =============================================================================

class EyeState(Enum):
    """Enumeration of possible eye states."""
    OPEN = "open"
    CLOSED = "closed"
    UNKNOWN = "unknown"


class BlinkType(Enum):
    """Enumeration of blink types for Morse code."""
    DOT = "."
    DASH = "-"
    NONE = ""


class CalibrationMethod(Enum):
    """Available calibration methods."""
    PREDEFINED_WORD = "predefined_word"
    SINGLE_LETTER = "single_letter"
    FREE_BLINK = "free_blink"


@dataclass
class EyeData:
    """Container for eye-related data from a single frame."""
    left_ear: float = 0.0
    right_ear: float = 0.0
    avg_ear: float = 0.0
    normalized_ear: float = 0.0
    left_crop: Optional[np.ndarray] = None
    right_crop: Optional[np.ndarray] = None
    landmarks_detected: bool = False


@dataclass
class YOLOResult:
    """Container for YOLO classification results."""
    state: EyeState = EyeState.UNKNOWN
    confidence: float = 0.0
    open_prob: float = 0.0
    closed_prob: float = 0.0


@dataclass
class BlinkEvent:
    """Container for a detected blink event."""
    start_frame: int = 0
    end_frame: int = 0
    duration_frames: int = 0
    duration_ms: float = 0.0
    blink_type: BlinkType = BlinkType.NONE
    confidence: float = 0.0


@dataclass
class SystemConfig:
    """System configuration parameters."""
    # Confidence fusion
    alpha: float = 0.4  # Weight for YOLO confidence (1-alpha for EAR)
    
    # Blink detection
    blink_threshold: float = 0.5  # Confidence threshold for blink detection
    
    # Timing (in seconds)
    letter_gap_seconds: float = 1.5  # Pause duration for letter gap
    word_gap_seconds: float = 3.0    # Pause duration for word gap
    sentence_gap_seconds: float = 5.0  # Pause duration for sentence gap
    
    # EAR normalization
    ear_min: float = 0.15  # Minimum EAR (closed eyes)
    ear_max: float = 0.35  # Maximum EAR (open eyes)
    
    # Smoothing
    smoothing_window: int = 5  # Frames for rolling average
    ema_alpha: float = 0.3    # EMA smoothing factor
    
    # Calibration
    calibration_blinks: int = 5  # Number of blinks for calibration
    default_blink_duration_ms: float = 200.0  # Default short blink duration
    
    # Model paths
    yolo_model_path: str = "runs/classify/nano_100/weights/best.pt"
    
    # Performance
    target_fps: int = 15
    use_gpu: bool = True


class CalibrationPhase(Enum):
    """Calibration phases."""
    NOT_STARTED = "not_started"
    COLLECTING_DOTS = "collecting_dots"
    COLLECTING_DASHES = "collecting_dashes"
    COMPLETED = "completed"


class AppCalibrationStage(Enum):
    """Mandatory application-level calibration stages."""
    NOT_STARTED = "not_started"
    EAR_OPEN = "ear_open"
    EAR_CLOSED = "ear_closed"
    BLINK_DOT_DASH = "blink_dot_dash"
    COMPLETED = "completed"


@dataclass
class CalibrationData:
    """Container for calibration results."""
    is_calibrated: bool = False
    avg_blink_duration_ms: float = 200.0  # Threshold between dot and dash
    avg_dot_duration_ms: float = 150.0
    avg_dash_duration_ms: float = 400.0
    dot_durations: List[float] = field(default_factory=list)
    dash_durations: List[float] = field(default_factory=list)
    ear_baseline_open: float = 0.3
    ear_baseline_closed: float = 0.15
    calibration_method: CalibrationMethod = CalibrationMethod.FREE_BLINK


# =============================================================================
# EYE ANALYSIS MODULE
# =============================================================================

# Download FaceLandmarker model if not exists
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def download_face_landmarker_model():
    """Download the FaceLandmarker model if it doesn't exist."""
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        print(f"Downloading FaceLandmarker model...")
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL_PATH)
        print(f"Model downloaded to {FACE_LANDMARKER_MODEL_PATH}")

class EyeAnalyzer:
    """
    Handles eye landmark detection and geometric analysis using MediaPipe FaceLandmarker (Tasks API).
    Computes Eye Aspect Ratio (EAR) and crops eye regions for YOLO inference.
    """
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the EyeAnalyzer with MediaPipe FaceLandmarker.
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        # Download model if needed
        download_face_landmarker_model()
        
        # Create FaceLandmarker options
        base_options = mp_tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        
    def compute_ear(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """
        Compute Eye Aspect Ratio (EAR) from landmarks.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Where p1-p6 are the 6 eye landmarks in order:
        p1: outer corner, p2: upper outer, p3: upper inner,
        p4: inner corner, p5: lower inner, p6: lower outer
        
        Args:
            landmarks: Array of facial landmarks
            eye_indices: Indices of the 6 eye landmarks
            
        Returns:
            Eye Aspect Ratio value
        """
        try:
            # Extract eye points
            eye_points = landmarks[eye_indices]
            
            # Compute vertical distances
            v1 = np.linalg.norm(eye_points[1] - eye_points[5])  # p2-p6
            v2 = np.linalg.norm(eye_points[2] - eye_points[4])  # p3-p5
            
            # Compute horizontal distance
            h = np.linalg.norm(eye_points[0] - eye_points[3])   # p1-p4
            
            # Avoid division by zero
            if h < 1e-6:
                return 0.0
                
            ear = (v1 + v2) / (2.0 * h)
            return ear
            
        except Exception as e:
            return 0.0
    
    def normalize_ear(self, ear: float, ear_min: float = 0.15, 
                      ear_max: float = 0.35) -> float:
        """
        Normalize EAR value to [0, 1] range.
        
        Args:
            ear: Raw EAR value
            ear_min: Minimum expected EAR (closed eyes)
            ear_max: Maximum expected EAR (open eyes)
            
        Returns:
            Normalized EAR in [0, 1] range (0=closed, 1=open)
        """
        normalized = (ear - ear_min) / (ear_max - ear_min + 1e-6)
        return np.clip(normalized, 0.0, 1.0)
    
    def crop_eye_region(self, frame: np.ndarray, landmarks: np.ndarray,
                        eye_region_indices: List[int], padding: float = 0.3) -> Optional[np.ndarray]:
        """
        Crop eye region from frame based on landmark coordinates.
        
        Args:
            frame: Input frame (BGR)
            landmarks: Facial landmarks as pixel coordinates
            eye_region_indices: Indices of landmarks defining eye region
            padding: Padding ratio around the eye region
            
        Returns:
            Cropped eye region or None if cropping fails
        """
        try:
            h, w = frame.shape[:2]
            
            # Get eye region points
            eye_points = landmarks[eye_region_indices]
            
            # Compute bounding box
            x_min = int(np.min(eye_points[:, 0]))
            x_max = int(np.max(eye_points[:, 0]))
            y_min = int(np.min(eye_points[:, 1]))
            y_max = int(np.max(eye_points[:, 1]))
            
            # Add padding
            pad_x = int((x_max - x_min) * padding)
            pad_y = int((y_max - y_min) * padding)
            
            x_min = max(0, x_min - pad_x)
            x_max = min(w, x_max + pad_x)
            y_min = max(0, y_min - pad_y)
            y_max = min(h, y_max + pad_y)
            
            # Crop
            crop = frame[y_min:y_max, x_min:x_max]
            
            if crop.size == 0:
                return None
                
            return crop
            
        except Exception as e:
            return None
    
    def process_frame(self, frame: np.ndarray, config: SystemConfig) -> Tuple[EyeData, np.ndarray]:
        """
        Process a frame to extract eye data and annotated frame.
        
        Args:
            frame: Input frame (BGR)
            config: System configuration
            
        Returns:
            Tuple of (EyeData, annotated_frame)
        """
        eye_data = EyeData()
        annotated_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = MpImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect face landmarks
        results = self.face_landmarker.detect(mp_image)
        
        if not results.face_landmarks or len(results.face_landmarks) == 0:
            return eye_data, annotated_frame
        
        face_landmarks = results.face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Convert landmarks to numpy array (pixel coordinates)
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w]
            for lm in face_landmarks
        ])
        
        eye_data.landmarks_detected = True
        
        # Compute EAR for both eyes
        eye_data.left_ear = self.compute_ear(landmarks, LEFT_EYE_LANDMARKS)
        eye_data.right_ear = self.compute_ear(landmarks, RIGHT_EYE_LANDMARKS)
        eye_data.avg_ear = (eye_data.left_ear + eye_data.right_ear) / 2.0
        
        # Normalize EAR
        eye_data.normalized_ear = self.normalize_ear(
            eye_data.avg_ear, config.ear_min, config.ear_max
        )
        
        # Crop eye regions
        eye_data.left_crop = self.crop_eye_region(
            frame, landmarks, LEFT_EYE_REGION
        )
        eye_data.right_crop = self.crop_eye_region(
            frame, landmarks, RIGHT_EYE_REGION
        )
        
        # Draw landmarks on annotated frame
        self._draw_eye_landmarks(annotated_frame, landmarks)
        
        return eye_data, annotated_frame
    
    def _draw_eye_landmarks(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw eye landmarks on the frame."""
        # Draw left eye
        for idx in LEFT_EYE_LANDMARKS:
            pt = tuple(landmarks[idx][:2].astype(int))
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)
        
        # Draw right eye
        for idx in RIGHT_EYE_LANDMARKS:
            pt = tuple(landmarks[idx][:2].astype(int))
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)
        
        # Connect eye landmarks
        self._draw_eye_contour(frame, landmarks, LEFT_EYE_LANDMARKS, (0, 255, 0))
        self._draw_eye_contour(frame, landmarks, RIGHT_EYE_LANDMARKS, (0, 255, 0))
    
    def _draw_eye_contour(self, frame: np.ndarray, landmarks: np.ndarray,
                          indices: List[int], color: Tuple[int, int, int]):
        """Draw eye contour connecting landmarks."""
        points = landmarks[indices][:, :2].astype(int)
        for i in range(len(points)):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1) % len(points)])
            cv2.line(frame, pt1, pt2, color, 1)
    
    def close(self):
        """Release resources."""
        if self.face_landmarker:
            self.face_landmarker.close()


# =============================================================================
# YOLO CLASSIFIER MODULE
# =============================================================================

def preprocess_for_yolo(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for YOLO inference to match training preprocessing.
    
    Training preprocessing steps:
    1. Auto-orient (handled by camera/cv2)
    2. Center-crop to portrait 9:16 aspect ratio
    3. Resize: stretch to 512×512 (no letterbox, no aspect ratio preservation)
    4. Color: RGB
    5. Normalization: handled internally by YOLO
    
    Args:
        image: Input image (BGR format from OpenCV)
        
    Returns:
        Preprocessed image (RGB, 512x512)
    """
    if image is None or image.size == 0:
        return image

    # Step 1: Center-crop to portrait 9:16 (w:h) before square stretching.
    # This only affects YOLO input preprocessing and not the displayed frontend frame.
    h, w = image.shape[:2]
    target_ratio = 9.0 / 16.0  # width / height
    current_ratio = w / (h + 1e-6)

    if current_ratio > target_ratio:
        # Image too wide: keep height, crop width.
        new_w = max(1, int(h * target_ratio))
        x0 = max(0, (w - new_w) // 2)
        image = image[:, x0:x0 + new_w]
    else:
        # Image too tall/narrow: keep width, crop height.
        new_h = max(1, int(w / target_ratio))
        y0 = max(0, (h - new_h) // 2)
        image = image[y0:y0 + new_h, :]
    
    # Step 2: Convert BGR to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 3: Resize to 512x512 using stretch (no aspect ratio preservation)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    return img


class YOLOEyeClassifier:
    """
    Deep learning eye state classifier using YOLOv26-cls.
    Classifies eye crops as 'open' or 'closed'.
    """
    
    def __init__(self, model_path: str, use_gpu: bool = True):
        """
        Initialize the YOLO classifier.
        
        Args:
            model_path: Path to the YOLOv26-cls model weights
            use_gpu: Whether to use GPU for inference
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self.class_names = ['closed', 'open']  # Adjust based on your training
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model and run a dummy warmup to pre-compile the inference graph."""
        try:
            self.model = YOLO(self.model_path)
            if self.use_gpu:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                else:
                    print("GPU not available, falling back to CPU")
                    self.use_gpu = False
            self._warmup()
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.model = None

    def _warmup(self):
        """Run one dummy inference so PyTorch JIT-compiles the graph before first real frame."""
        if self.model is None:
            return
        try:
            dummy = np.zeros((512, 512, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
        except Exception:
            pass
    
    def classify(self, image: np.ndarray) -> YOLOResult:
        """
        Classify an eye image as open or closed.
        
        Args:
            image: Eye crop image (BGR)
            
        Returns:
            YOLOResult with classification results
        """
        result = YOLOResult()
        
        if self.model is None or image is None or image.size == 0:
            return result
        
        try:
            # Apply training-matched preprocessing:
            # BGR→RGB, resize 512x512 stretch, histogram equalization on luminance
            preprocessed = preprocess_for_yolo(image)
            
            # Run inference (YOLO handles normalization internally)
            predictions = self.model(preprocessed, verbose=False)
            
            if predictions and len(predictions) > 0:
                probs = predictions[0].probs
                
                if probs is not None:
                    # Get class probabilities
                    class_probs = probs.data.cpu().numpy()
                    
                    # Map to open/closed (adjust indices based on your model)
                    # Assuming index 0 = closed, index 1 = open
                    result.closed_prob = float(class_probs[0])
                    result.open_prob = float(class_probs[1])
                    
                    # Determine state
                    if result.open_prob > result.closed_prob:
                        result.state = EyeState.OPEN
                        result.confidence = result.open_prob
                    else:
                        result.state = EyeState.CLOSED
                        result.confidence = result.closed_prob
            
        except Exception as e:
            print(f"YOLO inference error: {e}")
        
        return result
    
    def classify_dual_eye(self, left_crop: Optional[np.ndarray],
                          right_crop: Optional[np.ndarray]) -> YOLOResult:
        """
        Classify using both eye crops with aggregation.
        
        Args:
            left_crop: Left eye crop
            right_crop: Right eye crop
            
        Returns:
            Aggregated YOLOResult
        """
        results = []
        
        if left_crop is not None and left_crop.size > 0:
            results.append(self.classify(left_crop))
        
        if right_crop is not None and right_crop.size > 0:
            results.append(self.classify(right_crop))
        
        if not results:
            return YOLOResult()
        
        # Aggregate results (average probabilities)
        avg_result = YOLOResult()
        avg_result.open_prob = np.mean([r.open_prob for r in results])
        avg_result.closed_prob = np.mean([r.closed_prob for r in results])
        
        if avg_result.open_prob > avg_result.closed_prob:
            avg_result.state = EyeState.OPEN
            avg_result.confidence = avg_result.open_prob
        else:
            avg_result.state = EyeState.CLOSED
            avg_result.confidence = avg_result.closed_prob
        
        return avg_result


# =============================================================================
# CONFIDENCE FUSION MODULE
# =============================================================================

class ConfidenceFusion:
    """
    Fuses YOLO confidence and normalized EAR into a final confidence score.
    
    The fusion formula:
        final_confidence = α * yolo_confidence + (1 - α) * normalized_EAR
    
    Where:
        - yolo_confidence: Probability of eyes being OPEN from YOLO
        - normalized_EAR: Normalized Eye Aspect Ratio (0=closed, 1=open)
        - α (alpha): Configurable fusion weight
    """
    
    def __init__(self, smoothing_window: int = 5, ema_alpha: float = 0.3):
        """
        Initialize the confidence fusion module.
        
        Args:
            smoothing_window: Window size for rolling average
            ema_alpha: Alpha for exponential moving average
        """
        self.smoothing_window = smoothing_window
        self.ema_alpha = ema_alpha
        self.confidence_history = deque(maxlen=smoothing_window)
        self.ema_value = None
    
    def fuse(self, yolo_result: YOLOResult, normalized_ear: float,
             alpha: float) -> float:
        """
        Fuse YOLO and EAR confidence scores.
        
        Args:
            yolo_result: YOLO classification result
            normalized_ear: Normalized EAR value [0, 1]
            alpha: Fusion weight for YOLO confidence
            
        Returns:
            Fused confidence score [0, 1] (higher = more likely open)
        """
        # Use YOLO's open probability as confidence
        yolo_conf = yolo_result.open_prob if yolo_result.open_prob > 0 else 0.5
        
        # Fuse confidence scores
        fused = alpha * yolo_conf + (1 - alpha) * normalized_ear
        
        return np.clip(fused, 0.0, 1.0)
    
    def smooth_rolling(self, confidence: float) -> float:
        """
        Apply rolling average smoothing.
        
        Args:
            confidence: Raw confidence value
            
        Returns:
            Smoothed confidence value
        """
        self.confidence_history.append(confidence)
        return np.mean(self.confidence_history)
    
    def smooth_ema(self, confidence: float) -> float:
        """
        Apply exponential moving average smoothing.
        
        Args:
            confidence: Raw confidence value
            
        Returns:
            Smoothed confidence value
        """
        if self.ema_value is None:
            self.ema_value = confidence
        else:
            self.ema_value = self.ema_alpha * confidence + (1 - self.ema_alpha) * self.ema_value
        
        return self.ema_value
    
    def reset(self):
        """Reset smoothing state."""
        self.confidence_history.clear()
        self.ema_value = None


# =============================================================================
# BLINK DETECTION MODULE
# =============================================================================

class BlinkDetector:
    """
    Detects blink events from confidence scores and classifies them as dots or dashes.
    """
    
    def __init__(self, config: SystemConfig, calibration: CalibrationData):
        """
        Initialize the blink detector.
        
        Args:
            config: System configuration
            calibration: Calibration data
        """
        self.config = config
        self.calibration = calibration
        
        # State tracking
        self.is_blinking = False
        self.blink_start_frame = 0
        self.blink_start_time = 0.0
        self.current_frame = 0
        self.last_blink_end_frame = 0
        
        # FPS estimation
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.estimated_fps = 30.0
    
    def update_fps(self):
        """Update FPS estimation."""
        current_time = time.time()
        delta = current_time - self.last_frame_time
        if delta > 0:
            self.fps_history.append(1.0 / delta)
            self.estimated_fps = np.mean(self.fps_history)
        self.last_frame_time = current_time
    
    def frames_to_ms(self, frames: int) -> float:
        """Convert frame count to milliseconds."""
        if self.estimated_fps > 0:
            return (frames / self.estimated_fps) * 1000
        return frames * 33.33  # Fallback to ~30 FPS
    
    def process(self, confidence: float) -> Optional[BlinkEvent]:
        """
        Process a confidence value and detect blink events.
        
        Args:
            confidence: Current confidence score (0=closed, 1=open)
            
        Returns:
            BlinkEvent if a blink just ended, None otherwise
        """
        self.update_fps()
        self.current_frame += 1
        
        # Detect blink start (confidence drops below threshold)
        if not self.is_blinking and confidence < self.config.blink_threshold:
            self.is_blinking = True
            self.blink_start_frame = self.current_frame
            self.blink_start_time = time.time()
            return None
        
        # Detect blink end (confidence rises above threshold)
        if self.is_blinking and confidence >= self.config.blink_threshold:
            self.is_blinking = False
            
            # Create blink event
            event = BlinkEvent()
            event.start_frame = self.blink_start_frame
            event.end_frame = self.current_frame
            event.duration_frames = event.end_frame - event.start_frame
            event.duration_ms = self.frames_to_ms(event.duration_frames)
            event.confidence = confidence
            
            # Classify blink type based on calibration
            threshold_ms = self.calibration.avg_blink_duration_ms
            if event.duration_ms < threshold_ms:
                event.blink_type = BlinkType.DOT
            else:
                event.blink_type = BlinkType.DASH
            
            self.last_blink_end_frame = self.current_frame
            return event
        
        return None
    
    def get_frames_since_last_blink(self) -> int:
        """Get number of frames since the last blink ended."""
        if self.last_blink_end_frame == 0:
            return 0
        return self.current_frame - self.last_blink_end_frame
    
    def is_letter_gap(self) -> bool:
        """Check if enough time has passed for a letter gap."""
        frames_elapsed = self.get_frames_since_last_blink()
        elapsed_ms = self.frames_to_ms(frames_elapsed)
        return elapsed_ms >= self.config.letter_gap_seconds * 1000.0
    
    def is_word_gap(self) -> bool:
        """Check if enough time has passed for a word gap."""
        frames_elapsed = self.get_frames_since_last_blink()
        elapsed_ms = self.frames_to_ms(frames_elapsed)
        return elapsed_ms >= self.config.word_gap_seconds * 1000.0

    def is_sentence_gap(self) -> bool:
        """Check if enough time has passed for a sentence gap."""
        frames_elapsed = self.get_frames_since_last_blink()
        elapsed_ms = self.frames_to_ms(frames_elapsed)
        return elapsed_ms >= self.config.sentence_gap_seconds * 1000.0
    
    def reset(self):
        """Reset detector state."""
        self.is_blinking = False
        self.blink_start_frame = 0
        self.current_frame = 0
        self.last_blink_end_frame = 0


# =============================================================================
# CALIBRATION MODULE
# =============================================================================

class CalibrationManager:
    """
    Manages calibration process for blink duration threshold.
    User must provide 3 intentional short blinks (dots) and 3 long blinks (dashes).
    The threshold is computed as the midpoint between average dot and dash durations.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the calibration manager.

        Args:
            config: System configuration
        """
        self.config = config
        self.calibration = CalibrationData()

        # Calibration state
        self.is_calibrating = False
        self.calibration_phase = CalibrationPhase.NOT_STARTED
        self.dot_blinks = []  # Store dot durations
        self.dash_blinks = []  # Store dash durations
        self.calibration_start_time = 0.0
        self.target_dots = 3  # Number of dots to collect
        self.target_dashes = 3  # Number of dashes to collect

        # Optional hook fired after _finalize_calibration completes successfully
        # so an external store (e.g. SQLite) can persist the result.
        self.on_calibration_complete: Optional[Callable[[CalibrationData], None]] = None
    
    def start_calibration(self, method: CalibrationMethod = CalibrationMethod.FREE_BLINK,
                          target_blinks: int = 3):
        """
        Start the calibration process.
        First phase: collect dots (short blinks)
        Second phase: collect dashes (long blinks)
        
        Args:
            method: Calibration method to use
            target_blinks: Number of each type to collect (default 3)
        """
        self.is_calibrating = True
        self.calibration_phase = CalibrationPhase.COLLECTING_DOTS
        self.dot_blinks = []
        self.dash_blinks = []
        self.calibration_start_time = time.time()
        self.target_dots = target_blinks
        self.target_dashes = target_blinks
        self.calibration.calibration_method = method
        self.calibration.is_calibrated = False
    
    def add_blink(self, duration_ms: float) -> bool:
        """
        Add a blink duration to calibration data based on current phase.
        
        Args:
            duration_ms: Blink duration in milliseconds
            
        Returns:
            True if calibration is complete
        """
        if not self.is_calibrating:
            return False
        
        # Filter out very short or very long blinks (noise)
        if duration_ms < 30 or duration_ms > 3000:
            return False
        
        if self.calibration_phase == CalibrationPhase.COLLECTING_DOTS:
            # Collecting short blinks (dots)
            self.dot_blinks.append(duration_ms)
            
            # Check if we have enough dots
            if len(self.dot_blinks) >= self.target_dots:
                # Move to dash collection phase
                self.calibration_phase = CalibrationPhase.COLLECTING_DASHES
            return False
            
        elif self.calibration_phase == CalibrationPhase.COLLECTING_DASHES:
            # Collecting long blinks (dashes)
            self.dash_blinks.append(duration_ms)
            
            # Check if we have enough dashes
            if len(self.dash_blinks) >= self.target_dashes:
                self._finalize_calibration()
                return True
        
        return False
    
    def _finalize_calibration(self):
        """Finalize calibration and compute thresholds."""
        self.is_calibrating = False
        self.calibration_phase = CalibrationPhase.COMPLETED

        if len(self.dot_blinks) > 0 and len(self.dash_blinks) > 0:
            # Compute average durations for dots and dashes
            avg_dot = np.mean(self.dot_blinks)
            avg_dash = np.mean(self.dash_blinks)

            # Store averages
            self.calibration.avg_dot_duration_ms = avg_dot
            self.calibration.avg_dash_duration_ms = avg_dash

            # Threshold is the midpoint between average dot and dash
            self.calibration.avg_blink_duration_ms = (avg_dot + avg_dash) / 2.0

            # Store the collected durations
            self.calibration.dot_durations = self.dot_blinks.copy()
            self.calibration.dash_durations = self.dash_blinks.copy()
            self.calibration.is_calibrated = True
        else:
            # Fallback to default
            self.calibration.avg_blink_duration_ms = self.config.default_blink_duration_ms

        # Notify external persistence hook (e.g. SQLite save in server.py).
        # Exceptions in the callback must never break calibration.
        if self.on_calibration_complete is not None and self.calibration.is_calibrated:
            try:
                self.on_calibration_complete(self.calibration)
            except Exception as e:
                print(f"on_calibration_complete callback error: {e}")
    
    def reset(self):
        """Reset calibration."""
        self.is_calibrating = False
        self.calibration_phase = CalibrationPhase.NOT_STARTED
        self.dot_blinks = []
        self.dash_blinks = []
        self.calibration = CalibrationData()
    
    def get_calibration(self) -> CalibrationData:
        """Get current calibration data."""
        return self.calibration
    
    def get_progress(self) -> Tuple[str, int, int]:
        """Get calibration progress (phase_name, current, target)."""
        if self.calibration_phase == CalibrationPhase.COLLECTING_DOTS:
            return ("DOTS", len(self.dot_blinks), self.target_dots)
        elif self.calibration_phase == CalibrationPhase.COLLECTING_DASHES:
            return ("DASHES", len(self.dash_blinks), self.target_dashes)
        else:
            return ("DONE", 0, 0)
    
    def get_phase(self) -> CalibrationPhase:
        """Get current calibration phase."""
        return self.calibration_phase


# =============================================================================
# MORSE CODE DECODER MODULE
# =============================================================================

class MorseDecoder:
    """
    Decodes Morse code sequences into text.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Morse decoder.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.morse_dict = MORSE_CODE_DICT
        
        # State
        self.current_sequence = ""
        self.decoded_text = ""
        self.pending_letter = False
        self.pending_word = False
    
    def add_symbol(self, symbol: str):
        """
        Add a dot or dash to the current sequence.
        
        Args:
            symbol: '.' for dot, '-' for dash
        """
        if symbol in ['.', '-']:
            self.current_sequence += symbol
            self.pending_letter = True
            self.pending_word = True
    
    def process_letter_gap(self) -> Optional[str]:
        """
        Process a letter gap - decode current sequence.
        
        Returns:
            Decoded character or None
        """
        if not self.pending_letter or not self.current_sequence:
            return None
        
        self.pending_letter = False
        
        # Decode the sequence
        char = self.decode_sequence(self.current_sequence)
        
        if char:
            self.decoded_text += char
        else:
            # Unknown sequence - mark as unknown with ?
            self.decoded_text += "?"
        
        result = self.current_sequence
        self.current_sequence = ""
        
        return char
    
    def process_word_gap(self) -> bool:
        """
        Process a word gap - add newline.
        
        Returns:
            True if newline was added
        """
        # First process any pending letter
        self.process_letter_gap()
        
        if not self.pending_word:
            return False
        
        self.pending_word = False
        
        # Add newline if not already ending with newline
        if self.decoded_text and not self.decoded_text.endswith('\n'):
            self.decoded_text += '\n'
            return True
        
        return False

    def process_sentence_gap(self) -> bool:
        """
        Process a sentence gap - ensure word separation and terminate sentence.

        Returns:
            True if sentence boundary was added
        """
        # First process as word gap to flush pending symbols/letters.
        self.process_word_gap()

        if not self.decoded_text:
            return False

        # Use a double newline as explicit sentence boundary marker.
        if not self.decoded_text.endswith('\n\n'):
            self.decoded_text = self.decoded_text.rstrip('\n') + '\n\n'
            return True

        return False
    
    def decode_sequence(self, sequence: str) -> Optional[str]:
        """
        Decode a Morse sequence into a character.
        
        Args:
            sequence: Morse code sequence (e.g., '.-' for 'A')
            
        Returns:
            Decoded character or None if invalid
        """
        return self.morse_dict.get(sequence)
    
    def get_current_sequence(self) -> str:
        """Get the current Morse sequence in progress."""
        return self.current_sequence
    
    def get_decoded_text(self) -> str:
        """Get the decoded text so far."""
        return self.decoded_text
    
    def clear_sequence(self):
        """Clear the current sequence."""
        self.current_sequence = ""
        self.pending_letter = False
    
    def clear_text(self):
        """Clear all decoded text."""
        self.decoded_text = ""
        self.clear_sequence()
        self.pending_word = False
    
    def backspace(self):
        """Remove the last character from decoded text."""
        if self.decoded_text:
            self.decoded_text = self.decoded_text[:-1]
    
    def remove_unresolved(self):
        """Remove all unresolved (?) characters from decoded text."""
        self.decoded_text = self.decoded_text.replace("?", "")
        # Clean up extra spaces line-by-line and keep newline-separated words
        lines = self.decoded_text.split('\n')
        cleaned_lines = [' '.join(line.split()) for line in lines]
        self.decoded_text = '\n'.join(cleaned_lines).strip()
    
    def remove_last_symbol(self):
        """Remove the last symbol from current sequence."""
        if self.current_sequence:
            self.current_sequence = self.current_sequence[:-1]


# =============================================================================
# NLP CORRECTION MODULE (RESERVED EXTENSION)
# =============================================================================

class NLPCorrector(ABC):
    """
    Abstract base class for NLP-based text correction.
    This is a reserved extension point for future implementation.
    """
    
    @abstractmethod
    def correct(self, text: str) -> str:
        """
        Apply correction to the input text.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text
        """
        pass
    
    @abstractmethod
    def get_suggestions(self, text: str) -> List[str]:
        """
        Get correction suggestions for the input text.
        
        Args:
            text: Input text
            
        Returns:
            List of suggested corrections
        """
        pass


class RuleBasedCorrector(NLPCorrector):
    """
    Rule-based text corrector (placeholder implementation).
    """
    
    def __init__(self):
        """Initialize the rule-based corrector."""
        # Common word corrections
        self.corrections = {
            'teh': 'the',
            'adn': 'and',
            'taht': 'that',
            'wiht': 'with',
        }
    
    def correct(self, text: str) -> str:
        """Apply rule-based corrections."""
        words = text.split()
        corrected = []
        
        for word in words:
            lower = word.lower()
            if lower in self.corrections:
                corrected.append(self.corrections[lower])
            else:
                corrected.append(word)
        
        return ' '.join(corrected)
    
    def get_suggestions(self, text: str) -> List[str]:
        """Get suggestions (returns single correction for now)."""
        corrected = self.correct(text)
        if corrected != text:
            return [corrected]
        return []


def load_indobert_corrector_model():
    """
    Load IndoBERT Seq2Seq model and tokenizer with Streamlit caching.
    Model is loaded only once and reused across reruns.
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    import torch
    
    # Model is in a subfolder on HuggingFace
    model_repo = "ZerXXX/indobert-corrector"
    subfolder = "indoBERT-best-corrector"
    
    # Load tokenizer and model from Hugging Face Hub with subfolder
    tokenizer = AutoTokenizer.from_pretrained(model_repo, subfolder=subfolder)
    model = EncoderDecoderModel.from_pretrained(model_repo, subfolder=subfolder)
    
    # Explicitly set token IDs from config.json (required for generation)
    # These values are from the model's config.json on HuggingFace
    model.config.decoder_start_token_id = 2  # [CLS] token
    model.config.eos_token_id = 3            # [SEP] token
    model.config.pad_token_id = 0            # [PAD] token
    model.config.bos_token_id = 2            # Same as decoder_start
    
    # Also set on generation_config
    if model.generation_config is not None:
        model.generation_config.decoder_start_token_id = 2
        model.generation_config.eos_token_id = 3
        model.generation_config.pad_token_id = 0
        model.generation_config.bos_token_id = 2
    
    # Set model to evaluation mode
    model.eval()
    
    # Determine device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model, tokenizer, device


class IndoBERTCorrector(NLPCorrector):
    """
    IndoBERT Seq2Seq text corrector using EncoderDecoderModel.
    Loads model from Hugging Face Hub: ZerXXX/indobert-corrector/indoBERT-best-corrector
    """
    
    def __init__(self):
        """
        Initialize the IndoBERT corrector.
        Uses Streamlit cached resource for model loading.
        """
        self.model, self.tokenizer, self.device = load_indobert_corrector_model()
        self.max_length = 64
        self.num_beams = 4
    
    def correct(self, text: str) -> str:
        """
        Apply IndoBERT Seq2Seq correction to the input text.
        
        Args:
            text: Input text to correct
            
        Returns:
            Corrected text
        """
        if not text or not text.strip():
            return text
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate correction with deterministic settings (no sampling)
            # Token IDs from model config.json on HuggingFace
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    do_sample=False,
                    early_stopping=True,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                    pad_token_id=0,
                    bos_token_id=2
                )
            
            # Decode output
            corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return corrected if corrected else text
            
        except Exception as e:
            print(f"IndoBERT correction error: {e}")
            return text
    
    def get_suggestions(self, text: str) -> List[str]:
        """
        Get correction suggestions for the input text.
        Uses beam search to generate multiple candidates.
        
        Args:
            text: Input text
            
        Returns:
            List of suggested corrections
        """
        if not text or not text.strip():
            return []
        
        try:
            import torch
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate multiple suggestions using beam search
            # Token IDs from model config.json on HuggingFace
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_length,
                    num_beams=self.num_beams,
                    num_return_sequences=min(self.num_beams, 3),
                    do_sample=False,
                    early_stopping=True,
                    decoder_start_token_id=2,
                    eos_token_id=3,
                    pad_token_id=0,
                    bos_token_id=2
                )
            
            # Decode all outputs
            suggestions = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                if decoded and decoded != text and decoded not in suggestions:
                    suggestions.append(decoded)
            
            return suggestions
            
        except Exception as e:
            print(f"IndoBERT suggestions error: {e}")
            return []


class NLPCorrectionManager:
    """
    Manager for NLP-based text correction.
    Supports toggling and pluggable correctors.
    """
    
    def __init__(self):
        self.enabled = False
        self.corrector: Optional[NLPCorrector] = None
        self.raw_text = ""
        self.corrected_text = ""
        self.corrected_sentences: List[str] = []
        self.sentence_suggestions: List[List[str]] = []  # up to 3 per sentence
        self._raw_parts: List[str] = []  # raw text of each completed sentence

    def _ensure_corrector_loaded(self):
        """Load IndoBERT on first use."""
        if self.corrector is None:
            self.corrector = IndoBERTCorrector()
    
    def set_corrector(self, corrector: NLPCorrector):
        """
        Set the NLP corrector to use.
        
        Args:
            corrector: NLPCorrector implementation
        """
        self.corrector = corrector
    
    def enable(self):
        """Enable NLP correction."""
        self.enabled = True
    
    def disable(self):
        """Disable NLP correction."""
        self.enabled = False
    
    def toggle(self) -> bool:
        """Toggle NLP correction on/off. Loads the IndoBERT model on first enable."""
        new_state = not self.enabled
        if new_state:
            self._ensure_corrector_loaded()
        self.enabled = new_state
        return self.enabled

    def reset(self):
        """Reset internal NLP tracking state without touching raw decoded text."""
        self.raw_text = ""
        self.corrected_text = ""
        self.corrected_sentences = []
        self.sentence_suggestions = []
        self._raw_parts = []

    def _get_suggestions(self, sentence: str) -> List[str]:
        """Return up to 3 NLP correction suggestions for a completed sentence.
        Falls back to [raw_text] if the model is not ready yet."""
        compact = ' '.join(sentence.split())
        if not compact:
            return []
        if self.corrector is None:
            return [compact]
        try:
            suggestions = self.corrector.get_suggestions(compact)
            return suggestions[:3] if suggestions else [compact]
        except Exception as e:
            print(f"NLP suggestion error: {e}")
            return [compact]

    def get_structured_sentences(self) -> List[Dict]:
        """Return completed sentences paired with their NLP suggestions for the frontend."""
        result = []
        for i in range(len(self._raw_parts)):
            raw = self._raw_parts[i]
            suggs = self.sentence_suggestions[i] if i < len(self.sentence_suggestions) else [raw]
            result.append({"raw": raw, "suggestions": suggs})
        return result

    def process(self, text: str, sentence_finished: bool = False) -> str:
        self.raw_text = text

        if self.enabled:
            parts = text.split('\n\n')

            if sentence_finished:
                completed_parts = [part for part in parts if part.strip()]
            else:
                completed_parts = [part for part in parts[:-1] if part.strip()]

            while len(self.corrected_sentences) < len(completed_parts):
                idx = len(self.corrected_sentences)
                raw_part = completed_parts[idx]
                self._raw_parts.append(raw_part)
                suggestions = self._get_suggestions(raw_part)
                self.sentence_suggestions.append(suggestions)
                self.corrected_sentences.append(suggestions[0] if suggestions else raw_part)

            self.corrected_text = '\n\n'.join(self.corrected_sentences)
            return self.corrected_text

        return text

    def get_suggestions(self, text: str) -> List[str]:
        """Get correction suggestions."""
        if self.corrector:
            return self.corrector.get_suggestions(text)
        return []


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class EyeBlinkMorseSystem:
    """
    Main system class that orchestrates all components for eye-blink
    to Morse code conversion.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the eye-blink Morse code system.
        
        Args:
            config: System configuration (uses defaults if None)
        """
        self.config = config or SystemConfig()
        
        # Initialize components
        self.eye_analyzer = EyeAnalyzer()
        self.yolo_classifier = YOLOEyeClassifier(
            self.config.yolo_model_path,
            self.config.use_gpu
        )
        self.confidence_fusion = ConfidenceFusion(
            self.config.smoothing_window,
            self.config.ema_alpha
        )
        self.calibration_manager = CalibrationManager(self.config)
        self.blink_detector = BlinkDetector(
            self.config, 
            self.calibration_manager.get_calibration()
        )
        self.morse_decoder = MorseDecoder(self.config)
        self.nlp_manager = NLPCorrectionManager()
        
        # State
        self.is_running = False
        self.current_confidence = 0.5
        self.current_eye_state = EyeState.UNKNOWN
        self.last_blink_event: Optional[BlinkEvent] = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.processing_time_ms = 0.0
    
    def process_frame(self, frame: np.ndarray, enable_detection: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Process a single frame through the entire pipeline.
        
        Args:
            frame: Input frame (BGR)
            enable_detection: Whether to run blink decoding and Morse pipeline
            
        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        start_time = time.time()
        
        results = {
            'eye_state': EyeState.UNKNOWN,
            'confidence': 0.5,
            'ear': 0.0,
            'yolo_result': YOLOResult(),
            'blink_event': None,
            'morse_sequence': '',
            'raw_decoded_text': '',
            'nlp_smoothed_text': '',
            'decoded_text': '',
            'is_calibrating': self.calibration_manager.is_calibrating,
            'calibration_progress': self.calibration_manager.get_progress(),  # (phase, current, target)
            'calibration_phase': self.calibration_manager.get_phase(),
            'fps': self.fps,
        }
        
        # 1. Eye analysis with MediaPipe
        eye_data, annotated_frame = self.eye_analyzer.process_frame(frame, self.config)
        results['ear'] = eye_data.avg_ear
        
        if not eye_data.landmarks_detected:
            # Fallback: no face detected
            return annotated_frame, results
        
        # 2. YOLO classification
        yolo_result = self.yolo_classifier.classify_dual_eye(
            eye_data.left_crop,
            eye_data.right_crop
        )
        results['yolo_result'] = yolo_result
        
        # 3. Confidence fusion
        raw_confidence = self.confidence_fusion.fuse(
            yolo_result,
            eye_data.normalized_ear,
            self.config.alpha
        )
        
        # Apply smoothing
        smoothed_confidence = self.confidence_fusion.smooth_ema(raw_confidence)
        self.current_confidence = smoothed_confidence
        results['confidence'] = smoothed_confidence
        
        # Determine eye state
        if smoothed_confidence >= self.config.blink_threshold:
            self.current_eye_state = EyeState.OPEN
        else:
            self.current_eye_state = EyeState.CLOSED
        results['eye_state'] = self.current_eye_state
        
        # 4. Blink detection and Morse decoding
        sentence_finished = False
        if enable_detection:
            if not self.calibration_manager.is_calibrating:
                # Update calibration reference
                self.blink_detector.calibration = self.calibration_manager.get_calibration()
                
                blink_event = self.blink_detector.process(smoothed_confidence)
                
                if blink_event:
                    self.last_blink_event = blink_event
                    results['blink_event'] = blink_event
                    
                    # Add to Morse sequence
                    self.morse_decoder.add_symbol(blink_event.blink_type.value)

                # Check for sentence/word/letter gaps (largest gap first)
                if self.blink_detector.is_sentence_gap():
                    sentence_finished = self.morse_decoder.process_sentence_gap()
                elif self.blink_detector.is_word_gap():
                    self.morse_decoder.process_word_gap()
                elif self.blink_detector.is_letter_gap():
                    self.morse_decoder.process_letter_gap()
            else:
                # Calibration mode
                blink_event = self.blink_detector.process(smoothed_confidence)
                if blink_event:
                    self.calibration_manager.add_blink(blink_event.duration_ms)
                results['calibration_progress'] = self.calibration_manager.get_progress()
        
        # 5. Get Morse state
        results['morse_sequence'] = self.morse_decoder.get_current_sequence()
        
        # 6. Apply NLP correction if enabled
        raw_text = self.morse_decoder.get_decoded_text()
        results['raw_decoded_text'] = raw_text
        results['nlp_smoothed_text'] = self.nlp_manager.process(raw_text, sentence_finished=sentence_finished)
        results['decoded_text'] = raw_text
        results['sentence_finished'] = sentence_finished
        
        # Update performance metrics
        self.frame_count += 1
        self.processing_time_ms = (time.time() - start_time) * 1000
        
        # Add overlays to frame
        annotated_frame = self._add_overlays(annotated_frame, results)
        
        return annotated_frame, results
    
    def _add_overlays(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Add status overlays to the frame."""
        h, w = frame.shape[:2]
        
        # Eye state indicator
        state_color = (0, 255, 0) if results['eye_state'] == EyeState.OPEN else (0, 0, 255)
        cv2.putText(frame, f"Eye: {results['eye_state'].value}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # Confidence bar
        conf = results['confidence']
        bar_width = int(200 * conf)
        cv2.rectangle(frame, (10, 50), (210, 70), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, 50), (10 + bar_width, 70), state_color, -1)
        cv2.putText(frame, f"Conf: {conf:.2f}", 
                    (220, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # EAR value
        cv2.putText(frame, f"EAR: {results['ear']:.3f}", 
                    (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Current Morse sequence
        if results['morse_sequence']:
            cv2.putText(frame, f"Morse: {results['morse_sequence']}", 
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Decoded text
        if results['decoded_text']:
            cv2.putText(frame, f"Text: {results['decoded_text'][-30:]}", 
                        (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Calibration indicator
        if results['is_calibrating']:
            progress = results['calibration_progress']
            cv2.putText(frame, f"CALIBRATING: {progress[0]}/{progress[1]}", 
                        (w // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        return frame
    
    def start_calibration(self, method: CalibrationMethod = CalibrationMethod.FREE_BLINK,
                          target_blinks: int = 5):
        """Start calibration process."""
        self.calibration_manager.start_calibration(method, target_blinks)
        self.blink_detector.reset()

    def next_calibration_step(self):
        """Advance calibration to the next phase regardless of sample count.

        - If collecting dots, jump to collecting dashes.
        - If collecting dashes, finalize calibration with whatever was collected.
        - Otherwise no-op.
        """
        mgr = self.calibration_manager
        if not mgr.is_calibrating:
            return
        if mgr.calibration_phase == CalibrationPhase.COLLECTING_DOTS:
            mgr.calibration_phase = CalibrationPhase.COLLECTING_DASHES
        elif mgr.calibration_phase == CalibrationPhase.COLLECTING_DASHES:
            mgr._finalize_calibration()

    def reset_calibration(self):
        """Reset calibration."""
        self.calibration_manager.reset()

    def set_calibration_complete_callback(self, callback):
        """Wire an external persistence hook into the calibration manager."""
        self.calibration_manager.on_calibration_complete = callback

    def apply_calibration(self, data: CalibrationData):
        """Replace the active calibration with a pre-existing CalibrationData
        (e.g. loaded from a per-user database row).

        The blink detector keeps a direct reference to the calibration object,
        so we must also re-point it at the new instance.
        """
        self.calibration_manager.calibration = data
        self.calibration_manager.is_calibrating = False
        self.calibration_manager.calibration_phase = CalibrationPhase.COMPLETED
        self.calibration_manager.dot_blinks = list(data.dot_durations)
        self.calibration_manager.dash_blinks = list(data.dash_durations)
        self.blink_detector.calibration = data
    
    def clear_text(self):
        """Clear decoded text."""
        self.morse_decoder.clear_text()
        self.nlp_manager.reset()
    
    def toggle_nlp(self) -> bool:
        """Toggle NLP correction."""
        return self.nlp_manager.toggle()
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def close(self):
        """Release resources."""
        self.eye_analyzer.close()

