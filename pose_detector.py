"""
pose_detector.py
================
MediaPipe Pose Estimation & Joint Angle Calculation Module for Cricket Vision AI.

Uses Google MediaPipe's modern Tasks API to detect 33 human pose landmarks,
calculates geometric joint angles using vector mathematics with NumPy,
and renders an analytical cricket coaching skeleton overlay.
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Landmark mapping constants for convenient reference
LANDMARK_NAMES = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "MOUTH_LEFT",
    10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX"
}

# Major skeletal connections for cricket stance visualization
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12),  # Shoulder to shoulder
    (11, 23),  # Left shoulder to left hip
    (12, 24),  # Right shoulder to right hip
    (23, 24),  # Hip to hip
    # Left Arm
    (11, 13),  # Shoulder to elbow
    (13, 15),  # Elbow to wrist
    # Right Arm
    (12, 14),  # Shoulder to elbow
    (14, 16),  # Elbow to wrist
    # Left Leg
    (23, 25),  # Hip to knee
    (25, 27),  # Knee to ankle
    (27, 29),  # Ankle to heel
    (29, 31),  # Heel to toe
    (27, 31),  # Ankle to toe
    # Right Leg
    (24, 26),  # Hip to knee
    (26, 28),  # Knee to ankle
    (28, 30),  # Ankle to heel
    (30, 32),  # Heel to toe
    (28, 32),  # Ankle to toe
    # Head / Neck
    (0, 11),   # Nose to left shoulder
    (0, 12),   # Nose to right shoulder
    (1, 2), (2, 3), (4, 5), (5, 6) # Eyes
]


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the interior angle (in degrees) formed by three points: A - B - C.
    The middle point 'point_b' is the vertex of the angle.

    Args:
        point_a: [x, y] or [x, y, z] coordinates of first point
        point_b: [x, y] or [x, y, z] coordinates of vertex (joint)
        point_c: [x, y] or [x, y, z] coordinates of third point

    Returns:
        float: Angle in degrees between 0.0 and 180.0, rounded to 1 decimal place.
               Returns 0.0 if points are degenerate.
    """
    if point_a is None or point_b is None or point_c is None:
        return 0.0

    a = np.array([point_a[0], point_a[1]], dtype=np.float64)
    b = np.array([point_b[0], point_b[1]], dtype=np.float64)
    c = np.array([point_c[0], point_c[1]], dtype=np.float64)

    # Vectors from vertex B to points A and C
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    # Avoid division by zero
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    # Cosine formula: dot(ba, bc) / (|ba| * |bc|)
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    # Clip to handle slight floating point arithmetic inaccuracies
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return round(float(np.degrees(angle)), 1)


class PoseDetector:
    """
    Encapsulates MediaPipe PoseLandmarker model loading, inference,
    landmark coordinate extraction, and analytical skeleton drawing.
    """

    def __init__(self, model_path=None, min_detection_confidence=0.5, min_presence_confidence=0.5):
        """
        Initialize the PoseDetector with a pre-trained MediaPipe task model.
        """
        # Determine model path
        if model_path is None:
            # Check current working directory for available models
            base_dir = os.path.dirname(os.path.abspath(__file__))
            preferred_models = [
                os.path.join(base_dir, "pose_landmarker_full.task"),
                os.path.join(base_dir, "pose_landmarker_lite.task"),
                "pose_landmarker_full.task",
                "pose_landmarker_lite.task"
            ]
            for p in preferred_models:
                if os.path.exists(p):
                    model_path = p
                    break

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe Pose model not found at {model_path}. "
                "Ensure pose_landmarker_full.task is in the project directory."
            )

        self.model_path = model_path
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence

        # Configure MediaPipe PoseLandmarkerOptions
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def detect(self, bgr_image: np.ndarray | None):
        """
        Run MediaPipe Pose estimation on an OpenCV BGR image frame.

        Args:
            bgr_image: NumPy ndarray (H, W, 3) in BGR color space.

        Returns:
            dict: {
                "detected": bool,
                "landmarks": list of dicts or None,
                "raw_result": PoseLandmarkerResult
            }
        """
        if bgr_image is None or bgr_image.size == 0:
            return {"detected": False, "landmarks": None, "raw_result": None}

        h, w = bgr_image.shape[:2]

        # MediaPipe expects RGB format
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        try:
            result = self.detector.detect(mp_image)
        except Exception as e:
            print(f"[PoseDetector] Inference error: {e}")
            return {"detected": False, "landmarks": None, "raw_result": None}

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return {"detected": False, "landmarks": None, "raw_result": result}

        # Extract first person's landmarks
        first_pose = result.pose_landmarks[0]
        landmarks_dict = {}

        for idx, lm in enumerate(first_pose):
            name = LANDMARK_NAMES.get(idx, f"LANDMARK_{idx}")
            landmarks_dict[name] = {
                "id": idx,
                "name": name,
                "x": lm.x,                     # Normalized [0, 1]
                "y": lm.y,                     # Normalized [0, 1]
                "z": lm.z,                     # Normalized relative depth
                "visibility": getattr(lm, "visibility", 1.0) or 1.0,
                "presence": getattr(lm, "presence", 1.0) or 1.0,
                "px_x": int(lm.x * w),         # Pixel coordinate X
                "px_y": int(lm.y * h)          # Pixel coordinate Y
            }

        return {
            "detected": True,
            "landmarks": landmarks_dict,
            "raw_result": result
        }

    def compute_all_angles(self, landmarks):
        """
        Compute cricket-relevant joint angles from extracted landmarks.

        Args:
            landmarks: dict of landmarks returned by detect()["landmarks"]

        Returns:
            dict: computed angles in degrees, or None if landmarks are missing/unreliable.
        """
        if not landmarks:
            return {}

        def get_coords(name):
            lm = landmarks.get(name)
            if lm and lm.get("visibility", 0) >= 0.35:
                return (lm["x"], lm["y"])
            return None

        # Extract landmark points
        l_shoulder = get_coords("LEFT_SHOULDER")
        r_shoulder = get_coords("RIGHT_SHOULDER")
        l_elbow    = get_coords("LEFT_ELBOW")
        r_elbow    = get_coords("RIGHT_ELBOW")
        l_wrist    = get_coords("LEFT_WRIST")
        r_wrist    = get_coords("RIGHT_WRIST")
        l_hip      = get_coords("LEFT_HIP")
        r_hip      = get_coords("RIGHT_HIP")
        l_knee     = get_coords("LEFT_KNEE")
        r_knee     = get_coords("RIGHT_KNEE")
        l_ankle    = get_coords("LEFT_ANKLE")
        r_ankle    = get_coords("RIGHT_ANKLE")
        nose       = get_coords("NOSE")

        angles = {}

        # 1. Knee Angles (Hip - Knee - Ankle)
        angles["left_knee"] = calculate_angle(l_hip, l_knee, l_ankle) if (l_hip and l_knee and l_ankle) else None
        angles["right_knee"] = calculate_angle(r_hip, r_knee, r_ankle) if (r_hip and r_knee and r_ankle) else None

        # 2. Elbow Angles (Shoulder - Elbow - Wrist)
        angles["left_elbow"] = calculate_angle(l_shoulder, l_elbow, l_wrist) if (l_shoulder and l_elbow and l_wrist) else None
        angles["right_elbow"] = calculate_angle(r_shoulder, r_elbow, r_wrist) if (r_shoulder and r_elbow and r_wrist) else None

        # 3. Hip Angles (Shoulder - Hip - Knee)
        angles["left_hip"] = calculate_angle(l_shoulder, l_hip, l_knee) if (l_shoulder and l_hip and l_knee) else None
        angles["right_hip"] = calculate_angle(r_shoulder, r_hip, r_knee) if (r_shoulder and r_hip and r_knee) else None

        # 4. Torso inclination angle relative to vertical axis
        if l_shoulder and r_shoulder and l_hip and r_hip:
            mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2.0, (l_shoulder[1] + r_shoulder[1]) / 2.0)
            mid_hip = ((l_hip[0] + r_hip[0]) / 2.0, (l_hip[1] + r_hip[1]) / 2.0)
            # Torso vector pointing from hip to shoulder
            dx = mid_shoulder[0] - mid_hip[0]
            dy = mid_shoulder[1] - mid_hip[1]  # note: y increases downwards in image coordinates
            # Angle with the upward vertical vector (0, -1)
            up_vector = np.array([0.0, -1.0])
            torso_vector = np.array([dx, dy])
            norm_t = np.linalg.norm(torso_vector)
            if norm_t > 1e-6:
                cos_tilt = np.dot(torso_vector, up_vector) / norm_t
                cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
                angles["torso_tilt"] = round(float(np.degrees(np.arccos(cos_tilt))), 1)
            else:
                angles["torso_tilt"] = None
        else:
            angles["torso_tilt"] = None

        # 5. Stance Width Ratio (Ankle spread vs Shoulder spread)
        if l_shoulder and r_shoulder and l_ankle and r_ankle:
            shoulder_width = float(np.linalg.norm(np.array(r_shoulder) - np.array(l_shoulder)))
            ankle_width = float(np.linalg.norm(np.array(r_ankle) - np.array(l_ankle)))
            if shoulder_width > 0.04:
                angles["stance_width_ratio"] = round(float(ankle_width / shoulder_width), 2)
            else:
                angles["stance_width_ratio"] = None
        else:
            angles["stance_width_ratio"] = None

        # 6. Head tilt (Deviation from horizontal plane)
        l_eye = get_coords("LEFT_EYE")
        r_eye = get_coords("RIGHT_EYE")
        if l_eye and r_eye:
            eye_dx = r_eye[0] - l_eye[0]
            eye_dy = r_eye[1] - l_eye[1]
            raw_angle = abs(np.degrees(np.arctan2(eye_dy, eye_dx))) % 180.0
            if raw_angle > 90.0:
                raw_angle = 180.0 - raw_angle
            angles["head_tilt"] = round(float(raw_angle), 1)
        else:
            angles["head_tilt"] = None

        return angles

    def draw_skeleton(self, image: np.ndarray, landmarks, angles=None, stance_status=None) -> np.ndarray:
        """
        Render a high-contrast cricket coaching skeleton overlay directly onto image.

        Args:
            image: OpenCV BGR image (ndarray)
            landmarks: dictionary of landmarks
            angles: computed angles dictionary
            stance_status: status string e.g. "BALANCED STANCE"

        Returns:
            annotated_image: ndarray with coaching graphics
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        if not landmarks:
            # Draw semi-transparent header banner showing NO BATSMAN DETECTED
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, 50), (18, 20, 28), -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            cv2.putText(
                annotated,
                "CRICKET VISION AI  |  LOOKING FOR BATSMAN...",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )
            return annotated

        # Colors (BGR)
        COLOR_CRICKET_GREEN = (0, 230, 118)   # Primary bone glow
        COLOR_CYAN          = (255, 229, 0)   # Secondary joint connection
        COLOR_WHITE         = (255, 255, 255) # Joint center
        COLOR_AMBER         = (0, 165, 255)   # Warning / attention
        COLOR_DARK_CARD     = (24, 28, 36)    # Overlay HUD background

        # 1. Draw Skeleton Bones (Lines)
        for start_id, end_id in SKELETON_CONNECTIONS:
            start_name = LANDMARK_NAMES.get(start_id)
            end_name = LANDMARK_NAMES.get(end_id)
            lm_start = landmarks.get(start_name)
            lm_end = landmarks.get(end_name)

            if lm_start and lm_end:
                if lm_start.get("visibility", 0) > 0.35 and lm_end.get("visibility", 0) > 0.35:
                    pt1 = (lm_start["px_x"], lm_start["px_y"])
                    pt2 = (lm_end["px_x"], lm_end["px_y"])

                    # Draw outer thicker soft glow
                    cv2.line(annotated, pt1, pt2, (40, 120, 50), 6, cv2.LINE_AA)
                    # Draw core crisp line
                    bone_color = COLOR_CRICKET_GREEN if start_id >= 23 else COLOR_CYAN
                    cv2.line(annotated, pt1, pt2, bone_color, 2, cv2.LINE_AA)

        # 2. Draw Landmark Joints (Circles)
        for name, lm in landmarks.items():
            if lm.get("visibility", 0) > 0.35:
                px = (lm["px_x"], lm["px_y"])
                idx = lm["id"]

                # Larger nodes for primary joints
                is_major_joint = idx in [11, 12, 13, 14, 23, 24, 25, 26, 27, 28]
                radius = 6 if is_major_joint else 4

                # Outer halo
                cv2.circle(annotated, px, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                # Primary ring
                node_color = COLOR_CRICKET_GREEN if is_major_joint else COLOR_CYAN
                cv2.circle(annotated, px, radius, node_color, -1, cv2.LINE_AA)
                # Core white dot
                cv2.circle(annotated, px, max(2, radius - 3), COLOR_WHITE, -1, cv2.LINE_AA)

        # 3. Draw On-Screen Angle Badges near joints
        if angles:
            def draw_angle_badge(pt, label, value):
                if pt and value is not None:
                    text = f"{label}: {value:.0f} deg"
                    x, y = pt
                    # Offset badge
                    bx = min(max(10, x + 15), w - 140)
                    by = min(max(25, y - 10), h - 15)

                    # Badge background
                    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(annotated, (bx - 4, by - th - 4), (bx + tw + 4, by + baseline + 2), COLOR_DARK_CARD, -1)
                    cv2.rectangle(annotated, (bx - 4, by - th - 4), (bx + tw + 4, by + baseline + 2), COLOR_CRICKET_GREEN, 1)
                    cv2.putText(annotated, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WHITE, 1, cv2.LINE_AA)

            # Left and right knee
            lk = landmarks.get("LEFT_KNEE")
            rk = landmarks.get("RIGHT_KNEE")
            if lk and lk.get("visibility", 0) > 0.4:
                draw_angle_badge((lk["px_x"], lk["px_y"]), "L-Knee", angles.get("left_knee"))
            if rk and rk.get("visibility", 0) > 0.4:
                draw_angle_badge((rk["px_x"], rk["px_y"]), "R-Knee", angles.get("right_knee"))

            # Left and right elbow
            le = landmarks.get("LEFT_ELBOW")
            re = landmarks.get("RIGHT_ELBOW")
            if le and le.get("visibility", 0) > 0.4:
                draw_angle_badge((le["px_x"], le["px_y"]), "L-Elbow", angles.get("left_elbow"))
            if re and re.get("visibility", 0) > 0.4:
                draw_angle_badge((re["px_x"], re["px_y"]), "R-Elbow", angles.get("right_elbow"))

        # 4. Top HUD Header
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), COLOR_DARK_CARD, -1)
        cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

        # Status text
        status_str = stance_status if stance_status else "BATSMAN DETECTED - SKELETON TRACKED"
        cv2.putText(
            annotated,
            "CRICKET VISION AI",
            (16, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            COLOR_WHITE,
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            f"|  {status_str}",
            (210, 30),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            COLOR_CRICKET_GREEN,
            1,
            cv2.LINE_AA
        )

        return annotated

    def close(self):
        """Release MediaPipe resources gracefully."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
