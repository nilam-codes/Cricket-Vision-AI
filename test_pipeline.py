"""
test_pipeline.py
================
Comprehensive automated test suite for Cricket Vision AI.
Validates math, MediaPipe model loading, posture rules, and image rendering.
"""

import os
import cv2
import numpy as np
from pose_detector import PoseDetector, calculate_angle
from posture_analyzer import CricketPostureAnalyzer


def test_calculate_angle():
    print("--> Testing calculate_angle mathematical functions...")
    # Right angle (90 deg)
    assert calculate_angle([0, 1], [0, 0], [1, 0]) == 90.0

    # Straight line (180 deg)
    assert calculate_angle([0, 1], [0, 0], [0, -1]) == 180.0

    # 45 deg angle
    assert calculate_angle([1, 1], [0, 0], [1, 0]) == 45.0

    # Collinear overlap (0 deg)
    assert calculate_angle([1, 0], [0, 0], [2, 0]) == 0.0

    # Degenerate zero magnitude vector (should handle division by zero safely)
    assert calculate_angle([0, 0], [0, 0], [1, 0]) == 0.0

    # None inputs (missing landmark handling)
    assert calculate_angle(None, [0, 0], [1, 0]) == 0.0
    print("[OK] All calculate_angle tests passed.")


def test_pose_detector_init_and_blank_frame():
    print("--> Testing PoseDetector initialization & blank frame handling...")
    detector = PoseDetector()
    assert detector is not None
    assert os.path.exists(detector.model_path)

    # Blank image (no person)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = detector.detect(blank_frame)
    assert res["detected"] is False
    assert res["landmarks"] is None

    # Draw on blank frame
    ann = detector.draw_skeleton(blank_frame, None)
    assert ann.shape == blank_frame.shape

    detector.close()
    print("[OK] PoseDetector blank frame handling passed.")


def test_posture_analyzer_rules():
    print("--> Testing CricketPostureAnalyzer rule evaluation...")
    analyzer = CricketPostureAnalyzer("Right-Hand")

    # Empty inputs
    empty_res = analyzer.analyze(None, None)
    assert empty_res["status_code"] == "no_pose"
    assert empty_res["overall_score"] == 0

    dummy_landmarks = {"NOSE": {"x": 0.5, "y": 0.2, "visibility": 0.95}}

    # Balanced Stance Case
    balanced_angles = {
        "left_knee": 150.0,
        "right_knee": 152.0,
        "left_elbow": 120.0,
        "right_elbow": 118.0,
        "left_hip": 160.0,
        "right_hip": 158.0,
        "torso_tilt": 12.0,
        "stance_width_ratio": 1.15,
        "head_tilt": 3.0
    }
    bal_res = analyzer.analyze(dummy_landmarks, balanced_angles)
    assert bal_res["status_code"] == "balanced"
    assert bal_res["overall_score"] >= 85
    assert len(bal_res["positive_observations"]) > 0

    # Locked Knees & Tilted Head Case
    locked_angles = {
        "left_knee": 178.0,
        "right_knee": 177.0,
        "left_elbow": 168.0,
        "right_elbow": 167.0,
        "left_hip": 178.0,
        "right_hip": 177.0,
        "torso_tilt": 1.0,
        "stance_width_ratio": 0.65,
        "head_tilt": 16.0
    }
    bad_res = analyzer.analyze(dummy_landmarks, locked_angles)
    assert bad_res["overall_score"] < 65
    assert any("locked" in x.lower() or "straight" in x.lower() for x in bad_res["areas_for_improvement"])

    # Left-Hand Batsman configuration
    analyzer.set_batting_hand("Left-Hand")
    lhb_res = analyzer.analyze(dummy_landmarks, balanced_angles)
    assert lhb_res["overall_score"] >= 80

    print("[OK] All CricketPostureAnalyzer rules passed.")


def test_end_to_end_image_detection():
    print("--> Testing end-to-end detection on sample batsman image...")
    sample_path = os.path.join("assets", "sample_cricket_stance.jpg")
    if not os.path.exists(sample_path):
        print("[WARN] Sample image not found, skipping image test.")
        return

    img = cv2.imread(sample_path)
    assert img is not None

    detector = PoseDetector()
    det_res = detector.detect(img)
    assert det_res["detected"] is True
    assert "LEFT_KNEE" in det_res["landmarks"]
    assert "RIGHT_KNEE" in det_res["landmarks"]

    angles = detector.compute_all_angles(det_res["landmarks"])
    assert angles.get("left_knee") is not None
    assert angles.get("right_knee") is not None

    analyzer = CricketPostureAnalyzer("Right-Hand")
    analysis = analyzer.analyze(det_res["landmarks"], angles)
    assert analysis["overall_score"] > 0
    assert len(analysis["coaching_recommendations"]) > 0

    annotated = detector.draw_skeleton(img, det_res["landmarks"], angles, analysis["stance_status"])
    assert annotated.shape == img.shape

    detector.close()
    print(f"[OK] End-to-end image test passed. Overall score: {analysis['overall_score']}/100.")


if __name__ == "__main__":
    print("==================================================")
    print("CRICKET VISION AI: RUNNING AUTOMATED TEST SUITE")
    print("==================================================")
    test_calculate_angle()
    test_pose_detector_init_and_blank_frame()
    test_posture_analyzer_rules()
    test_end_to_end_image_detection()
    print("==================================================")
    print("ALL TESTS COMPLETED SUCCESSFULLY! SYSTEM READY.")
    print("==================================================")
