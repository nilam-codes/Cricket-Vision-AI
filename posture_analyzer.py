"""
posture_analyzer.py
===================
Rule-Based Cricket Batting Posture Analysis Engine.

Evaluates human body landmarks and geometric joint angles against
cricket batting fundamentals to provide actionable, objective coaching feedback.
"""

from typing import Dict, Any, List, Optional


class CricketPostureAnalyzer:
    """
    Expert rule-based analysis engine for beginner cricket batting stances.
    Evaluates knee flexion, torso inclination, head level, arm position, and stance balance.
    """

    # Configurable biomechanical thresholds for cricket stance
    THRESHOLDS = {
        "knee_ideal_min": 138.0,       # Athletic knee flexion lower bound
        "knee_ideal_max": 168.0,       # Athletic knee flexion upper bound
        "knee_locked": 172.0,          # Completely straight/locked knee
        "knee_overly_bent": 125.0,     # Excessive crouch

        "torso_tilt_min": 5.0,         # Minimum athletic forward lean (degrees from vertical)
        "torso_tilt_max": 22.0,        # Maximum athletic forward lean
        "torso_excessive_lean": 28.0,  # Excessive forward hunch/lean

        "head_tilt_max": 8.0,          # Maximum allowable eye tilt for level vision (degrees)

        "stance_width_min": 0.85,      # Minimum ankle-to-shoulder width ratio
        "stance_width_max": 1.55,      # Maximum ankle-to-shoulder width ratio

        "elbow_ideal_min": 95.0,       # Relaxed elbow flexion lower bound
        "elbow_ideal_max": 150.0,      # Relaxed elbow flexion upper bound
        "elbow_stiff": 165.0           # Locked/stiff arm
    }

    def __init__(self, batting_hand="Right-Hand"):
        """
        Args:
            batting_hand (str): 'Right-Hand' or 'Left-Hand' batsman stance.
        """
        self.batting_hand = batting_hand

    def set_batting_hand(self, batting_hand: str):
        """Update the player's batting hand configuration."""
        self.batting_hand = batting_hand

    def analyze(self, landmarks: Optional[Dict[str, Any]], angles: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute comprehensive rule-based evaluation of batting posture.

        Args:
            landmarks: dictionary of 33 normalized landmarks from PoseDetector.
            angles: dictionary of calculated angles from PoseDetector.

        Returns:
            dict containing:
                - stance_status (str): High-level verdict
                - status_code (str): 'balanced', 'adjustment_needed', 'incomplete'
                - overall_score (int): Posture score from 0 to 100
                - knee_analysis (dict)
                - torso_analysis (dict)
                - balance_analysis (dict)
                - head_analysis (dict)
                - arm_analysis (dict)
                - positive_observations (list of str)
                - areas_for_improvement (list of str)
                - coaching_recommendations (list of str)
                - confidence_level (str)
                - disclaimer (str)
        """
        if not landmarks or not angles:
            return self._empty_result("No person or pose detected in frame.")

        # Identify front vs back limbs based on batting hand
        is_rhb = "right" in self.batting_hand.lower()
        front_knee_key = "left_knee" if is_rhb else "right_knee"
        back_knee_key = "right_knee" if is_rhb else "left_knee"
        front_elbow_key = "left_elbow" if is_rhb else "right_elbow"
        back_elbow_key = "right_elbow" if is_rhb else "left_elbow"

        front_knee = angles.get(front_knee_key)
        back_knee = angles.get(back_knee_key)
        front_elbow = angles.get(front_elbow_key)
        back_elbow = angles.get(back_elbow_key)
        torso_tilt = angles.get("torso_tilt")
        stance_ratio = angles.get("stance_width_ratio")
        head_tilt = angles.get("head_tilt")

        # Track category scores (0 - 100) and feedback items
        scores = []
        positives: List[str] = []
        improvements: List[str] = []
        recommendations: List[str] = []

        # ----------------------------------------------------
        # 1. KNEE POSITION ANALYSIS
        # ----------------------------------------------------
        knee_status = "Good"
        knee_desc = "Good athletic knee flexion observed."
        knee_score = 85

        active_knees = [k for k in [front_knee, back_knee] if k is not None]
        if not active_knees:
            knee_status = "Uncertain"
            knee_desc = "Lower body landmarks partially obscured or low confidence."
            knee_score = 60
        else:
            has_locked = any(k >= self.THRESHOLDS["knee_locked"] for k in active_knees)
            has_crouched = any(k <= self.THRESHOLDS["knee_overly_bent"] for k in active_knees)

            if has_locked:
                knee_status = "Locked Knees"
                knee_desc = "Knees appear straight and locked. Restricts weight transfer and quick foot movement."
                knee_score = 50
                improvements.append("Straight/locked knee posture detected.")
                recommendations.append("Flex knees slightly (aim for ~145°–160°) to establish a springy, athletic base.")
            elif has_crouched:
                knee_status = "Excessive Crouch"
                knee_desc = "Knees appear bent too low, which can tire leg muscles and limit reach on high deliveries."
                knee_score = 65
                improvements.append("Knee bend is deeper than standard batting stance.")
                recommendations.append("Stand slightly taller with a relaxed, natural knee bend.")
            else:
                knee_status = "Balanced Flexion"
                knee_desc = "Excellent athletic knee flexion maintaining readiness for front-foot or back-foot drive."
                knee_score = 95
                positives.append("Great knee flexion providing a stable, reactive base.")

        scores.append(knee_score)

        # ----------------------------------------------------
        # 2. UPPER BODY & TORSO ALIGNMENT
        # ----------------------------------------------------
        torso_status = "Good"
        torso_desc = "Torso alignment appears steady."
        torso_score = 85

        if torso_tilt is None:
            torso_status = "Uncertain"
            torso_desc = "Shoulder or hip landmarks obscured; cannot reliably measure torso inclination."
            torso_score = 60
        else:
            if torso_tilt < self.THRESHOLDS["torso_tilt_min"]:
                torso_status = "Too Upright"
                torso_desc = f"Upper body is very upright ({torso_tilt:.1f}° tilt). Batsmen benefit from a gentle forward lean."
                torso_score = 60
                improvements.append("Upper body is standing too upright.")
                recommendations.append("Hinge gently from the hips to tilt the spine forward toward the pitch (~10°–18°).")
            elif torso_tilt > self.THRESHOLDS["torso_excessive_lean"]:
                torso_status = "Over-leaning Forward"
                torso_desc = f"Noticeable forward lean detected ({torso_tilt:.1f}°). Risk of falling over to the off-side."
                torso_score = 55
                improvements.append("Excessive forward tilt from the hips.")
                recommendations.append("Keep your chest open and centered; avoid falling over your toes.")
            else:
                torso_status = "Athletic Posture"
                torso_desc = f"Torso shows a well-calibrated forward tilt ({torso_tilt:.1f}°), ideal for balance and head stability."
                torso_score = 95
                positives.append("Solid athletic spine angle supporting good center-of-gravity control.")

        scores.append(torso_score)

        # ----------------------------------------------------
        # 3. BODY BALANCE & STANCE WIDTH
        # ----------------------------------------------------
        balance_status = "Good"
        balance_desc = "Stance width appears stable."
        balance_score = 80

        if stance_ratio is None:
            balance_status = "Uncertain"
            balance_desc = "Ankles or shoulders not fully detected to determine stance width."
            balance_score = 60
        else:
            if stance_ratio < self.THRESHOLDS["stance_width_min"]:
                balance_status = "Narrow Stance"
                balance_desc = f"Feet are close together ({stance_ratio:.2f}x shoulder width). Base of support is reduced."
                balance_score = 55
                improvements.append("Stance width is narrower than shoulder width.")
                recommendations.append("Widen your feet to roughly shoulder-width (~1.0x–1.3x) for firmer footing.")
            elif stance_ratio > self.THRESHOLDS["stance_width_max"]:
                balance_status = "Wide Stance"
                balance_desc = f"Stance appears quite wide ({stance_ratio:.2f}x shoulder width). May restrict nimble footwork."
                balance_score = 65
                improvements.append("Feet spread wider than usual batting stance.")
                recommendations.append("Bring feet slightly closer so you can stride forward or back without overreaching.")
            else:
                balance_status = "Well Balanced"
                balance_desc = f"Solid base of support ({stance_ratio:.2f}x shoulder width). Promotes quick weight transfer."
                balance_score = 95
                positives.append("Balanced base with feet positioned approximately shoulder-width apart.")

        scores.append(balance_score)

        # ----------------------------------------------------
        # 4. HEAD POSITION & EYE ALIGNMENT
        # ----------------------------------------------------
        head_status = "Good"
        head_desc = "Head alignment appears steady."
        head_score = 85

        if head_tilt is None:
            head_status = "Uncertain"
            head_desc = "Facial keypoints not clearly visible."
            head_score = 65
        else:
            if head_tilt > self.THRESHOLDS["head_tilt_max"]:
                head_status = "Head Tilted"
                head_desc = f"Head appears tilted by {head_tilt:.1f}°. Cricket coaching emphasizes keeping both eyes level."
                head_score = 60
                improvements.append("Head tilted sideways away from horizontal.")
                recommendations.append("Keep your eyes level with the horizon to track the ball's trajectory and seam cleanly.")
            else:
                head_status = "Level & Steady"
                head_desc = f"Eyes and head are level ({head_tilt:.1f}° tilt), enabling optimal visual tracking of the bowler's release."
                head_score = 95
                positives.append("Head is upright and eyes are horizontal with the pitch.")

        scores.append(head_score)

        # ----------------------------------------------------
        # 5. ARM & ELBOW POSITION
        # ----------------------------------------------------
        arm_status = "Good"
        arm_desc = "Arm positioning appears functional."
        arm_score = 80

        active_elbows = [e for e in [front_elbow, back_elbow] if e is not None]
        if not active_elbows:
            arm_status = "Uncertain"
            arm_desc = "Elbow or wrist keypoints not detected."
            arm_score = 65
        else:
            has_stiff = any(e >= self.THRESHOLDS["elbow_stiff"] for e in active_elbows)
            if has_stiff:
                arm_status = "Stiff Arms"
                arm_desc = "Arms/elbows appear rigidly straight. Can lead to a rigid backlift and slow bat swing."
                arm_score = 60
                improvements.append("Arms appear locked straight rather than soft.")
                recommendations.append("Maintain relaxed, soft elbows to encourage an unhindered bat swing.")
            else:
                arm_status = "Comfortable Flexion"
                arm_desc = "Elbows maintain relaxed flexion ready for backlift and bat path initiation."
                arm_score = 90
                positives.append("Arms and elbows relaxed in an athletic ready position.")

        scores.append(arm_score)

        # ----------------------------------------------------
        # OVERALL POSTURE ASSESSMENT & SCORE
        # ----------------------------------------------------
        overall_score = int(sum(scores) / len(scores)) if scores else 50

        if overall_score >= 85:
            stance_status = "Balanced Athletic Stance"
            status_code = "balanced"
        elif overall_score >= 68:
            stance_status = "Minor Adjustments Recommended"
            status_code = "adjustment_needed"
        else:
            stance_status = "Key Posture Corrections Needed"
            status_code = "correction_needed"

        # Provide a default positive if empty
        if not positives:
            positives.append("Body posture detected and landmarks recognized by MediaPipe.")

        # Provide default recommendation if empty
        if not recommendations:
            recommendations.append("Maintain this stable posture, breathe naturally, and stay light on your feet.")

        return {
            "stance_status": stance_status,
            "status_code": status_code,
            "overall_score": overall_score,
            "knee_analysis": {
                "status": knee_status,
                "description": knee_desc,
                "score": knee_score,
                "front_knee_angle": front_knee,
                "back_knee_angle": back_knee
            },
            "torso_analysis": {
                "status": torso_status,
                "description": torso_desc,
                "score": torso_score,
                "torso_tilt_deg": torso_tilt
            },
            "balance_analysis": {
                "status": balance_status,
                "description": balance_desc,
                "score": balance_score,
                "stance_width_ratio": stance_ratio
            },
            "head_analysis": {
                "status": head_status,
                "description": head_desc,
                "score": head_score,
                "head_tilt_deg": head_tilt
            },
            "arm_analysis": {
                "status": arm_status,
                "description": arm_desc,
                "score": arm_score,
                "front_elbow_angle": front_elbow,
                "back_elbow_angle": back_elbow
            },
            "positive_observations": positives,
            "areas_for_improvement": improvements,
            "coaching_recommendations": recommendations,
            "confidence_level": "High" if len(scores) >= 4 else "Moderate",
            "disclaimer": (
                "Educational Prototype: Rule-based heuristic assessments provide general posture feedback. "
                "Proper cricket batting technique varies significantly across individual stances, batting styles "
                "(e.g., open vs side-on), and coach philosophies. Not medical or professional biomechanics advice."
            )
        }

    def _empty_result(self, reason: str) -> Dict[str, Any]:
        """Return a structured fallback result when no pose is visible."""
        return {
            "stance_status": "No Batsman Detected",
            "status_code": "no_pose",
            "overall_score": 0,
            "knee_analysis": {"status": "Undetected", "description": reason, "score": 0},
            "torso_analysis": {"status": "Undetected", "description": reason, "score": 0},
            "balance_analysis": {"status": "Undetected", "description": reason, "score": 0},
            "head_analysis": {"status": "Undetected", "description": reason, "score": 0},
            "arm_analysis": {"status": "Undetected", "description": reason, "score": 0},
            "positive_observations": ["Waiting for user to step in front of the camera."],
            "areas_for_improvement": ["Ensure your full body (head to feet) is visible in the frame."],
            "coaching_recommendations": ["Position yourself 2–3 meters away with good lighting."],
            "confidence_level": "None",
            "disclaimer": "Educational Prototype: No pose data currently available."
        }

