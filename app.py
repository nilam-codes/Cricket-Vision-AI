"""
app.py
======
CRICKET VISION AI: Real-Time AI Batting Posture Analysis Dashboard.

Live Webcam Only: MediaPipe Pose estimation, NumPy geometric vector mathematics,
and cricket biomechanics coaching logic with thread-safe live dashboard updates.
"""

import time
import threading
from typing import cast, Optional, Dict, Any
import cv2
import numpy as np
from numpy.typing import NDArray
import streamlit as st

# Import local modules
from pose_detector import PoseDetector
from posture_analyzer import CricketPostureAnalyzer

# WebRTC components
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        WebRtcMode,
        RTCConfiguration
    )
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# =====================================================================
# 1. STREAMLIT PAGE CONFIGURATION & THEME STYLING
# =====================================================================
st.set_page_config(
    page_title="Cricket Vision AI - Live Batting Posture Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Sports Analytics Dark Theme CSS
CUSTOM_CSS = """
<style>
    /* Dark charcoal background and modern font */
    .stApp {
        background-color: #0d1117;
        color: #f0f6fc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main container padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Header styling */
    .dashboard-header {
        background: linear-gradient(135deg, #161b22 0%, #1a222e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .header-title-box {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .cricket-logo {
        font-size: 2.4rem;
        background: rgba(0, 230, 118, 0.12);
        padding: 0.5rem 0.75rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 230, 118, 0.3);
    }
    .header-title {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        color: #ffffff;
    }
    .header-subtitle {
        margin: 0;
        font-size: 0.92rem;
        color: #00E676;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(0, 230, 118, 0.15);
        color: #00E676;
        border: 1px solid rgba(0, 230, 118, 0.4);
        padding: 0.4rem 0.9rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Cards styling */
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00E676;
    }
    .card-title {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8b949e;
        margin-bottom: 0.4rem;
    }
    .card-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
    }
    .card-status {
        font-size: 0.82rem;
        color: #00E676;
        font-weight: 500;
    }

    /* Assessment banner */
    .assessment-card {
        background: linear-gradient(135deg, #1c2431 0%, #161b22 100%);
        border-left: 5px solid #00E676;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
    }
    .assessment-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.3rem;
    }
    .assessment-score {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00E676;
    }

    /* Streamlit sidebar tweaks */
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =====================================================================
# 2. THREAD-SAFE STATE STORAGE FOR WEBRTC -> STREAMLIT UI
# =====================================================================
class PostureResultStore:
    """
    Thread-safe shared storage bridging the WebRTC background video thread
    and Streamlit's dashboard rendering thread.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._data: Optional[Dict[str, Any]] = None
        self._last_update: float = 0.0

    def update(self, detected: bool, landmarks: Any, angles: Dict[str, Any], analysis: Dict[str, Any]) -> None:
        with self._lock:
            self._data = {
                "detected": detected,
                "landmarks": landmarks,
                "angles": angles,
                "analysis": analysis,
                "timestamp": time.time()
            }
            self._last_update = time.time()

    def get_latest(self, max_age_seconds: float = 1.5) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._data is None:
                return None
            # Stale check: if no new frame processed recently (batsman walked away)
            if time.time() - self._last_update > max_age_seconds:
                return None
            return dict(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data = None
            self._last_update = 0.0


@st.cache_resource
def get_result_store() -> PostureResultStore:
    """Singleton instance of thread-safe result store."""
    return PostureResultStore()

result_store = get_result_store()


# =====================================================================
# 3. CACHED MODEL INITIALIZATION
# =====================================================================
@st.cache_resource
def load_pose_detector() -> Optional[PoseDetector]:
    """Load and cache the MediaPipe PoseDetector to prevent reinitialization on reruns."""
    try:
        return PoseDetector()
    except Exception as e:
        st.error(f"Failed to load MediaPipe Pose Model: {e}")
        return None

@st.cache_resource
def load_posture_analyzer() -> CricketPostureAnalyzer:
    """Load and cache the CricketPostureAnalyzer."""
    return CricketPostureAnalyzer(batting_hand="Right-Hand")

detector = load_pose_detector()
analyzer = load_posture_analyzer()


# =====================================================================
# 4. WEBRTC VIDEO PROCESSOR (REAL-TIME STREAMING)
# =====================================================================
class CricketVideoProcessor(VideoProcessorBase):
    """
    Video processor for streamlit-webrtc.
    Processes live webcam frames, runs pose estimation, overlays skeleton lines & joint angles,
    and updates the thread-safe result store for live dashboard rendering.
    """
    def __init__(self):
        self.detector = detector
        self.analyzer = analyzer

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Mirror frame horizontally for intuitive live feedback
        img = cv2.flip(img, 1)

        if self.detector is not None:
            # 1. Pose estimation on live frame
            detection = self.detector.detect(img)
            landmarks = detection.get("landmarks")

            if detection.get("detected") and landmarks:
                # 2. Geometric angle calculations
                angles = self.detector.compute_all_angles(landmarks)
                # 3. Cricket biomechanics posture evaluation
                analysis = self.analyzer.analyze(landmarks, angles)
                # 4. Render coaching skeleton and angle badges
                annotated = self.detector.draw_skeleton(
                    img, landmarks, angles, stance_status=analysis.get("stance_status")
                )
                # 5. Push live result to thread-safe store
                result_store.update(True, landmarks, angles, analysis)
            else:
                annotated = self.detector.draw_skeleton(img, None)
                # Update store with undetected state
                empty_analysis = self.analyzer._empty_result("Waiting for batsman...")
                empty_analysis["stance_status"] = "Waiting for Batsman..."
                result_store.update(False, None, {}, empty_analysis)
        else:
            annotated = img

        frame_array = np.ascontiguousarray(annotated, dtype=np.uint8)
        frame_array = cast(NDArray[np.uint8], frame_array)

        return av.VideoFrame.from_ndarray(frame_array, format="bgr24")


# =====================================================================
# 5. SIDEBAR CONFIGURATION (LIVE WEBCAM CONTROLS ONLY)
# =====================================================================
with st.sidebar:
    st.markdown("### 🏏 Coaching Controls")
    st.write("Configure batsman orientation and stance guidelines.")

    batting_hand = st.radio(
        "Batsman Stance Orientation:",
        ("Right-Hand Batsman (RHB)", "Left-Hand Batsman (LHB)"),
        index=0,
        help="In cricket, RHB positions the left side toward the bowler, while LHB positions the right side."
    )
    is_rhb = "Right" in batting_hand
    if analyzer:
        analyzer.set_batting_hand("Right-Hand" if is_rhb else "Left-Hand")

    st.markdown("---")
    st.markdown("### 🎯 Stance Checklist")
    st.markdown(
        """
        - **Knees**: Soft athletic flexion ($140°–168°$).
        - **Spine**: Gentle forward hip hinge ($5°–22°$).
        - **Base**: Feet roughly shoulder-width ($0.9×–1.5×$).
        - **Head**: Still, level eyes parallel to horizon.
        - **Arms**: Soft elbows ready for backlift.
        """
    )

    st.markdown("---")
    st.markdown("### ℹ️ Live Webcam Engine")
    st.caption("Powered by Google MediaPipe Tasks API + NumPy + Streamlit WebRTC.")


# =====================================================================
# 6. DASHBOARD HEADER
# =====================================================================
st.markdown(
    """
    <div class="dashboard-header">
        <div class="header-title-box">
            <div class="cricket-logo">🏏</div>
            <div>
                <h1 class="header-title">CRICKET VISION AI</h1>
                <p class="header-subtitle">LIVE AI-POWERED BATTING POSTURE ANALYSIS & COACHING</p>
            </div>
        </div>
        <div>
            <div class="status-badge">● LIVE WEBCAM ENGINE</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =====================================================================
# 7. MAIN TWO-COLUMN WORKSPACE
# =====================================================================
left_col, right_col = st.columns([1.15, 1.0], gap="large")

with left_col:
    st.markdown("#### 📹 Live Batsman Video Stream & Skeleton Overlay")

    if WEBRTC_AVAILABLE:
        st.caption("Click **START** to activate your webcam. Stand 2–3 meters back so your full body is visible.")

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_ctx = webrtc_streamer(
            key="cricket-pose-streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=CricketVideoProcessor,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
            async_processing=True
        )

        if webrtc_ctx.state.playing:
            st.success("🟢 Camera is streaming. Stand in your batting stance facing the camera.")
        else:
            st.info("Webcam is idle. Press **START** above to begin live posture analysis.")
    else:
        st.error("streamlit-webrtc package not available. Ensure dependencies are installed.")


# =====================================================================
# 8. RIGHT COLUMN: PERIODIC LIVE METRICS PANEL (FRAGMENT AUTO-REFRESH)
# =====================================================================
with right_col:
    @st.fragment(run_every="500ms")
    def render_live_metrics(is_camera_playing: bool):
        st.markdown("#### 📊 Live Posture Assessment & Metrics")

        # Pull latest frame data from thread-safe store
        live_data = result_store.get_latest(max_age_seconds=1.5)

        if is_camera_playing and live_data and live_data.get("detected"):
            current_analysis = live_data.get("analysis", {})
            current_angles = live_data.get("angles", {})
            is_detected = True
        else:
            is_detected = False
            current_angles = {}
            if is_camera_playing:
                current_analysis = analyzer._empty_result("Waiting for batsman to step into camera frame...")
                current_analysis["stance_status"] = "Waiting for Batsman..."
            else:
                current_analysis = analyzer._empty_result("Camera idle. Press START to begin live posture analysis.")
                current_analysis["stance_status"] = "Camera Idle"

        score = current_analysis.get("overall_score", 0) if is_detected else 0
        stance_status = current_analysis.get("stance_status", "Waiting for Batsman...")
        status_code = current_analysis.get("status_code", "no_pose")

        # Status Badge Color
        if is_detected:
            if status_code == "balanced":
                badge_color = "#00E676"
            elif status_code == "adjustment_needed":
                badge_color = "#f0883e"
            else:
                badge_color = "#ff7b72"
        else:
            badge_color = "#8b949e"

        confidence_info = (
            f"🟢 Live tracking active (33 landmarks recognized)"
            if is_detected else
            ("⏳ Waiting for batsman to step into camera frame..." if is_camera_playing else "Click START above to activate webcam")
        )

        # Top Assessment Card
        st.markdown(
            f"""
            <div class="assessment-card" style="border-left-color: {badge_color};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; color: #8b949e;">OVERALL VERDICT</div>
                        <div class="assessment-title" style="color: {badge_color};">{stance_status}</div>
                    </div>
                    <div style="text-align: right;">
                        <div class="assessment-score" style="color: {badge_color};">{score} <span style="font-size: 1.1rem; color: #8b949e;">/ 100</span></div>
                    </div>
                </div>
                <div style="margin-top: 0.5rem; font-size: 0.88rem; color: #c9d1d9;">
                    {confidence_info}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Joint Angle Metric Cards (Row 1: Knees)
        st.markdown("##### 🦵 Knee Angles & Flexion")
        mcol1, mcol2 = st.columns(2)

        l_knee = current_angles.get("left_knee") if is_detected else None
        r_knee = current_angles.get("right_knee") if is_detected else None

        with mcol1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="card-title">Front Knee</div>
                    <div class="card-value">{f"{l_knee:.1f}°" if l_knee is not None else "--"}</div>
                    <div class="card-status">{ "Ideal Flexion (140°-168°)" if (l_knee and 138 <= l_knee <= 168) else "Target: 145°-165°" }</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with mcol2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="card-title">Back Knee</div>
                    <div class="card-value">{f"{r_knee:.1f}°" if r_knee is not None else "--"}</div>
                    <div class="card-status">{ "Ideal Flexion (140°-168°)" if (r_knee and 138 <= r_knee <= 168) else "Target: 145°-165°" }</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Joint Angle Metric Cards (Row 2: Torso & Stance)
        st.markdown("##### 📐 Torso Lean & Stance Balance")
        mcol3, mcol4 = st.columns(2)

        torso_tilt = current_angles.get("torso_tilt") if is_detected else None
        stance_ratio = current_angles.get("stance_width_ratio") if is_detected else None

        with mcol3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="card-title">Torso Spine Tilt</div>
                    <div class="card-value">{f"{torso_tilt:.1f}°" if torso_tilt is not None else "--"}</div>
                    <div class="card-status">{ "Athletic Lean (5°-20°)" if (torso_tilt and 5 <= torso_tilt <= 22) else "Target: 8°-18°" }</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with mcol4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="card-title">Stance Width Ratio</div>
                    <div class="card-value">{f"{stance_ratio:.2f}x" if stance_ratio is not None else "--"}</div>
                    <div class="card-status">{ "Balanced Base (0.9x-1.5x)" if (stance_ratio and 0.85 <= stance_ratio <= 1.55) else "Target: 1.0x-1.3x" }</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Joint Angle Metric Cards (Row 3: Elbows & Head)
        st.markdown("##### 🏏 Arms & Head Alignment")
        mcol5, mcol6 = st.columns(2)

        l_elbow = current_angles.get("left_elbow") if is_detected else None
        head_tilt = current_angles.get("head_tilt") if is_detected else None

        with mcol5:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="card-title">Lead Elbow Angle</div>
                    <div class="card-value">{f"{l_elbow:.1f}°" if l_elbow is not None else "--"}</div>
                    <div class="card-status">{ "Soft & Relaxed" if (l_elbow and 95 <= l_elbow <= 150) else "Target: 95°-145°" }</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with mcol6:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="card-title">Head Horizontal Tilt</div>
                    <div class="card-value">{f"{head_tilt:.1f}°" if head_tilt is not None else "--"}</div>
                    <div class="card-status">{ "Level Eyes (< 8°)" if (head_tilt and head_tilt <= 8) else "Keep Eyes Level" }</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Invoke fragment with current webcam playing state
    is_playing = bool(WEBRTC_AVAILABLE and 'webrtc_ctx' in locals() and webrtc_ctx and webrtc_ctx.state.playing)
    render_live_metrics(is_playing)


# =====================================================================
# 9. LOWER SECTION: PERIODIC LIVE ANALYSIS & RECOMMENDATIONS TABS
# =====================================================================
@st.fragment(run_every="500ms")
def render_live_lower_section(is_camera_playing: bool):
    live_data = result_store.get_latest(max_age_seconds=1.5)

    if is_camera_playing and live_data and live_data.get("detected"):
        current_analysis = live_data.get("analysis", {})
        is_detected = True
    else:
        is_detected = False
        if is_camera_playing:
            current_analysis = analyzer._empty_result("Waiting for batsman to step into camera frame...")
        else:
            current_analysis = analyzer._empty_result("Webcam is idle. Press START to begin live posture analysis.")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs([
        "🔍 Detailed Posture Analysis",
        "💡 Coaching Recommendations",
        "⚙️ How It Works (Pipeline)"
    ])

    with tab1:
        st.markdown("### 🔍 Biomechanical Stance Breakdown")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 🦵 Knee & Lower Body")
            k_info = current_analysis.get("knee_analysis", {})
            st.write(f"**Status:** {k_info.get('status', 'N/A')}")
            st.write(f"**Evaluation:** {k_info.get('description', '')}")

            st.markdown("#### 🧍 Torso & Spine Inclination")
            t_info = current_analysis.get("torso_analysis", {})
            st.write(f"**Status:** {t_info.get('status', 'N/A')}")
            st.write(f"**Evaluation:** {t_info.get('description', '')}")

        with col_b:
            st.markdown("#### ⚖️ Base Balance & Stance Width")
            b_info = current_analysis.get("balance_analysis", {})
            st.write(f"**Status:** {b_info.get('status', 'N/A')}")
            st.write(f"**Evaluation:** {b_info.get('description', '')}")

            st.markdown("#### 👁️ Head & Vision Alignment")
            h_info = current_analysis.get("head_analysis", {})
            st.write(f"**Status:** {h_info.get('status', 'N/A')}")
            st.write(f"**Evaluation:** {h_info.get('description', '')}")

            st.markdown("#### 🏏 Arm & Elbow Relaxedness")
            a_info = current_analysis.get("arm_analysis", {})
            st.write(f"**Status:** {a_info.get('status', 'N/A')}")
            st.write(f"**Evaluation:** {a_info.get('description', '')}")

    with tab2:
        st.markdown("### 💡 Coaching Recommendations & Adjustments")
        rcol1, rcol2 = st.columns(2)

        with rcol1:
            st.markdown("#### ✅ Positive Observations")
            positives = current_analysis.get("positive_observations", [])
            if positives and is_detected:
                for p in positives:
                    st.markdown(f"- 🟢 **{p}**")
            else:
                st.write("Awaiting live detection from webcam.")

            st.markdown("#### ⚠️ Areas for Improvement")
            improvements = current_analysis.get("areas_for_improvement", [])
            if improvements and is_detected:
                for imp in improvements:
                    st.markdown(f"- 🟠 {imp}")
            else:
                st.markdown("- 🟢 No major flaws identified.")

        with rcol2:
            st.markdown("#### 🎯 Actionable Coaching Adjustments")
            recs = current_analysis.get("coaching_recommendations", [])
            for r in recs:
                st.info(f"👉 {r}")

            st.caption(f"**Note:** {current_analysis.get('disclaimer', '')}")

    with tab3:
        st.markdown("### ⚙️ System Architecture & Biomechanical Pipeline")
        st.markdown(
            """
            **Cricket Vision AI** operates as a 100% real-time sports technology system:

            1. **Live Webcam Ingestion**:
               Video frames captured via WebRTC at 30 FPS.
            2. **MediaPipe Pose Estimation**:
               Extracts 33 high-precision body landmark coordinates in real-time.
            3. **Vector Angle Calculation**:
               Computes interior 3-point joint angles using pure NumPy vector mathematics.
            4. **Rule-Based Cricket Biomechanics Engine**:
               Evaluates knee flexion, torso spine tilt, base width, and head level against calibrated batting coaching rules.
            5. **Thread-Safe Live Dashboard**:
               Lock-protected state store continuously refreshes metrics without freezing or re-initializing the video stream.
            """
        )
        st.success("Designed as an educational sports technology prototype for batting stance awareness.")

# Invoke lower section fragment
is_playing_lower = bool(WEBRTC_AVAILABLE and 'webrtc_ctx' in locals() and webrtc_ctx and webrtc_ctx.state.playing)
render_live_lower_section(is_playing_lower)

# Invoke lower section fragment
is_playing_lower = bool(WEBRTC_AVAILABLE and 'webrtc_ctx' in locals() and webrtc_ctx and webrtc_ctx.state.playing)
render_live_lower_section(is_playing_lower)

