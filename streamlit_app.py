# streamlit_app.py

import streamlit as st
import tempfile
import time
from pathlib import Path
from collections import OrderedDict

import pandas as pd
import numpy as np
import altair as alt

import cv2

import torch
from torchvision import transforms
from torchvision.models.alexnet import AlexNet

from keras.models import load_model as keras_load_model
from keras.layers import DepthwiseConv2D as KDepthwiseConv2D

# ── CONSTANTS ───────────────────────────────────────────────────────────────────

# These three are the classes you fine‐tuned in your notebook.
ENGAGEMENT_TYPES = ["not engaged", "engaged-negative", "engaged-positive"]

# Seven emotions for the Keras model (unchanged).
EMOTION_LABELS = [
    "Angry",       # index 0
    "Disgusted",   # index 1
    "Fearful",     # index 2
    "Happy",       # index 3
    "Neutral",     # index 4
    "Sad",         # index 5
    "Surprised"    # index 6
]

# The folder that contains both your PyTorch and Keras weights.
BASE_DIR = Path(__file__).parent.parent


# ── CUSTOM DEPTHWISECONV2D FOR .h5 DESERIALIZATION ──────────────────────────────

class CustomDepthwiseConv2D(KDepthwiseConv2D):
    """
    Subclass of Keras DepthwiseConv2D that strips out any 'groups' kwarg
    during deserialization. Allows loading older H5 models whose config
    includes {"groups":1}.
    """
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# ── SMART ENGAGEMENT MODEL LOADING WITH MULTIPLE STRATEGIES ────────────────────

@st.experimental_singleton
def load_engagement_model() -> torch.nn.Module:
    """
    1) Attempts to torch.load(...) the checkpoint.
    2) If it's already an nn.Module, just eval() it.
    3) Otherwise, grabs state_dict out of it, strips any "module." prefixes,
       instantiates AlexNet(num_classes=3), and loads the weights (strict=True
       or, failing that, strict=False).
    4) Runs a dummy inference to verify everything works.
    """
    model_path = BASE_DIR / "models" / "alexnet_best_finetuned.pth"
    if not model_path.exists():
        st.error(f"❌ Engagement model not found at: {model_path}")
        st.stop()

    try:
        checkpoint = torch.load(str(model_path), map_location=torch.device("cpu"))
    except Exception as e:
        st.error(f"❌ Failed to load checkpoint: {e}")
        st.stop()

    # If the checkpoint is already a full nn.Module, we can just eval() it.
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.eval()
        return _test_and_return_model(model)

    # Otherwise, we expect an OrderedDict or a dict containing a "state_dict" key.
    state_dict = None
    if isinstance(checkpoint, dict):
        # Look for common keys that might store the actual state_dict
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            # Maybe it's a bare state_dict already
            state_dict = checkpoint
    else:
        # If it's not a dict at all, assume it's already the raw state_dict
        state_dict = checkpoint

    # Strip any "module." prefixes that come from DataParallel training.
    if isinstance(state_dict, dict):
        cleaned = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            cleaned[new_key] = v
        state_dict = cleaned

    # Instantiate AlexNet with exactly 3 output classes (["not", "neg", "pos"]).
    model = AlexNet(num_classes=len(ENGAGEMENT_TYPES))

    # Try to load in strict mode first, then fallback to non‐strict if needed.
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e2:
            st.error(f"❌ Failed to load state_dict even with strict=False: {e2}")
            st.stop()

    model.eval()
    return _test_and_return_model(model)


def _test_and_return_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Run a single dummy inference on CPU to confirm the model signature matches.
    """
    try:
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = model(dummy)
            _ = torch.nn.functional.softmax(out, dim=1)[0]
        return model
    except Exception as e:
        st.error(f"❌ Model dummy test failed: {e}")
        st.stop()


# ── MULTIPLE PREPROCESSING STRATEGIES FOR ENGAGEMENT ───────────────────────────

def get_preprocessing_transforms():
    """
    Returns four different transforms, all resizing to 224×224 but normalizing
    differently, so you can choose the one that best matches your webcam lighting.
    """
    return {
        "imagenet": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        "simple": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ]),
        "no_norm": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]),
        "custom": transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }


def predict_engagement_robust(
    model: torch.nn.Module,
    roi_rgb: np.ndarray,
    transform_name: str = "imagenet",
    temperature: float = 1.5
):
    """
    Given a face‐crop (RGB, uint8, shape H×W×3):
      1) Apply exactly the chosen ImageNet‐style transform.
      2) Forward through AlexNet → logits (shape [1,3]).
      3) Divide logits by `temperature` and apply softmax → probs (length‐3).
      4) Label   = ENGAGEMENT_TYPES[argmax(probs)].
      5) raw_score = P(engaged-positive) − P(engaged-negative).
      6) Return (label, probs, used_transform_name, raw_score).

    If the exact named transform fails, it automatically falls back through
    each of the four preprocessing variants in `get_preprocessing_transforms()`. 
    """
    transforms_dict = get_preprocessing_transforms()

    # First attempt: use exactly transform_name
    try:
        transform = transforms_dict[transform_name]
        tensor = transform(roi_rgb).unsqueeze(0)  # shape [1,3,224,224]
        with torch.no_grad():
            logits = model(tensor)  # shape [1,3]
            scaled = logits / temperature
            probs = torch.nn.functional.softmax(scaled, dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        label = ENGAGEMENT_TYPES[idx]
        raw_score = float(probs[2] - probs[1])  # P(pos) – P(neg)
        return label, probs, transform_name, raw_score
    except Exception:
        pass

    # Fallback: try each transform until one works
    for name, transform in transforms_dict.items():
        try:
            tensor = transform(roi_rgb).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                scaled = logits / temperature
                probs = torch.nn.functional.softmax(scaled, dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            label = ENGAGEMENT_TYPES[idx]
            raw_score = float(probs[2] - probs[1])
            return label, probs, name, raw_score
        except Exception:
            continue

    # If all fail, return “not engaged” with raw_score=0
    return "not engaged", np.array([1.0, 0.0, 0.0]), "fallback", 0.0


# ── “ZOOM” HELPER FUNCTION ──────────────────────────────────────────────────────

def zoom_score(s: float) -> float:
    """
    Takes any mapped engagement score s ∈ [1,5]. 
    - If s < 4.0, leave it unchanged.
    - If s ∈ [4.0,5.0], linearly remap that interval onto [1.0,5.0].
      i.e. zoomed = 1 + 4*(s−4).  (So 4→1, 5→5, 4.5→3.)
    """
    if s < 4.0:
        return s
    return 1.0 + 4.0 * (s - 4.0)


# ── KERAS EMOTION MODEL LOADING (UNCHANGED) ───────────────────────────────────

@st.experimental_singleton
def load_keras_emotion_model():
    """
    Load the Keras .h5 file (best_model.h5) with the CustomDepthwiseConv2D subclass,
    so that older h5 configs including {"groups":1} deserialize correctly.
    """
    model_path = BASE_DIR / "models" / "best_model.h5"
    if not model_path.exists():
        st.error(f"❌ Emotion model not found at: {model_path}")
        st.stop()

    model = keras_load_model(
        str(model_path),
        custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D}
    )
    return model


@st.experimental_singleton
def get_face_cascade() -> cv2.CascadeClassifier:
    """
    Return a cached Haar Cascade for face detection (frontalface_default.xml).
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        st.error("❌ Failed to load Haar Cascade for face detection.")
        st.stop()
    return face_cascade


def preprocess_face_for_emotion(roi_bgr: np.ndarray, target_size=(96, 96)) -> np.ndarray:
    """
    Convert a BGR face‐crop → RGB, resize to `target_size`, normalize to [0,1],
    and add a batch dimension (shape becomes [1,H,W,3]). Used by Keras emotion model.
    """
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(roi_rgb, target_size, interpolation=cv2.INTER_CUBIC)
    face_norm = face_resized.astype("float32") / 255.0
    return np.expand_dims(face_norm, axis=0)


# ── STREAMLIT UI SETUP ─────────────────────────────────────────────────────────

st.set_page_config(page_title="Continuous Engagement & Emotion App", layout="wide")
st.title("Our app for the final project. Emotion detection and RT engagement")
st.markdown("""
**Overview**  
- Below, choose between the AlexNet‐based Engagement Model or a Keras Emotion Model.  
- You can either upload a video or run a live webcam.  
- For Engagement, we display a continuous **1–5** “Engagement Score”  
""")


# ── SINGLE‐IMAGE ENGAGEMENT SANITY CHECK ─────────────────────────────────────────

st.markdown("## 1) Single‐Image Engagement Sanity Check")
uploaded_single = st.file_uploader(
    "📸 Upload a JPG/PNG to see the continuous engagement score:", type=["jpg", "jpeg", "png"]
)

if uploaded_single is not None:
    # Read the uploaded image as a NumPy BGR array
    raw_bytes = np.asarray(bytearray(uploaded_single.read()), dtype="uint8")
    frame_bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        st.error("⚠️ Could not decode that image. Try another file.")
    else:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption="Uploaded Image", width=300)

        # Let user pick which preprocessing to try first (dropdown)
        chosen_transform = st.selectbox(
            "Choose preprocessing for this test:",
            ["imagenet", "simple", "no_norm", "custom"],
            index=0
        )
        model = load_engagement_model()
        label, probs, used_t, raw_score = predict_engagement_robust(
            model, frame_rgb,
            transform_name=chosen_transform,
            temperature=1.5
        )

        # Map raw_score ∈ [−1,+1] → mapped_score ∈ [1,5]
        mapped_score = (raw_score + 1.0) * 2.0 + 1.0
        # Then “zoom” any mapped_score ≥ 4.0 down onto the full [1..5] band
        zoomed = zoom_score(mapped_score)

        st.write(f"▶️ Used transform: **{used_t}**")
        st.markdown(f"### 🔸 Mapped Engagement Score (1–5): **{mapped_score:.3f}**")
        st.markdown(f"### 🔹 Zoomed Engagement Score (1–5): **{zoomed:.3f}**")

st.markdown("---")


# ── SIDEBAR: Configuration ───────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    model_choice = st.radio(
        "Select a model:",
        ("Engagement Model (AlexNet)", "Emotion Model (Keras)")
    )

    input_mode = st.radio(
        "Choose input mode:",
        ("Upload Video File", "Use Webcam Live")
    )

    # If Engagement was chosen, let them pick the preprocessing variant
    if model_choice.startswith("Engagement"):
        st.subheader("🎨 Preprocessing Method")
        preprocess_method = st.selectbox(
            "Face‐crop preprocessing:",
            ["imagenet", "simple", "no_norm", "custom"]
        )

    # If the user wants to upload a video, let them pick a frame skip
    if input_mode == "Upload Video File":
        st.subheader("🎬 Video Settings")
        frame_skip = st.slider(
            "Process every N frames:", 1, 10, 1,
            help="Higher = skip more frames (faster)."
        )

    # If the user wants to run a live webcam, let them choose how many frames
    if input_mode == "Use Webcam Live":
        st.subheader("📷 Webcam Settings")
        webcam_frames = st.slider(
            "Number of frames to capture:", 50, 500, 200, 50
        )


# ── ENGAGEMENT: VIDEO PROCESSING FUNCTION ───────────────────────────────────────

def process_video_engagement(path: str, frame_skip: int = 1) -> pd.DataFrame:
    """
    Run engagement inference on an uploaded video file:
      - Every `frame_skip` frames, detect largest face via Haar cascade.
      - Crop (with a small pad), convert to RGB, then call predict_engagement_robust().
      - Collect: ['time_sec', 'engagement_lbl', 'transform_used', 'raw_score'].
      - Finally compute `mapped_score` ∈ [1,5], and return that column as well.
    """
    model = load_engagement_model()
    face_cascade = get_face_cascade()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("❌ Could not open the video file for engagement processing.")
        return pd.DataFrame()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times, lbls, methods, raw_scores = [], [], [], []
    progress_bar = st.progress(0.0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
            )

            if len(faces) > 0:
                # Take the largest detected face
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                roi_bgr = frame[y1:y2, x1:x2]
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                label, probs, used_t, raw_score = predict_engagement_robust(
                    model, roi_rgb,
                    transform_name=preprocess_method,
                    temperature=1.5
                )
            else:
                label = "no face"
                used_t = "—"
                raw_score = 0.0

            times.append(frame_idx / fps)
            lbls.append(label)
            methods.append(used_t)
            raw_scores.append(raw_score)

        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    progress_bar.empty()

    # Build a DataFrame and also compute the mapped 1–5 score
    df_out = pd.DataFrame({
        "time_sec": times,
        "engagement_lbl": lbls,
        "transform_used": methods,
        "raw_score": raw_scores
    })
    # Map raw_score ∈ [−1,+1] → mapped_score ∈ [1,5]
    df_out["engagement_score"] = df_out["raw_score"].apply(lambda r: (r + 1.0) * 2.0 + 1.0)
    # Then add a zoomed version
    df_out["zoomed_score"] = df_out["engagement_score"].apply(zoom_score)

    return df_out


# ── ENGAGEMENT: WEBCAM PROCESSING FUNCTION ─────────────────────────────────────

def process_camera_engagement(num_frames: int = 200, fps_delay: float = 0.03) -> pd.DataFrame:
    """
    Run live‐webcam engagement inference for `num_frames` frames:
      - Each frame: detect largest face, call predict_engagement_robust().
      - Draw a bounding box + “Score: X.XX” on the video preview.
      - Keep a running line‐chart of Engagement Score ∈ [1,5], but actually plot “zoomed_score” ∈ [1,5].
      - Return a DataFrame with ['time_sec', 'engagement_lbl', 'transform_used', 'raw_score', 'engagement_score', 'zoomed_score'].
    """
    model = load_engagement_model()
    face_cascade = get_face_cascade()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam for engagement processing.")
        return pd.DataFrame()

    image_ph = st.empty()
    face_ph = st.empty()
    chart_ph = st.empty()
    status_ph = st.empty()

    times, lbls, methods, raw_scores = [], [], [], []
    counts = {label: 0 for label in ENGAGEMENT_TYPES}
    confidences_list = []
    start_time = time.time()

    with st.spinner("🚀 Starting live engagement detection..."):
        for idx in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                status_ph.error("⚠️ Lost camera connection")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
            )

            engagement_label = "no face"
            used_transform = "—"
            raw_score = 0.0

            if len(faces) > 0:
                # Take the largest face
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                pad = 20
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                roi_bgr = frame[y1:y2, x1:x2]
                roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

                engagement_label, top_confs, used_transform, raw_score = predict_engagement_robust(
                    model, roi_rgb,
                    transform_name=preprocess_method,
                    temperature=1.5
                )
                counts[engagement_label] += 1
                confidences_list.append(top_confs)

                # Draw bounding box & “Score: X.XX” on frame (mapped)
                mapped_score = (raw_score + 1.0) * 2.0 + 1.0
                color = (0, 255, 0) if engagement_label != "not engaged" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"Score: {mapped_score:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA
                )

                # Show just the face crop thumbnail (for debugging / visual confirmation)
                face_rgb_display = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                face_ph.image(
                    face_rgb_display,
                    channels="RGB",
                    caption=f"Face Crop ({used_transform})",
                    width=200
                )

            times.append(time.time() - start_time)
            lbls.append(engagement_label)
            methods.append(used_transform)
            raw_scores.append(raw_score)

            # Display webcam frame
            rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_ph.image(
                rgb_display,
                channels="RGB",
                caption=f"Webcam Frame {idx+1}/{num_frames}"
            )

            # Show running counts of discrete labels
            status_ph.text(
                f"👥 Faces: {len(faces)}  | "
                f"😐 Not: {counts['not engaged']}  | "
                f"😞 Neg: {counts['engaged-negative']}  | "
                f"😊 Pos: {counts['engaged-positive']}"
            )

            # Live “Engagement Score” line chart once we have >5 valid frames
            # We build a small DataFrame of time and **zoomed** score
            valid_data = []
            for t, r, lbl in zip(times, raw_scores, lbls):
                if lbl in ENGAGEMENT_TYPES:
                    mapped = (r + 1.0) * 2.0 + 1.0
                    zoomed = zoom_score(mapped)
                    valid_data.append((t, zoomed))

            if len(valid_data) > 5:
                chart_times, chart_scores = zip(*valid_data)
                chart_df = pd.DataFrame({
                    "time_sec": chart_times,
                    "zoomed_score": chart_scores
                })
                live_chart = (
                    alt.Chart(chart_df)
                    .mark_line(point=True, color="#ff4b4b")
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (seconds)"),
                        y=alt.Y(
                            "zoomed_score:Q",
                            title="Zoomed Engagement Score (1–5)",
                            scale=alt.Scale(domain=[1, 5])
                        )
                    )
                    .properties(height=250)
                )
                chart_ph.altair_chart(live_chart, use_container_width=True)

            time.sleep(fps_delay)

    cap.release()
    status_ph.success("✅ Engagement detection complete!")

    # Final summary: show average raw confidences internally for debugging (hidden in normal use)
    if confidences_list:
        avg_conf = np.mean(confidences_list, axis=0)
        st.markdown("### 📊 (Internal) Final Average Softmax Confidences")
        st.write("_(Not shown in normal UI; debugging only.)_")
        st.write(f"• not engaged: {avg_conf[0]:.3f}  |  neg: {avg_conf[1]:.3f}  |  pos: {avg_conf[2]:.3f}")

        # Show a histogram of **zoomed** engagement scores
        hist_scores = [(zoom_score((r + 1.0) * 2.0 + 1.0)) for r in raw_scores]
        hist_df = pd.DataFrame({"zoomed_score": hist_scores})
        hist = (
            alt.Chart(hist_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "zoomed_score:Q",
                    bin=alt.Bin(maxbins=30),
                    title="Zoomed Score (1–5)"
                ),
                y=alt.Y("count()", title="Frame Count")
            )
            .properties(height=300)
        )
        st.altair_chart(hist, use_container_width=True)

    # Return a DataFrame that includes raw_score, mapped 1..5, and zoomed_score
    df_out = pd.DataFrame({
        "time_sec": times,
        "engagement_lbl": lbls,
        "transform_used": methods,
        "raw_score": raw_scores
    })
    df_out["engagement_score"] = df_out["raw_score"].apply(lambda r: (r + 1.0) * 2.0 + 1.0)
    df_out["zoomed_score"] = df_out["engagement_score"].apply(zoom_score)
    return df_out


# ── EMOTION: VIDEO PROCESSING FUNCTION (UNCHANGED) ─────────────────────────────

def process_video_emotion(path: str, frame_skip: int = 1) -> pd.DataFrame:
    """
    Run the Keras emotion model on an uploaded video file:
    - Every `frame_skip` frames, detect largest face, crop → preprocess → predict.
    - Return ['time_sec', 'emotion_idx', 'emotion_lbl'].
    """
    model = load_keras_emotion_model()
    face_cascade = get_face_cascade()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        st.error("❌ Could not open the video file for emotion processing.")
        return pd.DataFrame()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    times, idxs, lbls = [], [], []
    progress_bar = st.progress(0.0)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                roi_bgr = frame[y:y + h, x:x + w]
                face_input = preprocess_face_for_emotion(roi_bgr, target_size=(96, 96))
                preds = model.predict(face_input, verbose=0)
                emotion_index = int(np.argmax(preds[0]))
                emotion_label = EMOTION_LABELS[emotion_index]
                times.append(frame_idx / fps)
                idxs.append(emotion_index)
                lbls.append(emotion_label)

        frame_idx += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    progress_bar.empty()

    return pd.DataFrame({
        "time_sec": times,
        "emotion_idx": idxs,
        "emotion_lbl": lbls
    })


# ── EMOTION: WEBCAM PROCESSING FUNCTION (UNCHANGED) ────────────────────────────

def process_camera_emotion(num_frames: int = 200, fps_delay: float = 0.03) -> pd.DataFrame:
    """
    Live webcam-based emotion capture:
    - For each of `num_frames`, detect face, run Keras model → emotion label.
    - Draw bounding box + emotion label on webcam feed.
    - Return ['time_sec', 'emotion_idx', 'emotion_lbl'].
    """
    model = load_keras_emotion_model()
    face_cascade = get_face_cascade()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam for emotion processing.")
        return pd.DataFrame()

    image_ph = st.empty()
    status_ph = st.empty()

    times, idxs, lbls = [], [], []
    start_time = time.time()
    frame_idx = 0

    with st.spinner("🚀 Starting emotion webcam inference..."):
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                status_ph.error("⚠️ Lost camera frame during emotion run.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                roi_bgr = frame[y:y + h, x:x + w]
                face_input = preprocess_face_for_emotion(roi_bgr, target_size=(96, 96))
                preds = model.predict(face_input, verbose=0)
                emotion_index = int(np.argmax(preds[0]))
                emotion_label = EMOTION_LABELS[emotion_index]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    emotion_label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                elapsed = time.time() - start_time
                times.append(elapsed)
                idxs.append(emotion_index)
                lbls.append(emotion_label)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_ph.image(img_rgb, channels="RGB")
            status_ph.text(f"Frame {frame_idx+1}/{num_frames} | Recognized faces: {len(times)}")

            frame_idx += 1
            time.sleep(fps_delay)

    cap.release()
    status_ph.success("✅ Emotion webcam run complete.")

    return pd.DataFrame({
        "time_sec": times,
        "emotion_idx": idxs,
        "emotion_lbl": lbls
    })


# ── MAIN WORKFLOW ───────────────────────────────────────────────────────────────

st.markdown("---")

if input_mode == "Upload Video File":
    st.subheader("📥 Upload a Video File")
    uploaded_file = st.file_uploader(
        "Choose a video (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
            tmpf.write(uploaded_file.read())
            tmp_path = tmpf.name

        st.info(f"🔄 Processing video with {model_choice}…")

        if model_choice.startswith("Engagement"):
            df = process_video_engagement(tmp_path, frame_skip=frame_skip)
            if not df.empty:
                # Filter out “no face” rows
                valid_df = df[df["engagement_lbl"].isin(ENGAGEMENT_TYPES)]

                st.subheader("📈 Engagement Over Time (Video)")
                if not valid_df.empty:
                    chart = (
                        alt.Chart(valid_df)
                        .mark_line(point=True, color="#ff4b4b")
                        .encode(
                            x=alt.X("time_sec:Q", title="Time (seconds)"),
                            y=alt.Y("zoomed_score:Q", title="Zoomed Engagement Score (1–5)",
                                    scale=alt.Scale(domain=[1, 5])),
                            tooltip=["time_sec", "zoomed_score"]
                        )
                        .properties(height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)

                st.subheader("📊 Final Engagement Distribution (Video)")
                counts = valid_df["engagement_lbl"].value_counts().reindex(ENGAGEMENT_TYPES, fill_value=0)
                df_counts = pd.DataFrame({
                    "Label": counts.index,
                    "Count": counts.values
                })
                bar = (
                    alt.Chart(df_counts)
                    .mark_bar()
                    .encode(
                        x=alt.X("Label:N", title="Engagement Label"),
                        y=alt.Y("Count:Q", title="Frame Count"),
                        color=alt.Color("Label:N", scale=alt.Scale(domain=ENGAGEMENT_TYPES))
                    )
                    .properties(height=300)
                )
                st.altair_chart(bar, use_container_width=True)

                total_valid = len(valid_df)
                engaged_count = counts["engaged-negative"] + counts["engaged-positive"]
                pct = (engaged_count / max(total_valid, 1) * 100) if total_valid > 0 else 0
                st.markdown(f"**% of frames ever ‘engaged’ (neg+pos): {pct:.1f}%**")

                st.markdown("### Engagement Score Histogram (Video)")
                hist_df = pd.DataFrame({"zoomed_score": valid_df["zoomed_score"].values})
                hist = (
                    alt.Chart(hist_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("zoomed_score:Q", bin=alt.Bin(maxbins=30), title="Zoomed Score (1–5)"),
                        y=alt.Y("count()", title="Frame Count")
                    )
                    .properties(height=300)
                )
                st.altair_chart(hist, use_container_width=True)

                st.download_button(
                    "📥 Download Engagement (Video) Results",
                    df.to_csv(index=False),
                    "engagement_video_results.csv",
                    "text/csv"
                )

        else:  # Emotion Model
            df = process_video_emotion(tmp_path, frame_skip=frame_skip)
            if not df.empty:
                st.subheader("⏱️ Emotion Over Time (Video)")
                line_chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color="#ff7f0e")
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (seconds)"),
                        y=alt.Y("emotion_idx:Q", title="Emotion Index"),
                        color=alt.Color("emotion_lbl:N"),
                        tooltip=["time_sec", "emotion_lbl"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(line_chart, use_container_width=True)

                st.subheader("📊 Final Emotion Counts (Video)")
                counts = df["emotion_lbl"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
                df_counts = pd.DataFrame({
                    "Emotion": counts.index,
                    "Count": counts.values
                })
                bar_chart = (
                    alt.Chart(df_counts)
                    .mark_bar()
                    .encode(
                        x="Emotion",
                        y="Count",
                        color=alt.Color("Emotion", legend=None),
                        tooltip=["Count"]
                    )
                    .properties(height=350)
                )
                st.altair_chart(bar_chart, use_container_width=True)

                total = len(df)
                distribution = (df["emotion_lbl"].value_counts() / total * 100).round(1)
                most_freq = distribution.idxmax()
                most_pct = distribution.max()
                st.markdown("**Emotion Distribution (%)**")
                dist_df = pd.DataFrame({
                    "Emotion": distribution.index,
                    "Percent (%)": distribution.values
                })
                st.table(dist_df)
                st.markdown(f"**Most Frequent Emotion:** {most_freq} ({most_pct:.1f}%)")

                st.download_button(
                    "📥 Download Emotion (Video) Results",
                    df.to_csv(index=False),
                    "emotion_video_results.csv",
                    "text/csv"
                )

elif input_mode == "Use Webcam Live":
    st.subheader("📷 Live Webcam Analysis")
    if st.button("▶️ Start Webcam"):
        if model_choice.startswith("Engagement"):
            df = process_camera_engagement(num_frames=webcam_frames, fps_delay=0.03)
            if not df.empty:
                valid_df = df[df["engagement_lbl"].isin(ENGAGEMENT_TYPES)]

                if not valid_df.empty:
                    st.subheader("📈 Engagement Over Time (Webcam)")
                    timeline = (
                        alt.Chart(valid_df)
                        .mark_line(point=True, color="#ff4b4b")
                        .encode(
                            x=alt.X("time_sec:Q", title="Time (seconds)"),
                            y=alt.Y("zoomed_score:Q", title="Zoomed Engagement Score (1–5)",
                                    scale=alt.Scale(domain=[1, 5])),
                            tooltip=["time_sec", "zoomed_score"]
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(timeline, use_container_width=True)

                    st.subheader("📊 Summary Stats (Webcam)")
                    counts = valid_df["engagement_lbl"].value_counts().reindex(ENGAGEMENT_TYPES, fill_value=0)
                    for lbl in ENGAGEMENT_TYPES:
                        c = counts.get(lbl, 0)
                        pct = (c / len(valid_df) * 100) if len(valid_df) > 0 else 0
                        st.metric(lbl.title(), f"{c} frames ({pct:.1f}%)")

                    st.markdown("### Engagement Score Histogram (Webcam)")
                    hist_df = pd.DataFrame({"zoomed_score": valid_df["zoomed_score"].values})
                    hist = (
                        alt.Chart(hist_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("zoomed_score:Q", bin=alt.Bin(maxbins=30), title="Zoomed Score (1–5)"),
                            y=alt.Y("count()", title="Frame Count")
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(hist, use_container_width=True)

                st.download_button(
                    "📥 Download Engagement (Webcam) Results",
                    df.to_csv(index=False),
                    "engagement_webcam_results.csv",
                    "text/csv"
                )

        else:  # Emotion Model
            df = process_camera_emotion(num_frames=webcam_frames, fps_delay=0.03)
            if not df.empty:
                st.subheader("😊 Emotion Over Time (Webcam)")
                line_chart = (
                    alt.Chart(df)
                    .mark_line(point=True, color="#ff7f0e")
                    .encode(
                        x=alt.X("time_sec:Q", title="Time (seconds)"),
                        y=alt.Y("emotion_idx:Q", title="Emotion Index"),
                        tooltip=["time_sec", "emotion_lbl"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(line_chart, use_container_width=True)

                st.subheader("📊 Final Emotion Distribution (Webcam)")
                counts = df["emotion_lbl"].value_counts().reindex(EMOTION_LABELS, fill_value=0)
                dist_df = pd.DataFrame({
                    "Emotion": counts.index,
                    "Count": counts.values,
                    "Percent (%)": (counts.values / len(df) * 100).round(1)
                })
                bar_chart = (
                    alt.Chart(dist_df)
                    .mark_bar()
                    .encode(
                        x="Emotion",
                        y="Count",
                        color=alt.Color("Emotion", legend=None),
                        tooltip=["Count", "Percent (%)"]
                    )
                    .properties(height=300)
                )
                st.altair_chart(bar_chart, use_container_width=True)

                most_common = counts.idxmax()
                st.success(f"🎯 Most common emotion: **{most_common}** ({counts.max()} frames)")

                st.download_button(
                    "📥 Download Emotion (Webcam) Results",
                    df.to_csv(index=False),
                    "emotion_webcam_results.csv",
                    "text/csv"
                )
                with st.expander("Show raw webcam emotion data"):
                    st.dataframe(df)
