import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from xgboost import XGBClassifier

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
RESNET_FILENAME = "resnet50_expanded_dataset_finetuned.keras"
RESNET_MODEL_PATH = str(BASE_DIR / RESNET_FILENAME)
XGB_MODEL_PATH = str(BASE_DIR / "resnet50_xgboost_classifier.json")

HF_REPO_ID = "PS21ps2005/surveillance-resnet50"

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Dangerous Situation",
    "Isolated Situation",
    "Safe Situation",
]

st.set_page_config(
    page_title="Surveillance Detection: ResNet50 vs ResNet50+XGBoost",
    layout="wide",
)


# ==========================================
# MODEL LOADING (Cached)
# ==========================================
@st.cache_resource
def load_models():
    # Use local .keras file on laptop if present; otherwise auto-download from Hugging Face
    if os.path.exists(RESNET_MODEL_PATH):
        resnet_path = RESNET_MODEL_PATH
    else:
        resnet_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=RESNET_FILENAME,
        )

    # 1. Load Fine-tuned ResNet50
    resnet_model = load_model(
        resnet_path,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
        safe_mode=False,
    )

    # 2. Extract GlobalAveragePooling2D layer for XGBoost feature extraction
    gap_layer = next(
        layer
        for layer in resnet_model.layers
        if isinstance(layer, GlobalAveragePooling2D)
    )
    feature_extractor = Model(
        inputs=resnet_model.inputs,
        outputs=gap_layer.output,
    )

    # 3. Load XGBoost Classifier
    xgb_model = XGBClassifier()
    xgb_model.load_model(XGB_MODEL_PATH)

    return resnet_model, feature_extractor, xgb_model


# ==========================================
# PREPROCESSING & INFERENCE HELPERS
# ==========================================
def prepare_rgb_batch(rgb_images: list[np.ndarray]) -> np.ndarray:
    """Resizes RGB uint8 arrays to (N, 224, 224, 3) float32 in [0, 255]."""
    resized = [
        cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
        for img in rgb_images
    ]
    return np.array(resized, dtype=np.float32)


def run_both_models(
    rgb_images: list[np.ndarray], resnet_model, feature_extractor, xgb_model
):
    """Runs inference for both ResNet50 and ResNet50+XGBoost on a batch of RGB images."""
    batch = prepare_rgb_batch(rgb_images)

    # 1. Standalone ResNet50 predictions
    resnet_probs = resnet_model(batch, training=False).numpy()
    resnet_preds = np.argmax(resnet_probs, axis=1)

    # 2. Extract GAP features -> XGBoost predictions
    gap_features = feature_extractor(batch, training=False).numpy()
    xgb_probs = xgb_model.predict_proba(gap_features)
    xgb_preds = np.argmax(xgb_probs, axis=1)

    results = []
    for i in range(len(rgb_images)):
        r_idx = int(resnet_preds[i])
        x_idx = int(xgb_preds[i])
        results.append({
            "resnet_label": CLASS_NAMES[r_idx],
            "resnet_conf": float(resnet_probs[i][r_idx]) * 100,
            "resnet_probs": resnet_probs[i],
            "xgb_label": CLASS_NAMES[x_idx],
            "xgb_conf": float(xgb_probs[i][x_idx]) * 100,
            "xgb_probs": xgb_probs[i],
            "agreement": r_idx == x_idx,
        })
    return results


def extract_video_frames(video_bytes: bytes, interval_sec: float):
    """Extracts 1 frame every `interval_sec` seconds using direct frame seeking."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 30.0

    frame_step = max(1, int(round(fps * interval_sec)))
    extracted_frames = []
    timestamps = []
    frame_indices = []

    frame_idx = 0
    while True:
        if total_frames > 0 and frame_idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        extracted_frames.append(frame_rgb)
        timestamps.append(round(frame_idx / fps, 2))
        frame_indices.append(frame_idx)

        frame_idx += frame_step

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)
    return extracted_frames, timestamps, frame_indices, fps


# ==========================================
# STREAMLIT UI
# ==========================================
st.title("Surveillance Classification: ResNet50 vs. ResNet50 + XGBoost")
st.write(
    "Compare predictions from the fine-tuned **ResNet50** model and the hybrid"
    " **ResNet50 + XGBoost** classifier."
)

# Sidebar Controls
st.sidebar.header("Configuration")
input_mode = st.sidebar.radio("Select Input Mode", ["Image", "Video"])

with st.sidebar.expander("Edit Class Names"):
    c0 = st.text_input("Class 0", value=CLASS_NAMES[0])
    c1 = st.text_input("Class 1", value=CLASS_NAMES[1])
    c2 = st.text_input("Class 2", value=CLASS_NAMES[2])
    CLASS_NAMES = [c0, c1, c2]

try:
    with st.spinner("Loading ResNet50 and XGBoost models..."):
        resnet_model, feature_extractor, xgb_model = load_models()
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ------------------------------------------
# MODE 1: IMAGE INFERENCE
# ------------------------------------------
if input_mode == "Image":
    uploaded_images = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )

    if uploaded_images:
        rgb_list = []
        for file in uploaded_images:
            pil_img = Image.open(file).convert("RGB")
            rgb_list.append(np.array(pil_img))

        with st.spinner("Running inference on images..."):
            predictions = run_both_models(
                rgb_list, resnet_model, feature_extractor, xgb_model
            )

        for file, rgb_img, pred in zip(uploaded_images, rgb_list, predictions):
            st.markdown("---")
            col_img, col_res1, col_res2 = st.columns([1.2, 1, 1])

            with col_img:
                st.image(rgb_img, caption=file.name, use_container_width=True)

            with col_res1:
                st.subheader("ResNet50 (Fine-Tuned)")
                st.metric(
                    label="Predicted Class",
                    value=pred["resnet_label"],
                    delta=f"{pred['resnet_conf']:.2f}% confidence",
                )
                prob_df_r = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability (%)": np.round(pred["resnet_probs"] * 100, 2),
                })
                st.dataframe(prob_df_r, hide_index=True, use_container_width=True)

            with col_res2:
                st.subheader("ResNet50 + XGBoost")
                st.metric(
                    label="Predicted Class",
                    value=pred["xgb_label"],
                    delta=f"{pred['xgb_conf']:.2f}% confidence",
                )
                prob_df_x = pd.DataFrame({
                    "Class": CLASS_NAMES,
                    "Probability (%)": np.round(pred["xgb_probs"] * 100, 2),
                })
                st.dataframe(prob_df_x, hide_index=True, use_container_width=True)

            if pred["agreement"]:
                st.success(f"Both models agree: **{pred['resnet_label']}**")
            else:
                st.warning(
                    f"Model Disagreement — ResNet50 predicted **{pred['resnet_label']}**, "
                    f"while ResNet50+XGBoost predicted **{pred['xgb_label']}**."
                )

# ------------------------------------------
# MODE 2: MULTI-VIDEO FRAME EXTRACTION & INFERENCE
# ------------------------------------------
elif input_mode == "Video":
    interval_sec = st.sidebar.number_input(
        "Frame Extraction Interval (seconds)",
        min_value=1.0,
        max_value=300.0,
        value=30.0,
        step=5.0,
        help="Extract 1 frame every N seconds of video (default: 30s).",
    )

    uploaded_videos = st.file_uploader(
        "Upload one or more videos",
        type=["mp4", "avi", "mov", "mkv"],
        accept_multiple_files=True,
    )

    if uploaded_videos:
        all_summary_rows = []

        for video_file in uploaded_videos:
            st.markdown("---")
            st.header(f"Video: {video_file.name}")

            with st.spinner(
                f"Extracting 1 frame every {interval_sec}s from {video_file.name}..."
            ):
                frames, timestamps, frame_indices, fps = extract_video_frames(
                    video_file.read(), interval_sec
                )

            if not frames:
                st.warning(f"No frames could be extracted from {video_file.name}.")
                continue

            st.caption(
                f"Detected FPS: {fps:.2f} | Interval: {interval_sec}s | "
                f"Extracted Frames: {len(frames)}"
            )

            with st.spinner(f"Running both models on {len(frames)} frames..."):
                preds = run_both_models(
                    frames, resnet_model, feature_extractor, xgb_model
                )

            video_rows = []
            for f_idx, ts, pred in zip(frame_indices, timestamps, preds):
                row = {
                    "Video": video_file.name,
                    "Frame #": f_idx,
                    "Timestamp (s)": ts,
                    "ResNet50 Prediction": pred["resnet_label"],
                    "ResNet50 Conf (%)": round(pred["resnet_conf"], 2),
                    "ResNet+XGB Prediction": pred["xgb_label"],
                    "ResNet+XGB Conf (%)": round(pred["xgb_conf"], 2),
                    "Match": "Yes" if pred["agreement"] else "No",
                }
                video_rows.append(row)
                all_summary_rows.append(row)

            df_video = pd.DataFrame(video_rows)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Extracted Frames", len(frames))
            with c2:
                top_r = df_video["ResNet50 Prediction"].mode()[0]
                st.metric("ResNet50 Majority Class", top_r)
            with c3:
                top_x = df_video["ResNet+XGB Prediction"].mode()[0]
                st.metric("ResNet+XGB Majority Class", top_x)

            st.subheader("Frame-by-Frame Predictions Table")
            st.dataframe(df_video, hide_index=True, use_container_width=True)

            st.subheader("Extracted Frames & Model Predictions")
            cols_per_row = 4
            for start_idx in range(0, len(frames), cols_per_row):
                cols = st.columns(cols_per_row)
                for col_offset in range(cols_per_row):
                    idx = start_idx + col_offset
                    if idx >= len(frames):
                        break
                    with cols[col_offset]:
                        p = preds[idx]
                        st.image(
                            frames[idx],
                            caption=f"t = {timestamps[idx]}s (Frame {frame_indices[idx]})",
                            use_container_width=True,
                        )
                        st.markdown(
                            f"**ResNet50:** `{p['resnet_label']}` ({p['resnet_conf']:.1f}%)  \n"
                            f"**ResNet+XGB:** `{p['xgb_label']}` ({p['xgb_conf']:.1f}%)"
                        )

        if len(uploaded_videos) > 1 and all_summary_rows:
            st.markdown("---")
            st.header("Combined Multi-Video Summary")
            df_all = pd.DataFrame(all_summary_rows)
            st.dataframe(df_all, hide_index=True, use_container_width=True)

            csv_data = df_all.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download All Video Predictions (CSV)",
                data=csv_data,
                file_name="multi_video_surveillance_predictions.csv",
                mime="text/csv",
            )