from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from agroshield.advice import advice_for
from agroshield.features import FEATURE_VERSION, decode_image, extract_features, preprocess

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "agroshield_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

st.set_page_config(page_title="AgroShield AI", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: float):
    del modified  # included in the cache key so retraining reloads the artifact
    payload = joblib.load(path)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("The model was trained with a different image feature version. Retrain it with the current project code.")
    return payload


def show_model_setup() -> None:
    st.warning("The disease-classification model has not been trained yet.")
    st.markdown("To enable predictions, download the training data and train the baseline model from the project folder:")
    st.code(
        "python scripts/download_plantvillage.py\n"
        "python scripts/train_model.py --data-dir data/raw\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.caption("The app will not invent or simulate a disease result. The upload and image-preprocessing flow is available, but prediction requires the trained model artifact.")


st.title("🌿 AgroShield AI")
st.subheader("Crop disease detection and treatment guidance")
st.write("Upload a clear crop or leaf image for a model-assisted first assessment. Results are predictions and should be confirmed with a local agricultural expert.")

with st.sidebar:
    st.header("Project status")
    if MODEL_PATH.exists():
        st.success("Trained model found")
        if METRICS_PATH.exists():
            try:
                metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                st.metric("Held-out benchmark accuracy", f"{metrics['accuracy']:.1%}")
                st.caption(f"Training evaluation: {metrics['n_test']:,} held-out images across {metrics['n_classes']} classes.")
            except (OSError, ValueError, KeyError):
                pass
    else:
        st.info("Model training required")
    st.caption("A benchmark score is not a guarantee of performance on field images.")

uploaded = st.file_uploader("Upload a crop or leaf photo", type=["jpg", "jpeg", "png", "webp", "bmp"])
crop = st.text_input("Crop (optional)", placeholder="For example: tomato")

if uploaded:
    try:
        image = decode_image(uploaded.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.image(image, caption=f"Uploaded image · {image.width} × {image.height}", use_container_width=True)
        prepared = preprocess(image)
        st.caption(f"Preprocessing complete: resized to {prepared.shape[1]} × {prepared.shape[0]} pixels and converted to RGB.")

    with right:
        if not MODEL_PATH.exists():
            show_model_setup()
        else:
            try:
                model_payload = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
                model = model_payload["model"]
                feature = extract_features(prepared).reshape(1, -1)
                scores = model.predict_proba(feature)[0]
                order = scores.argsort()[::-1][:3]
                labels = model_payload["labels"]
                winner = labels[int(order[0])]
                predicted_crop = winner.split("___", 1)[0].replace("_", " ")
                if crop and predicted_crop.casefold() not in crop.casefold():
                    st.warning(f"The model's top class is for {predicted_crop}. Check that the crop entered above is correct.")
                info = advice_for(winner)
                st.markdown("### Model result")
                st.markdown(f"**Probable condition:** {info['title']}")
                st.metric("Model score (uncalibrated)", f"{scores[order[0]]:.1%}")
                st.caption("This score is not a calibrated probability or a confirmed diagnosis.")
                st.info(info["summary"])
                st.markdown("#### General next steps")
                for action in info["actions"]:
                    st.markdown(f"- {action}")
                st.markdown("#### Other likely classes")
                top_results = [{"class": labels[int(i)], "model_score": round(float(scores[i]), 4)} for i in order]
                st.dataframe(top_results, hide_index=True, use_container_width=True)
                st.caption("Severity is not shown because this model is trained for disease classes, not severity levels.")
                report = {
                    "project": "AgroShield AI",
                    "crop_entered": crop or None,
                    "image_size": {"width": image.width, "height": image.height},
                    "preprocessed_size": list(prepared.shape[:2]),
                    "prediction": winner,
                    "model_score_uncalibrated": float(scores[order[0]]),
                    "top_three": top_results,
                    "severity": None,
                    "guidance": info["actions"],
                    "disclaimer": "Model prediction only; confirm with a local agricultural expert.",
                }
                st.download_button("Download result report (JSON)", json.dumps(report, indent=2), file_name="agroshield-result.json", mime="application/json")
            except Exception as exc:
                st.error(f"Could not load or run the model: {exc}")
                st.caption("If you just trained the model, restart Streamlit. Otherwise check artifacts/agroshield_model.joblib.")

st.divider()
st.caption("AgroShield AI is an academic prototype. Guidance is general information, not a pesticide prescription. Confirm diagnosis and treatment locally before taking action.")
