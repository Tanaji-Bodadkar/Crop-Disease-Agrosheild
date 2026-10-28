from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from agroshield.advice import advice_for
from agroshield.auth import authenticate_user, register_user
from agroshield.features import FEATURE_VERSION, decode_image, extract_features, preprocess

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "agroshield_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

st.set_page_config(page_title="AgroShield AI", page_icon="🌱", layout="wide", initial_sidebar_state="expanded")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
        :root { --ag-green:#176b4d; --ag-deep:#113f32; --ag-ink:#162b25; --ag-muted:#63756d; --ag-line:#e2ebe6; --ag-mint:#e8f4ed; }
        html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
        .stApp { background:linear-gradient(180deg,#f6faf7 0%,#ffffff 46%); color:var(--ag-ink); }
        [data-testid="stSidebar"] { background:#f3f8f4; border-right:1px solid var(--ag-line); }
        [data-testid="stSidebar"] > div:first-child { padding-top:1.5rem; }
        .block-container { max-width:1240px; padding-top:2.1rem; padding-bottom:3rem; }
        h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ag-deep); letter-spacing:-.035em; }
        h1 { font-size:2.35rem !important; font-weight:800 !important; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--ag-line); border-radius:16px; padding:16px 18px; box-shadow:0 5px 18px rgba(17,63,50,.035); }
        [data-testid="stMetricLabel"] { color:var(--ag-muted); }
        [data-testid="stForm"] { background:#fff; border:1px solid var(--ag-line); border-radius:20px; padding:1.25rem; box-shadow:0 12px 32px rgba(17,63,50,.05); }
        [data-testid="stFileUploader"] { background:#f7fbf8; border:1px dashed #a8c8b6; border-radius:16px; padding:.45rem; }
        div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background:var(--ag-green); border-color:var(--ag-green); border-radius:10px; font-weight:700; min-height:2.8rem; }
        .brand-lockup { display:flex; align-items:center; gap:12px; margin:0 0 2rem; }
        .brand-mark { width:44px;height:44px;border-radius:14px;background:var(--ag-deep);color:white;display:flex;align-items:center;justify-content:center;font-size:23px; }
        .brand-name { font-family:'Manrope',sans-serif;font-weight:800;font-size:18px;line-height:1.15;color:var(--ag-deep); }
        .brand-tag { color:var(--ag-muted);font-size:11px;letter-spacing:.08em;text-transform:uppercase; }
        .eyebrow { color:var(--ag-green);font-size:.76rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;margin-bottom:.4rem; }
        .muted { color:var(--ag-muted); }
        .hero-panel { background:linear-gradient(125deg,#123f32,#1c7955); color:white; border-radius:22px; padding:24px 28px; margin:1rem 0 1.4rem; }
        .hero-panel h3 { color:white !important; margin:.25rem 0 .45rem; }
        .hero-panel p { color:#d6ebe0; margin:0; }
        .section-title { font-family:'Manrope',sans-serif;font-size:1.15rem;font-weight:800;color:var(--ag-deep); }
        .result-label { display:inline-block;background:var(--ag-mint);color:var(--ag-deep);padding:5px 10px;border-radius:99px;font-size:.8rem;font-weight:700; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">🌱</div><div><div class="brand-name">AgroShield AI</div><div class="brand-tag">Crop health workspace</div></div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model(path: str, modified: float):
    del modified
    payload = joblib.load(path)
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("The model was trained with an incompatible image feature version. Retrain it with this project code.")
    return payload


def render_auth() -> None:
    brand()
    left, right = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
    with left:
        st.markdown('<div class="eyebrow">A clearer first look at crop health</div>', unsafe_allow_html=True)
        st.title("Make every field observation count.")
        st.markdown(
            "Upload a leaf image, add what you are seeing, and review a model-assisted assessment with practical next steps.",
            help="Predictions are educational and should be confirmed by an agricultural expert.",
        )
        st.markdown(
            '<div class="hero-panel"><div class="eyebrow" style="color:#b7e3c9">Your field workspace</div><h3>Observe · Assess · Act carefully</h3><p>Keep image review and field notes together in one simple workspace.</p></div>',
            unsafe_allow_html=True,
        )
        a, b, c = st.columns(3)
        a.metric("Supported classes", "39")
        b.metric("Test accuracy", "89.1%")
        c.metric("Advice style", "Cautious")
        st.caption("Benchmark results are from a held-out PlantVillage image split and do not guarantee field performance.")

    with right:
        st.subheader("Welcome")
        st.caption("Sign in to your workspace or create an account to get started.")
        login_tab, signup_tab = st.tabs(["Sign in", "Create account"])
        with login_tab:
            with st.form("sign_in_form"):
                email = st.text_input("Email address", placeholder="you@example.com", autocomplete="email")
                password = st.text_input("Password", type="password", autocomplete="current-password")
                submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if submitted:
                profile = authenticate_user(email, password)
                if profile:
                    st.session_state["agro_user"] = profile
                    st.session_state.pop("assessment_result", None)
                    st.rerun()
                st.error("We couldn't sign you in with those details. Check your email and password, or create an account.")

        with signup_tab:
            with st.form("create_account_form"):
                name = st.text_input("Full name", placeholder="Your name", autocomplete="name")
                email = st.text_input("Email address", placeholder="you@example.com", autocomplete="email", key="signup_email")
                password = st.text_input("Create password", type="password", autocomplete="new-password", key="signup_password")
                confirm = st.text_input("Confirm password", type="password", autocomplete="new-password", key="signup_confirm")
                st.caption("Use at least 10 characters. Your password is stored as a salted hash.")
                submitted = st.form_submit_button("Create my workspace", type="primary", use_container_width=True)
            if submitted:
                if password != confirm:
                    st.error("The passwords do not match. Please try again.")
                else:
                    try:
                        profile = register_user(name, email, password)
                    except ValueError as exc:
                        st.error(str(exc))
                    except OSError:
                        st.error("We couldn't create the account database. Check that the app can write to its data folder.")
                    else:
                        st.session_state["agro_user"] = profile
                        st.session_state.pop("assessment_result", None)
                        st.rerun()
    st.divider()
    st.caption("AgroShield AI is an academic prototype. It does not provide a confirmed diagnosis or pesticide prescription.")


def render_sidebar(profile: dict[str, str]) -> str:
    with st.sidebar:
        brand()
        st.markdown(f"**{profile['full_name']}**")
        st.caption(profile["email"])
        if st.button("Sign out", use_container_width=True):
            for key in ("agro_user", "assessment_result"):
                st.session_state.pop(key, None)
            st.session_state["upload_form_version"] = st.session_state.get("upload_form_version", 0) + 1
            st.rerun()
        st.divider()
        page = st.radio("Workspace", ["New assessment", "How it works", "Account"], label_visibility="collapsed")
        st.divider()
        if MODEL_PATH.exists():
            st.success("AI model ready")
            if METRICS_PATH.exists():
                try:
                    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                    st.caption(f"Held-out benchmark: {metrics['accuracy']:.1%} accuracy")
                except (OSError, ValueError, KeyError):
                    pass
        else:
            st.warning("Model training required")
        st.caption("Images and notes are processed for this session and are not saved as a case history.")
    return page


def show_model_setup() -> None:
    st.warning("The trained disease-classification model is not available in this deployment.")
    st.markdown("To enable predictions, install the project dependencies, download the dataset, and train the model:")
    st.code(
        "python scripts/download_plantvillage.py\n"
        "python scripts/train_model.py --data-dir data/raw",
        language="powershell",
    )
    st.caption("The assessment form is available, but AgroShield will not invent a prediction when the model file is missing.")


def analyze_upload(uploaded, crop: str, description: str) -> dict:
    raw = uploaded.getvalue()
    image = decode_image(raw)
    result = {
        "crop": crop.strip() or "Not specified",
        "description": description.strip(),
        "image": image,
        "width": image.width,
        "height": image.height,
    }
    if not MODEL_PATH.exists():
        return result

    model_payload = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    prepared = preprocess(image)
    feature = extract_features(prepared).reshape(1, -1)
    scores = model_payload["model"].predict_proba(feature)[0]
    order = scores.argsort()[::-1][:3]
    labels = model_payload["labels"]
    winner = labels[int(order[0])]
    predicted_crop = winner.split("___", 1)[0].replace("_", " ")
    result.update(
        {
            "prepared": prepared,
            "prediction": winner,
            "predicted_crop": predicted_crop,
            "score": float(scores[order[0]]),
            "top_results": [
                {"class": labels[int(i)].replace("___", " — ").replace("_", " "), "score": round(float(scores[i]), 4)}
                for i in order
            ],
            "advice": advice_for(winner),
        }
    )
    return result


def render_assessment(profile: dict[str, str]) -> None:
    st.markdown('<div class="eyebrow">Field assessment</div>', unsafe_allow_html=True)
    st.title(f"Good to see you, {profile['full_name'].split()[0]}.")
    st.markdown("Add a clear leaf photo and a few field notes to start an assessment.")
    stats = st.columns(3)
    stats[0].metric("Supported classes", "39")
    stats[1].metric("Model status", "Ready" if MODEL_PATH.exists() else "Not trained")
    stats[2].metric("Data handling", "Session only")

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="section-title">New crop assessment</div>', unsafe_allow_html=True)
        st.caption("Photos are checked in memory and are not saved as a case history.")
        form_version = st.session_state.get("upload_form_version", 0)
        with st.form(f"assessment_form_{form_version}", clear_on_submit=False):
            uploaded = st.file_uploader(
                "Leaf or crop image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                help="Use a clear, well-lit image focused on the affected leaf. Maximum 10 MB.",
                key=f"assessment_image_{form_version}",
            )
            crop = st.text_input("Crop type", placeholder="e.g. tomato, potato, corn")
            description = st.text_area(
                "Describe what you observed",
                placeholder="When did the symptoms start? What changed in the leaves? Are nearby plants affected?",
                height=120,
                max_chars=1200,
            )
            submitted = st.form_submit_button("Analyze image", type="primary", use_container_width=True)
        if submitted:
            if not uploaded:
                st.error("Upload a leaf image to start the assessment.")
            else:
                try:
                    with st.spinner("Reviewing the image…"):
                        st.session_state["assessment_result"] = analyze_upload(uploaded, crop, description)
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"The assessment could not be completed: {exc}")
                    st.caption("Check that the trained model matches this project version, then restart the app.")

    with right:
        st.markdown('<div class="section-title">A useful photo makes a better assessment</div>', unsafe_allow_html=True)
        st.markdown(
            "- Photograph one affected leaf clearly.\n"
            "- Use natural light and keep the leaf in focus.\n"
            "- Include the crop name and when symptoms began.\n"
            "- Note whether other plants show similar changes."
        )
        st.info("AgroShield gives a model-assisted first look. Confirm disease and treatment decisions with a local agricultural expert.")

    result = st.session_state.get("assessment_result")
    if not result:
        return
    st.divider()
    st.markdown('<div class="eyebrow">Assessment summary</div>', unsafe_allow_html=True)
    result_left, result_right = st.columns([0.8, 1.2], gap="large")
    with result_left:
        st.image(result["image"], caption=f"Uploaded image · {result['width']} × {result['height']}", use_container_width=True)
        st.caption(f"Crop entered: {result['crop']}")
    with result_right:
        st.markdown('<div class="section-title">Your field notes</div>', unsafe_allow_html=True)
        st.write(result["description"] or "No symptom description was provided.")
        if "prediction" not in result:
            show_model_setup()
        else:
            if result["crop"] != "Not specified" and result["predicted_crop"].casefold() not in result["crop"].casefold():
                st.warning(f"The top class is for {result['predicted_crop']}. Check that the crop type entered above is correct.")
            advice = result["advice"]
            st.markdown('<div class="section-title">Model result</div>', unsafe_allow_html=True)
            st.markdown(f'<span class="result-label">{advice["title"]}</span>', unsafe_allow_html=True)
            st.metric("Model score (uncalibrated)", f"{result['score']:.1%}")
            st.caption("This score is not a calibrated probability or a confirmed diagnosis.")
            st.info(advice["summary"])
            st.markdown("**Suggested next steps**")
            for action in advice["actions"]:
                st.markdown(f"- {action}")
            st.markdown("**Other likely classes**")
            st.dataframe(result["top_results"], hide_index=True, use_container_width=True)
            report = {
                "project": "AgroShield AI",
                "crop_entered": result["crop"],
                "field_description": result["description"],
                "image_size": {"width": result["width"], "height": result["height"]},
                "prediction": result["prediction"],
                "model_score_uncalibrated": result["score"],
                "top_three": result["top_results"],
                "guidance": advice["actions"],
                "disclaimer": "Model prediction only; confirm with a local agricultural expert.",
            }
            st.download_button(
                "Download assessment summary",
                json.dumps(report, indent=2),
                file_name="agroshield-assessment.json",
                mime="application/json",
            )


def render_about(profile: dict[str, str], page: str) -> None:
    if page == "How it works":
        st.markdown('<div class="eyebrow">About this prototype</div>', unsafe_allow_html=True)
        st.title("How AgroShield works")
        st.markdown("The workspace validates and resizes your image, extracts visual features, and passes them to a Random Forest model trained on labeled PlantVillage images.")
        st.code("Leaf image → RGB preprocessing → color and texture features → disease class → general guidance")
        st.subheader("Model and evaluation")
        if METRICS_PATH.exists():
            try:
                metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                x, y, z = st.columns(3)
                x.metric("Holdout accuracy", f"{metrics['accuracy']:.1%}")
                y.metric("Macro F1", f"{metrics['macro_f1']:.3f}")
                z.metric("Evaluation images", f"{metrics['n_test']:,}")
            except (OSError, ValueError, KeyError):
                st.info("Training metrics are not available in this deployment.")
        st.warning("The training benchmark is not a guarantee of accuracy on field photos. Similar symptoms can have different causes. Severity is not predicted.")
        st.caption("Training dataset: Plant Leaf Diseases Dataset, Mendeley Data, DOI 10.17632/tywbtsjrjv.1.")
    else:
        st.markdown('<div class="eyebrow">Personal workspace</div>', unsafe_allow_html=True)
        st.title("Account")
        st.write("Your workspace is signed in as:")
        st.metric("Name", profile["full_name"])
        st.metric("Email", profile["email"])
        st.info("This academic prototype does not keep an image or assessment history. Uploads and notes stay in the active session.")
        st.warning("For persistent accounts on a public deployment, connect the app to a managed identity provider and database. The local SQLite account store is intended for a single-instance prototype.")


def main() -> None:
    apply_styles()
    profile = st.session_state.get("agro_user")
    if not profile:
        render_auth()
        return
    page = render_sidebar(profile)
    if page == "New assessment":
        render_assessment(profile)
    else:
        render_about(profile, page)
    st.divider()
    st.caption("AgroShield AI is an academic prototype. Guidance is general information, not a pesticide prescription. Confirm diagnosis and treatment locally before taking action.")


main()
