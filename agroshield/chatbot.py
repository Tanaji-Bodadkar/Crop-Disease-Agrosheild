"""Session-local, rule-based crop-help responses grounded in the app's own output."""

from __future__ import annotations

import re
from typing import Any


def answer_crop_question(
    question: str,
    assessment: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Answer common crop-health questions without claiming to diagnose plants."""
    text = " ".join(question.casefold().split())
    if not text:
        return "Ask me about an assessment result, image quality, general prevention, or what to do next."

    if re.fullmatch(r"(hi|hello|hey|good morning|good afternoon)[!., ]*", text):
        return "Hello! I can explain an Agrosheild result, suggest general next steps, and help you prepare a clearer leaf photo. What would you like to know?"

    if any(term in text for term in ("accuracy", "macro f1", "model trained", "dataset", "training data")):
        if metrics:
            return (
                f"The ResNet18 CNN was evaluated on {metrics.get('n_test', 'a held-out set')} PlantVillage images "
                f"from {metrics.get('n_classes', 'multiple')} classes. Its held-out accuracy was "
                f"{metrics.get('accuracy', 0):.1%} and macro F1 was {metrics.get('macro_f1', 0):.3f}. "
                "These benchmark results do not establish performance on field photos."
            )
        return "The model benchmark is not available in this deployment. PlantVillage results also do not guarantee performance on field photos."

    if any(term in text for term in ("confidence", "score", "probability", "certain", "sure")):
        if assessment and "score" in assessment:
            return (
                f"The current model score is {assessment['score']:.1%} for **{assessment.get('prediction', 'the top class')}**. "
                "It is a softmax score, not a calibrated probability or a confirmed diagnosis. Check the top alternatives and compare with visible symptoms."
            )
        return "The app’s score is a model score, not a calibrated probability. Upload a clear leaf photo to get a result to discuss."

    if any(term in text for term in ("treatment", "pesticide", "spray", "medicine", "chemical", "cure")):
        if assessment and assessment.get("advice"):
            actions = assessment["advice"].get("actions", [])
            suggestions = "\n".join(f"- {action}" for action in actions)
            return (
                f"The current image was matched to **{assessment.get('advice', {}).get('title', assessment.get('prediction', 'a class'))}**. "
                "That is not a confirmed diagnosis. General next steps:\n" + suggestions
                + "\n\nConfirm the cause and any treatment with a local agricultural expert; I can’t prescribe chemicals."
            )
        return "I can’t prescribe a pesticide or confirm a treatment from text alone. Share the crop and symptoms, and consult a local agricultural expert before applying any product."

    if any(term in text for term in ("prevent", "prevention", "spread", "healthy", "protect", "avoid")):
        return (
            "General prevention: inspect plants regularly, keep tools clean, avoid moving wet or affected plant material between plots, "
            "and follow crop-specific guidance for your area. The right steps depend on the confirmed cause."
        )

    if any(term in text for term in ("photo", "picture", "image", "upload", "camera")):
        return (
            "Use one sharply focused leaf, natural light, and a plain background. Include the whole affected area, avoid shadows and glare, "
            "and enter the crop name plus when symptoms began. A photo-based result is still only a screening suggestion."
        )

    if assessment and assessment.get("prediction"):
        result_name = assessment.get("advice", {}).get("title", assessment["prediction"].replace("___", " — ").replace("_", " "))
        actions = assessment.get("advice", {}).get("actions", [])
        summary = assessment.get("advice", {}).get("summary", "This is a model suggestion, not a confirmed diagnosis.")
        next_step = actions[0] if actions else "Compare the result with symptoms and seek local advice if the plant is worsening."
        score = assessment.get("score")
        score_text = f" The CNN score is {score:.1%} (uncalibrated)." if isinstance(score, (float, int)) else ""
        alternatives = assessment.get("top_results", [])[:3]
        alternative_text = ""
        if alternatives:
            classes = ", ".join(item["class"] for item in alternatives)
            alternative_text = f" Other likely classes: {classes}."
        return (
            f"From the uploaded image, the CNN’s top match is **{result_name}**.{score_text} {summary} "
            f"A sensible next step is: {next_step}{alternative_text} The score is not a confirmed diagnosis; ask a local agricultural expert if symptoms are spreading or worsening."
        )

    return (
        "I can help with the current model result, its score, photo quality, and general prevention. "
        "For a tailored discussion, upload a clear leaf image and include the crop and symptoms. I can’t confirm a diagnosis or prescribe treatment."
    )


