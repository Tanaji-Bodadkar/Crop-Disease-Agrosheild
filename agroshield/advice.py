"""Conservative, non-prescriptive guidance based on the predicted label."""

from __future__ import annotations


def _display_name(label: str) -> str:
    return label.replace("___", " — ").replace("_", " ").strip()


def advice_for(label: str) -> dict[str, object]:
    normalized = label.casefold().replace("___", " ").replace("_", " ")
    display = _display_name(label)
    if "healthy" in normalized:
        return {
            "title": display,
            "summary": "The model matched this image to a healthy class in its training labels.",
            "actions": [
                "Continue routine crop scouting; this result cannot rule out every plant-health issue.",
                "Retake the image or seek local advice if symptoms are visible or worsening.",
            ],
        }

    if any(word in normalized for word in ("virus", "viral", "mosaic", "yellow leaf curl")):
        actions = [
            "Check the plant and nearby plants for similar symptoms; confirm the diagnosis with a local agricultural expert.",
            "Clean tools after handling suspected plants and avoid moving plant material between plots until checked.",
            "Ask an extension service about locally appropriate vector and crop-management measures.",
        ]
    elif "bacterial" in normalized:
        actions = [
            "Avoid handling plants while foliage is wet and reduce water splash between plants.",
            "Clean tools and remove affected plant debris according to local crop guidance.",
            "Confirm the cause with an agricultural expert before applying any control product.",
        ]
    elif any(word in normalized for word in ("mite", "spider mite")):
        actions = [
            "Inspect leaf undersides and nearby plants to confirm whether mites are present.",
            "Ask a local agricultural expert about suitable monitoring and control for this crop.",
            "Avoid indiscriminate pesticide use, which can harm beneficial insects.",
        ]
    else:
        actions = [
            "Remove or isolate visibly affected material only after checking local crop guidance.",
            "Improve airflow where practical and avoid prolonged leaf wetness.",
            "Confirm the disease and any treatment with an agricultural expert or extension service.",
        ]

    return {
        "title": display,
        "summary": "This is a model prediction, not a confirmed diagnosis. Similar symptoms can have different causes.",
        "actions": actions,
    }
