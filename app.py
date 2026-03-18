from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from detector import MODEL_NAMES, PlagiarismDetector, extract_text_from_bytes, normalize_text


BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 3 * 1024 * 1024


@lru_cache(maxsize=1)
def get_detector() -> PlagiarismDetector:
    return PlagiarismDetector(BASE_DIR)


@app.route("/")
def index():
    detector = get_detector()
    return render_template(
        "index.html",
        model_choices=detector.available_models(),
        default_model=detector.best_model,
    )


@app.route("/health")
def health():
    detector = get_detector()
    return jsonify(
        {
            "status": "ok",
            "available_models": [item["name"] for item in detector.available_models()],
            "best_model": detector.best_model,
        }
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    submitted_text = request.form.get("text", "")
    model_name = request.form.get("model", "lr").lower()
    uploaded_file = request.files.get("file")

    source_name = "Pasted text"
    try:
        if uploaded_file and uploaded_file.filename:
            source_name = uploaded_file.filename
            submitted_text = extract_text_from_bytes(uploaded_file.filename, uploaded_file.read())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    clean_text = normalize_text(submitted_text)
    if len(clean_text) < 40:
        return jsonify({"error": "Please paste more text or upload a supported text file."}), 400

    if model_name not in MODEL_NAMES:
        return jsonify({"error": "Invalid model selected."}), 400

    try:
        results = get_detector().analyze_text(clean_text, model_name=model_name)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"source_name": source_name, "text_preview": clean_text[:350], "results": results})


if __name__ == "__main__":
    get_detector()
    app.run(debug=True)
