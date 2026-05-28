from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from transformers import pipeline
import io

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'}

print("Loading AI image detection model...")
detector = pipeline(
    "image-classification",
    model="Organika/sdxl-detector",   # labels: "AI" and "Real"
    device=0 if torch.cuda.is_available() else -1
)
print("Model loaded successfully!")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((512, 512))
    return img


def classify_label(raw_label: str) -> str:
    """
    Organika/sdxl-detector returns 'AI' and 'Real'.
    Explicit mapping — no ambiguity.
    """
    label = raw_label.strip().lower()

    if label in ("ai", "artificial", "fake", "generated", "ai-generated", "ai_generated"):
        return "AI Generated"
    if label in ("real", "human", "authentic", "natural", "photograph"):
        return "Real Image"

    # Substring fallback
    if "real" in label or "human" in label or "authentic" in label:
        return "Real Image"
    if "ai" in label or "fake" in label or "gen" in label:
        return "AI Generated"

    # Last resort: return raw
    return raw_label


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, or WEBP."}), 400

    try:
        image_bytes = file.read()
        img = preprocess_image(image_bytes)

        # Get ALL scores (both classes)
        results = detector(img, top_k=None)

        # Debug: see raw output in terminal
        print("Raw model output:", results)

        # Build display-label → score dict
        score_dict = {}
        for r in results:
            display = classify_label(r["label"])
            score_dict[display] = max(score_dict.get(display, 0), r["score"])

        # Winner = highest score
        prediction = max(score_dict, key=score_dict.get)
        confidence = round(score_dict[prediction] * 100, 2)

        all_results = [
            {"label": lbl, "score": round(sc * 100, 2)}
            for lbl, sc in sorted(score_dict.items(), key=lambda x: -x[1])
        ]

        print(f"→ Prediction: {prediction}  Confidence: {confidence}%")

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "all_results": all_results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
