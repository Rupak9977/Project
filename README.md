# AI vs Real Image Detector

A Flask web app that detects whether an uploaded image is AI-generated or real,
using the `umm-maybe/AI-image-detector` model from HuggingFace.

---

## 📁 Project Structure

```
project/
├── app.py                  ← Flask backend
├── requirements.txt        ← Python dependencies
├── templates/
│   └── index.html          ← Frontend UI
├── static/
│   └── style.css           ← Styles
└── README.md
```

---

## ⚙️ Setup & Run (Step-by-Step)

### 1. Make sure Python 3.8+ is installed
```bash
python --version
```

### 2. (Recommended) Create a virtual environment
```bash
python -m venv venv

# Activate it:
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First run will download the model (~400MB). Subsequent runs are fast.

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## 🧠 How It Works

1. User uploads an image via the browser.
2. Flask receives the image, resizes it to 512×512.
3. The `umm-maybe/AI-image-detector` HuggingFace model (a fine-tuned ViT/ResNet classifier) runs inference.
4. The model returns class probabilities: "artificial" vs "human/real".
5. The top prediction and confidence score are sent back as JSON and displayed.

---

## ⚠️ Limitations

- Not 100% accurate — probabilistic output only.
- Works best on diffusion-model generated images (Stable Diffusion, DALL·E, Midjourney).
- May misclassify heavily edited real photos as AI-generated.
- Very low-resolution images may reduce accuracy.
- Adversarially crafted images can fool any classifier.

---

## 🎤 Viva Answer (2-liner)

> "Our system uses a pre-trained image classification model from HuggingFace that was trained to distinguish AI-generated images from real photographs.
> It analyzes texture patterns and artifact signatures that AI generators leave behind, then outputs a confidence-weighted prediction."

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| Slow on first run | Model downloading (~400MB) — wait once |
| CUDA error | CPU fallback is automatic, no action needed |
| Port 5000 in use | Change `port=5000` in `app.py` to `5001` |
