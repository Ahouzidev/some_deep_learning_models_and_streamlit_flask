# app_flask.py — same logic, prettier templates
from flask import Flask, request, render_template_string, redirect, url_for, abort
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
import io
import base64
import logging
import os

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fruits_cnn.h5")
ALLOWED_EXT = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2 MB upload limit

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Load model once (TF2). Wrap in try/except to show a clear error if model missing.
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
except Exception as e:
    logger.exception("Failed to load model: %s", e)
    model = None

classes = ["apple", "banana", "orange"]

# ---------- PRETTY TEMPLATES (Bootstrap 5) ----------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Fruit Classifier</title>
  <!-- Bootstrap 5 -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: linear-gradient(180deg,#f8fafc 0%, #ffffff 100%); min-height:100vh; }
    .card { border-radius: 1rem; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
    .logo { font-weight:700; letter-spacing: 0.2px; }
    .file-input { padding: .4rem; }
    .footer { font-size: .9rem; color: #6c757d; }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-light bg-white shadow-sm">
    <div class="container">
      <a class="navbar-brand logo" href="{{ url_for('index') }}">🍎 Fruit Classifier</a>
    </div>
  </nav>

  <main class="container py-5">
    <div class="row justify-content-center">
      <div class="col-md-8 col-lg-6">
        <div class="card p-4">
          <h5 class="mb-3">Upload an image</h5>
          <p class="text-muted small">Supported formats: PNG, JPG, JPEG. Max 2MB.</p>
          <form method="post" action="{{ url_for('predict') }}" enctype="multipart/form-data" class="row g-3">
            <div class="col-12">
              <input class="form-control file-input" type="file" name="file" accept="image/*" required>
            </div>
            <div class="col-12 d-grid">
              <button class="btn btn-primary" type="submit">Classify image</button>
            </div>
          </form>

          <hr class="my-4">

          <div class="text-center text-muted small">
            Tip: use square images for best results. The model expects 32×32 RGB inputs.
          </div>
        </div>

        <div class="text-center mt-3 footer">
          Built with Flask · Model inference runs server-side
        </div>
      </div>
    </div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Result — Fruit Classifier</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: linear-gradient(180deg,#f8fafc 0%, #ffffff 100%); min-height:100vh; }
    .card { border-radius: 1rem; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
    .label-pill { font-weight:600; font-size:1.05rem; }
    .prob-bar { height: 1rem; border-radius: .5rem; }
    .prob-row { gap: .6rem; align-items:center; }
  </style>
</head>
<body>
  <nav class="navbar navbar-expand-lg navbar-light bg-white shadow-sm">
    <div class="container">
      <a class="navbar-brand" href="{{ url_for('index') }}">🍎 Fruit Classifier</a>
    </div>
  </nav>

  <main class="container py-5">
    <div class="row justify-content-center">
      <div class="col-md-10 col-lg-8">
        <div class="card p-4">
          {% if error %}
            <div class="alert alert-danger" role="alert">
              {{ error }}
            </div>
            <div class="mt-3">
              <a class="btn btn-outline-primary" href="{{ url_for('index') }}">Try again</a>
            </div>
          {% else %}
            <div class="row g-3">
              <div class="col-md-5 text-center">
                <div class="border rounded p-2">
                  <img src="data:image/png;base64,{{ img_b64 }}" alt="uploaded" class="img-fluid" style="max-height:280px; object-fit:contain;">
                </div>
                <div class="mt-2 text-muted small">{{ filename }}</div>
              </div>

              <div class="col-md-7">
                <h4 class="mb-1">Prediction</h4>
                <div class="mb-3">
                  <span class="badge bg-success label-pill">{{ label|capitalize }}</span>
                  <span class="ms-2 text-muted">Confidence: <strong>{{ confidence }}%</strong></span>
                </div>

                <h6 class="mb-2">Class probabilities</h6>
                <div class="d-flex flex-column">
                  {% for k, v in probabilities.items() %}
                    <div class="d-flex prob-row mb-2">
                      <div style="width:95px; min-width:80px;">{{ k|capitalize }}</div>
                      <div class="flex-grow-1">
                        <div class="progress" style="height:16px; border-radius:8px;">
                          <div class="progress-bar" role="progressbar" style="width: {{ v }}%;" aria-valuenow="{{ v }}" aria-valuemin="0" aria-valuemax="100">
                            {{ v }}%
                          </div>
                        </div>
                      </div>
                    </div>
                  {% endfor %}
                </div>

                <div class="mt-4">
                  <a class="btn btn-outline-primary" href="{{ url_for('index') }}">Classify another</a>
                  <a class="btn btn-link text-muted" href="#" onclick="window.location.reload()">Refresh</a>
                </div>
              </div>
            </div>
          {% endif %}
        </div>

        <div class="text-center mt-3 small text-muted">
          Note: results are returned by a local TensorFlow model.
        </div>
      </div>
    </div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ---------- END TEMPLATES ----------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_image(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_resized = img.resize((32, 32))
    arr = np.array(img_resized).astype(np.float32)
   
    arr = np.expand_dims(arr, axis=0)
    return arr, img_resized

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(RESULT_HTML, error="Model failed to load. Check server logs."), 500

    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        return render_template_string(RESULT_HTML, error="File type not allowed. Use png/jpg/jpeg."), 400

    try:
        file_bytes = file.read()
        img_array, pil_img = preprocess_image(file_bytes)
    except UnidentifiedImageError:
        logger.exception("Uploaded file is not a valid image")
        return render_template_string(RESULT_HTML, error="Uploaded file is not a valid image."), 400
    except Exception:
        logger.exception("Error while preprocessing image")
        return render_template_string(RESULT_HTML, error="Error processing image."), 500

    try:
        preds = model.predict(img_array)
    except Exception:
        logger.exception("Error during model prediction")
        return render_template_string(RESULT_HTML, error="Model prediction failed."), 500

    # safe handling of preds shape
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[0] == 1:
        preds = preds[0]
    idx = int(np.argmax(preds))
    label = classes[idx]
    confidence = float(np.max(preds) * 100)
    probabilities = {classes[i]: round(float(preds[i] * 100), 2) for i in range(len(classes))}

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return render_template_string(RESULT_HTML, label=label, confidence=f"{confidence:.2f}",
                                  img_b64=img_b64, probabilities=probabilities, filename=getattr(request.files['file'], 'filename', 'image'))

# Helpful error handler for large uploads
@app.errorhandler(413)
def request_entity_too_large(error):
    return "File too large. Max size is 2 MB.", 413

if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    app.run(debug=True, port=5000)
