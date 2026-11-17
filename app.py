import os
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (Linear.pkl must sit in the same folder as app.py)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Linear.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

CURRENCY = "$"  # change to "₹" if your training data was in rupees

# Form fields: name, label, hint, input attributes, default value
FIELDS = [
    {"name": "Square_Footage", "label": "Square footage", "hint": "Built-up area in sq ft",
     "min": 300, "max": 10000, "step": 1, "default": 2000},
    {"name": "Num_Bedrooms", "label": "Bedrooms", "hint": "Number of bedrooms",
     "min": 1, "max": 10, "step": 1, "default": 3},
    {"name": "Num_Bathrooms", "label": "Bathrooms", "hint": "Number of bathrooms",
     "min": 1, "max": 10, "step": 1, "default": 2},
    {"name": "Year_Built", "label": "Year built", "hint": "Construction year",
     "min": 1900, "max": 2030, "step": 1, "default": 2005},
    {"name": "Lot_Size", "label": "Lot size", "hint": "Plot size in acres",
     "min": 0, "max": 20, "step": 0.01, "default": 2.5},
    {"name": "Garage_Size", "label": "Garage size", "hint": "Cars that fit in the garage",
     "min": 0, "max": 6, "step": 1, "default": 1},
    {"name": "Neighborhood_Quality", "label": "Neighborhood quality", "hint": "Rating from 1 (low) to 10 (high)",
     "min": 1, "max": 10, "step": 1, "default": 6},
]

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>House Price Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --ink: #1b1340;
      --muted: #6b6490;
      --line: #e6e1f7;
      --violet: #6a3df0;
      --magenta: #d946ef;
      --coral: #ff6b81;
      --amber: #ffb347;
      --teal: #22d3c5;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Plus Jakarta Sans", system-ui, sans-serif;
      color: var(--ink);
      min-height: 100vh;
      padding: 40px 20px;
      background:
        radial-gradient(900px 500px at 10% -10%, rgba(34, 211, 197, .55), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(255, 179, 71, .55), transparent 60%),
        linear-gradient(135deg, #3b1fa8 0%, #6a3df0 35%, #d946ef 70%, #ff6b81 100%);
      background-attachment: fixed;
    }
    .wrap { max-width: 1080px; margin: 0 auto; }
    header { color: #fff; margin-bottom: 28px; }
    header h1 { font-size: clamp(1.9rem, 4vw, 2.8rem); font-weight: 800; letter-spacing: -.02em; }
    header p { margin-top: 8px; font-size: 1.05rem; color: rgba(255,255,255,.85); max-width: 52ch; }

    .grid { display: grid; grid-template-columns: 1.35fr 1fr; gap: 24px; align-items: start; }
    @media (max-width: 860px) { .grid { grid-template-columns: 1fr; } }

    .panel {
      background: rgba(255,255,255,.94);
      border-radius: 24px;
      padding: 30px;
      box-shadow: 0 24px 60px rgba(30, 10, 90, .35);
    }
    .panel h2 { font-size: 1.15rem; font-weight: 700; margin-bottom: 20px; }

    .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 18px 16px; }
    @media (max-width: 560px) { .fields { grid-template-columns: 1fr; } }
    .field.wide { grid-column: 1 / -1; }
    label { display: block; font-size: .92rem; font-weight: 600; margin-bottom: 6px; }
    input[type=number] {
      width: 100%;
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: #faf9ff;
      border: 1.5px solid var(--line);
      border-radius: 12px;
      transition: border-color .15s, box-shadow .15s;
    }
    input[type=number]:focus {
      outline: none;
      border-color: var(--violet);
      box-shadow: 0 0 0 4px rgba(106, 61, 240, .18);
    }
    .hint { font-size: .78rem; color: var(--muted); margin-top: 5px; }

    button {
      margin-top: 26px;
      width: 100%;
      padding: 15px 20px;
      font: inherit;
      font-weight: 700;
      font-size: 1.02rem;
      color: #fff;
      border: 0;
      border-radius: 14px;
      cursor: pointer;
      background: linear-gradient(90deg, var(--violet), var(--magenta) 55%, var(--coral));
      box-shadow: 0 10px 24px rgba(217, 70, 239, .4);
      transition: transform .12s, box-shadow .12s;
    }
    button:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(217, 70, 239, .5); }
    button:focus-visible { outline: 3px solid var(--amber); outline-offset: 3px; }

    .result {
      position: sticky; top: 24px;
      color: #fff;
      border-radius: 24px;
      padding: 30px;
      background: linear-gradient(160deg, #1b1340 0%, #3b1fa8 55%, #6a3df0 100%);
      box-shadow: 0 24px 60px rgba(30, 10, 90, .4);
      overflow: hidden;
    }
    .result::after {
      content: "";
      position: absolute; right: -70px; bottom: -70px;
      width: 220px; height: 220px; border-radius: 50%;
      background: radial-gradient(circle, rgba(34,211,197,.55), transparent 70%);
    }
    .result small { display: block; color: rgba(255,255,255,.7); font-size: .95rem; }
    .price {
      margin-top: 10px;
      font-size: clamp(2.2rem, 5vw, 3.2rem);
      font-weight: 800;
      letter-spacing: -.02em;
      background: linear-gradient(90deg, var(--teal), #fff 50%, var(--amber));
      -webkit-background-clip: text;
      background-clip: text;
      color: transparent;
      word-break: break-word;
    }
    .result p { margin-top: 14px; font-size: .9rem; color: rgba(255,255,255,.75); line-height: 1.55; position: relative; z-index: 1; }
    .error { margin-top: 12px; padding: 12px 14px; border-radius: 12px; background: rgba(255,107,129,.18); color: #ffd7dd; font-size: .92rem; }
    .empty { color: rgba(255,255,255,.8); line-height: 1.6; position: relative; z-index: 1; }

    footer { text-align: center; color: rgba(255,255,255,.75); font-size: .85rem; margin-top: 26px; }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>House Price Predictor</h1>
      <p>Enter the property details and get an estimated sale price from a linear regression model.</p>
    </header>

    <div class="grid">
      <form class="panel" method="POST" action="/">
        <h2>Property details</h2>
        <div class="fields">
          {% for f in fields %}
          <div class="field {% if loop.last and fields|length is odd %}wide{% endif %}">
            <label for="{{ f.name }}">{{ f.label }}</label>
            <input type="number" id="{{ f.name }}" name="{{ f.name }}"
                   min="{{ f.min }}" max="{{ f.max }}" step="{{ f.step }}"
                   value="{{ values[f.name] }}" required>
            <div class="hint">{{ f.hint }}</div>
          </div>
          {% endfor %}
        </div>
        <button type="submit">Predict price</button>
      </form>

      <aside class="result" aria-live="polite">
        {% if error %}
          <small>Estimated price</small>
          <div class="error">{{ error }}</div>
        {% elif prediction is not none %}
          <small>Estimated price</small>
          <div class="price">{{ prediction }}</div>
          <p>This is an estimate from the model based on the seven values you entered. Change any value and predict again to compare.</p>
        {% else %}
          <div class="empty">Fill in the property details and press <b>Predict price</b>. Your estimate will appear here.</div>
        {% endif %}
      </aside>
    </div>

    <footer>Built with Flask and scikit-learn</footer>
  </div>
</body>
</html>
"""


def default_values():
    return {f["name"]: f["default"] for f in FIELDS}


@app.route("/", methods=["GET", "POST"])
def home():
    values = default_values()
    prediction = None
    error = None

    if request.method == "POST":
        try:
            row = {}
            for f in FIELDS:
                raw = request.form.get(f["name"], "").strip()
                values[f["name"]] = raw
                row[f["name"]] = float(raw)

            # Column order must match the order the model was trained on
            df = pd.DataFrame([row])[list(model.feature_names_in_)]
            result = float(model.predict(df)[0])
            prediction = f"{CURRENCY}{result:,.0f}"
        except ValueError:
            error = "Please enter a valid number in every field."
        except Exception as exc:  # keep the page alive on any model error
            error = f"Could not predict: {exc}"

    return render_template_string(
        PAGE, fields=FIELDS, values=values, prediction=prediction, error=error
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
