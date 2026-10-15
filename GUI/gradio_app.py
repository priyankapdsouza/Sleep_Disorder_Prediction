import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import gradio as gr
import numpy as np
import tensorflow as tf
import tf_keras as keras
import os
import mne
import matplotlib.pyplot as plt
import random

# ===== Model paths =====
healthy_model_path = "healthy_cap_new.h5"
disease_model_path = "model_skip_D70_new.h5"
health_model_path = "model_skip_HU_new.h5"
disease_caps = ['ins_cap_new.h5', 'nfle_cap_new.h5', 'narco_cap_new.h5', 'rbd_cap_new.h5', 'plm_cap_new.h5']
diseases = ['INS', 'NFLE', 'NARCO', 'RBD', 'PLM']

# ===== Quotes =====
QUOTES = [
      "💤 “Sleep is the best meditation.”",
    "🌙 “Health is the crown on the well person’s head that only the ill person can see.”",
    "🩺 “A good laugh and a long sleep are the best cures in the doctor’s book.”",
    "😴 “Your future depends on your dreams — so go to sleep.”",
    "^_^ “Take care of your sleep, and your sleep will take care of you.”",
    "🌷“Healthy Sleep, Healthy Mind.”",
    "✨“Sleep is the golden chain that ties health and our bodies together.”",
    "🛌“The best bridge between despair and hope is a good night’s sleep.” ",
    "🦋“Sleep is that golden thread that ties health and our bodies together.”",
    "🍁“A day without a nap is like a cupcake without frosting.”",
    "🤍“Sorrow can be alleviated by good sleep…”",
    "🕸“The worst thing in the world is to try to sleep and not to.”",
    "🌼“No day is so bad it can’t be fixed with a nap.”"

]

# ===== EDF/REC → array =====
def preprocess_signal(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data, _ = raw[:, :]
    if data.ndim > 1:
        data = np.mean(data, axis=0)
    data = data[:1024] if len(data) >= 1024 else np.pad(data, (0, 1024 - len(data)))
    return np.array(data).reshape(1, 1024)

# ===== CSV → array =====
def load_csv(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    data = data[:1024] if len(data) >= 1024 else np.pad(data, (0, 1024 - len(data)))
    return data.reshape((1, 1024))

# ===== Main Analysis =====
def cap_analysis(file):
    if file is None:
        return (
            "<div class='result-card'><h3>🩺 Health Report</h3><p>Please upload an EEG file.</p></div>",
            "<div class='result-card'><h3>🧬 Predicted Disorder</h3><p>—</p></div>",
            "<div class='result-card'><h3>🌙 CAP Phase</h3><p>—</p></div>",
            None
        )

    file_path = file.name
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".edf", ".rec"]:
        x_bal = preprocess_signal(file_path)
        waveform = x_bal.flatten()
    elif ext == ".csv":
        x_bal = load_csv(file_path)
        waveform = x_bal.flatten()
    else:
        return (
            "<div class='result-card'><h3>🩺 Health Report</h3><p>Unsupported file type. Please use .CSV, .EDF, or .REC.</p></div>",
            "<div class='result-card'><h3>🧬 Predicted Disorder</h3><p>—</p></div>",
            "<div class='result-card'><h3>🌙 CAP Phase</h3><p>—</p></div>",
            None
        )

    # --- Plot waveform ---
    plt.figure(figsize=(10, 3), facecolor='none')
    plt.plot(waveform, linewidth=1.5)
    plt.title("EEG Signal Preview")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plot_path = "waveform.png"
    plt.savefig(plot_path, transparent=True)
    plt.close()

  
    try:
        

        # Initial CAP on generic healthy CAP model
        model_cap_healthy = keras.models.load_model(
            healthy_model_path,
            compile=False
        )
        pred_cap_initial = model_cap_healthy.predict(x_bal, verbose=0)
        cap_text = '🌙 CAP phase: B' if pred_cap_initial[0][0] < 0.5 else '🌙 CAP phase: A'

        # Healthy vs Unhealthy
        health_model = keras.models.load_model(
            health_model_path,
            compile=False
        )
        pred_health = health_model.predict(x_bal, verbose=0)
    except Exception as e:
        return (
            f"<div class='result-card'><h3>🩺 Health Report</h3><p>Model loading error: {e}</p></div>",
            "<div class='result-card'><h3>🧬 Predicted Disorder</h3><p>—</p></div>",
            "<div class='result-card'><h3>🌙 CAP Phase</h3><p>—</p></div>",
            None
        )

    if pred_health[0][0] > 0.5:
        # Positive / unhealthy
        health_str = "🩺 Health Report: <span class='tag-bad'>Needs Attention</span>"
        disease_model = keras.models.load_model(disease_model_path)
        pred_dis = disease_model.predict(x_bal, verbose=0)
        i = pred_dis.argmax()
        disease_str = f"🧬 Predicted Disorder: <span class='tag-disorder'>{diseases[i]}</span>"

        # CAP based on disease-specific model
        model_disease_cap = keras.models.load_model(disease_caps[i])
        pred_cap_disease = model_disease_cap.predict(x_bal, verbose=0)
        cap_text = '🌙 CAP phase: B' if pred_cap_disease[0][0] < 0.5 else '🌙 CAP phase: A'
    else:
        health_str = "🩺 Health Report: <span class='tag-good'>Stable</span>"
        disease_str = "🧬 Predicted Disorder: <span class='tag-none'>No disorder detected</span>"
        # CAP stays from initial CAP model

    health_card = f"""
    <div class='result-card'>
      <h3>Sleep Health</h3>
      <p>{health_str}</p>
    </div>
    """

    disease_card = f"""
    <div class='result-card'>
      <h3>Disorder Screening</h3>
      <p>{disease_str}</p>
    </div>
    """

    cap_card = f"""
    <div class='result-card'>
      <h3>CAP Dynamics</h3>
      <p>{cap_text}</p>
    </div>
    """

    return health_card, disease_card, cap_card, plot_path

# ===== Navigation =====
def go_to_prediction():
    return gr.update(visible=False), gr.update(visible=True)

def go_back_home():
    return gr.update(visible=True), gr.update(visible=False)

# ===== CSS styled like modern landing page =====
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Plus+Jakarta+Sans:wght@600;800&display=swap');

:root {
  --primary: #e5e7eb;
  --accent: #6366f1;
  --accent-soft: rgba(99,102,241,0.25);
  --danger: #fb7185;
  --good: #22c55e;
  --text-muted: #9ca3af;
  --card-bg: rgba(15,23,42,0.98);
  --border-subtle: rgba(148, 163, 184, 0.28);
  --shadow-soft: 0 26px 60px rgba(15, 23, 42, 0.95);
}

/* ===== BACKGROUND (animated grey / dark) ===== */
html, body, .gradio-container {
  background: radial-gradient(circle at 0% 0%, #111827 0, #020617 45%, #020617 100%);
  background-image:
    radial-gradient(circle at 0% 0%, rgba(148,163,184,0.35) 0, transparent 50%),
    radial-gradient(circle at 100% 100%, rgba(55,65,81,0.55) 0, transparent 55%),
    linear-gradient(135deg, #020617, #030712, #020617);
  background-size: 220% 220%;
  animation: bg-move 22s ease-in-out infinite;
  font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--primary);
  margin: 0;
  padding: 0;
}

/* gentle animated gradient */
@keyframes bg-move {
  0%   { background-position: 0% 0%; }
  50%  { background-position: 100% 100%; }
  100% { background-position: 0% 0%; }
}

/* Remove stock panels */
.block, .gradio-container .gr-block, .gradio-container .gr-panel {
  background: transparent !important;
  box-shadow: none !important;
  border: none !important;
}

/* ===== NAVBAR ===== */
.navbar {
  max-width: 1180px;
  margin: 18px auto 0;
  padding: 12px 26px;
  border-radius: 26px;
  background: rgba(15, 23, 42, 0.96);
  box-shadow: 0 22px 60px rgba(15, 23, 42, 0.9);
  display: flex;
  align-items: center;
  justify-content: space-between;
  backdrop-filter: blur(18px);
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.nav-logo {
  width: 32px;
  height: 32px;
  border-radius: 999px;
  background: radial-gradient(circle at 30% 0, #ffffff 0, #6366f1 40%, #0f172a 100%);
  box-shadow: 0 0 0 2px rgba(148,163,184,0.5);
}

.nav-title {
  font-family: 'Plus Jakarta Sans', system-ui;
  font-weight: 800;
  letter-spacing: 0.08em;
  font-size: 18px;
  color: #f9fafb;
  text-transform: uppercase;
}

.nav-pill {
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(99,102,241,0.16);
  font-size: 11px;
  text-transform: uppercase;
  color: #c7d2fe;
}

/* ===== MAIN CURVED PANEL WRAPPER ===== */
.hero-wrap,
.prediction-wrap {
  max-width: 1180px;
  margin: 28px auto 30px;
  padding: 26px 26px 30px;
  border-radius: 32px;
  background:
    radial-gradient(circle at top left, rgba(148,163,184,0.28) 0, transparent 55%),
    radial-gradient(circle at bottom right, rgba(51,65,85,0.55) 0, transparent 60%),
    var(--card-bg);
  box-shadow: var(--shadow-soft);
  border: 1px solid rgba(31,41,55,0.9);
}

/* ===== HERO GRID ===== */
.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
  gap: 32px;
  align-items: center;
}

.hero-left-title {
  font-family: 'Plus Jakarta Sans', system-ui;
  font-size: 42px;
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: -0.03em;
  color: #f9fafb;
}

.hero-left-highlight {
  color: #a855f7;
}

.hero-subtitle {
  margin-top: 14px;
  font-size: 15px;
  color: var(--text-muted);
}

.hero-badges {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.hero-badge {
  padding: 6px 12px;
  border-radius: 999px;
  background: rgba(15,23,42,0.92);
  border: 1px solid rgba(148, 163, 184, 0.6);
  font-size: 11px;
  color: #e5e7eb;
}

.hero-cta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  align-items: center;
  margin-top: 22px;
}

.secondary-link {
  font-size: 13px;
  color: var(--text-muted);
}

/* ===== RIGHT HERO CARD ===== */
.hero-right-card {
  position: relative;
  padding: 20px 18px;
  border-radius: 26px;
  background:
    radial-gradient(circle at 0 0, rgba(79,70,229,0.35) 0, transparent 55%),
    radial-gradient(circle at 100% 100%, rgba(236,72,153,0.35) 0, transparent 55%),
    #020617;
  color: #e5e7eb;
  box-shadow: 0 26px 70px rgba(15, 23, 42, 0.95);
  overflow: hidden;
}

.hero-right-inner {
  position: relative;
  z-index: 1;
}

.hero-pill {
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 11px;
  background: rgba(15,23,42,0.9);
  border: 1px solid rgba(148,163,184,0.7);
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #e5e7eb;
}

.hero-pill-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #4ade80;
}

.hero-mini-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 10px;
  margin-top: 16px;
}

.hero-mini-card {
  padding: 10px 11px;
  border-radius: 16px;
  background: rgba(15,23,42,0.96);
  border: 1px solid rgba(148,163,184,0.4);
  font-size: 12px;
}

.hero-mini-label {
  color: #9ca3af;
  font-size: 11px;
}

.hero-mini-value {
  font-size: 16px;
  font-weight: 600;
  color: #e5e7eb;
}

/* Quote */
.quote-box {
  margin-top: 20px;
  font-size: 13px;
  color: var(--text-muted);
}

/* ===== RESULT CARDS ===== */
.result-card {
  background: rgba(15,23,42,0.98);
  border-radius: 22px;
  padding: 18px 18px 16px;
  border: 1px solid var(--border-subtle);
  box-shadow: 0 18px 40px rgba(15,23,42,0.9);
  font-size: 14px;
  color: #e5e7eb;
}

.result-card h3 {
  margin-top: 0;
  margin-bottom: 8px;
  font-size: 16px;
  color: #f9fafb;
}

/* Tags */
.tag-good {
  color: var(--good);
  font-weight: 600;
}

.tag-bad {
  color: var(--danger);
  font-weight: 600;
}

.tag-disorder {
  color: #fb923c;
  font-weight: 600;
}

.tag-none {
  color: var(--good);
  font-weight: 500;
}

/* ===== BUTTONS ===== */
.gr-button.primary-btn-class {
  background: linear-gradient(135deg, #6366f1, #06b6d4) !important;
  border-radius: 999px !important;
  color: #fff !important;
  font-weight: 600 !important;
  border: none !important;
  box-shadow: 0 18px 40px rgba(37, 99, 235, 0.6);
}

.gr-button.primary-btn-class:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

.gr-button.secondary-btn-class {
  background: rgba(15,23,42,0.9) !important;
  border-radius: 999px !important;
  border: 1px solid rgba(148,163,184,0.6) !important;
  color: #e5e7eb !important;
  font-weight: 500 !important;
  padding: 8px 18px !important;
  box-shadow: 0 10px 25px rgba(15,23,42,0.45);
  margin-top: 16px;
  align-self: center;
  transition: transform .18s ease, box-shadow .18s ease;
}

.gr-button.secondary-btn-class:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 35px rgba(15,23,42,0.65);
}

/* File + image styling */
.gradio-container .gr-file, 
.gradio-container .gr-image {
  background: rgba(15,23,42,0.98) !important;
  border-radius: 18px !important;
  border: 1px dashed rgba(148,163,184,0.85) !important;
  color: #e5e7eb !important;
}

/* Text / inputs (future-safe) */
.gradio-container input,
.gradio-container textarea,
.gradio-container .gr-textbox,
.gradio-container .gr-text {
  background: rgba(15,23,42,0.97) !important;
  color: #e5e7eb !important;
  border-radius: 12px !important;
  border: 1px solid rgba(148,163,184,0.6) !important;
}

/* Footer */
.footer {
  max-width: 1180px;
  margin: 0 auto 24px;
  padding: 10px 12px;
  font-size: 12px;
  color: var(--text-muted);
  text-align: center;
}

/* Responsive */
@media (max-width: 880px) {
  .hero-grid {
    grid-template-columns: minmax(0, 1fr);
  }
  .navbar {
    border-radius: 20px;
  }
  .hero-left-title {
    font-size: 32px;
  }
  .hero-wrap,
  .prediction-wrap {
    padding: 20px 18px 24px;
    border-radius: 24px;
  }
}
"""



# ===== Interface =====
with gr.Blocks(css=css, title="DeepNap – AI Sleep Wellness") as demo:
    # Navbar
    gr.Markdown("""
    <div class="navbar">
      <div class="nav-left">
        <div class="nav-logo"></div>
        <div>
          <div class="nav-title">DEEPNAP</div>
          <div class="nav-pill">AI Sleep Wellness</div>
        </div>
      </div>
      <div style="font-size:12px; color:#6b7280;">Clinical-inspired • Deep Learning • EEG Insights</div>
    </div>
    """)

    # --- Home / Hero Page --- #
    with gr.Group(visible=True) as home_page:
        gr.Markdown(f"""
        <div class="hero-wrap">
          <div class="hero-grid">
            <div>
              <div class="hero-left-title">
                Understand your<br>
                <span class="hero-left-highlight">sleep brainwaves</span><br>
                in just a few seconds.
              </div>
              <div class="hero-subtitle">
                DeepNap analyzes your EEG signals with deep learning models to screen for
                sleep disorders and visualize CAP phases — in a calm, human way.
              </div>
              <div class="hero-badges">
                <div class="hero-badge">EEG · Sleep Disorder Screening</div>
                <div class="hero-badge">CAP A/B Phase Detection</div>
                <div class="hero-badge">End-to-end AI Pipeline</div>
              </div>
              <div class="hero-cta-row">
                <!-- Button rendered by Gradio below -->
                <span class="secondary-link">Works with .EDF, .REC, and .CSV EEG files.</span>
              </div>
              <div class="quote-box">{random.choice(QUOTES)}</div>
            </div>
            <div>
              <div class="hero-right-card">
                <div class="hero-right-inner">
                  <div class="hero-pill">
                    <span class="hero-pill-dot"></span>
                     EEG Insight
                  </div>
                  <h3 style="margin-top:14px; margin-bottom:4px; font-size:18px;">
                    Tonight’s Sleep Snapshot
                  </h3>
                  <p style="font-size:13px; color:#d1d5db; margin-bottom:10px;">
                    Upload a segment of your EEG and get an instant AI-driven health overview.
                  </p>
                  <div class="hero-mini-grid">
                    <div class="hero-mini-card">
                      <div class="hero-mini-label">Models</div>
                      <div class="hero-mini-value">CNN · LSTM · Incep</div>
                    </div>
                    <div class="hero-mini-card">
                      <div class="hero-mini-label">Disorders</div>
                      <div class="hero-mini-value">INS · NFLE · RBD…</div>
                    </div>
                    <div class="hero-mini-card">
                      <div class="hero-mini-label">Status</div>
                      <div class="hero-mini-value">Analyzed Prediction</div>
                    </div>
                    <div class="hero-mini-card">
                      <div class="hero-mini-label">CAP Phase</div>
                      <div class="hero-mini-value">A / B</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """)

        # Start button (placed under hero text)
        start_btn = gr.Button("Start EEG Analysis →", elem_classes=["primary-btn-class"])

    # --- Prediction Page --- #
    with gr.Group(visible=False) as prediction_page:
        gr.Markdown("""
        <div class="prediction-wrap">
          <h2 style="font-family:'Plus Jakarta Sans',system-ui; font-size:26px; margin-bottom:4px;">
            Upload your EEG segment
          </h2>
          <p style="color:#6b7280; font-size:14px; margin-bottom:18px;">
            We’ll run it through our trained models to determine health status, probable disorder,
            and CAP phase dynamics.
          </p>
        </div>
        """)

        with gr.Column(elem_id="upload-area", elem_classes=["prediction-wrap"]):
            file_input = gr.File(label="📁 Upload EEG File (.edf, .rec, .csv)")
            analyze_btn = gr.Button("Run DeepNap Analysis", elem_classes=["primary-btn-class"])

        with gr.Row(elem_classes=["prediction-wrap"]):
            health = gr.Markdown()
            disease = gr.Markdown()
            cap = gr.Markdown()

        with gr.Column(elem_classes=["prediction-wrap"]):
            plot_output = gr.Image(label="📊 EEG Waveform Preview", show_label=True)
            back_btn = gr.Button("← Back to landing page", elem_classes=["secondary-btn-class"])

        analyze_btn.click(cap_analysis, inputs=file_input, outputs=[health, disease, cap, plot_output])
        back_btn.click(go_back_home, None, [home_page, prediction_page])

    # Wire hero button
    start_btn.click(go_to_prediction, None, [home_page, prediction_page])

    # Footer
    gr.Markdown("""
    <div class="footer">
      DeepNap • Academic Sleep Disorder Detection Project • Built with Python, TensorFlow & Gradio
    </div>
    """)

demo.launch(share=True)
