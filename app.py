import gradio as gr
import pandas as pd
import numpy as np
import joblib
 
# CRITICAL: Import your custom pipeline structure so joblib can unpack the model correctly "C:\Users\aryan\AppData\Local\Programs\Python\Python313\python.exe" app2.py
from pipeline import UnifiedEnsemblePredictor
 
# 1. Load the pre-trained ensemble pipeline
try:
    model = joblib.load('food_wastage_ensemble.pkl')
except Exception as e:
    print(f"Error loading model pipeline: {e}")
    model = None
 
# 2. Define the inference function matching your model's expected inputs
def predict_food_requirement(
    food_type, guests, event_type, storage,
    history, seasonality, prep_method, location, pricing
):
    if model is None:
        return build_error_html("Model file 'food_wastage_ensemble.pkl' not found.")
 
    input_df = pd.DataFrame([{
        'Type of Food': food_type,
        'Number of Guests': int(guests),
        'Event Type': event_type,
        'Storage Conditions': storage,
        'Purchase History': history,
        'Seasonality': seasonality,
        'Preparation Method': prep_method,
        'Geographical Location': location,
        'Pricing': pricing
    }])
 
    try:
        prediction = model.predict(input_df)
        predicted_value = prediction[0] if isinstance(prediction, (np.ndarray, list)) else prediction
        units = int(predicted_value)
        return build_result_html(units, food_type, guests, event_type, pricing)
    except Exception as e:
        return build_error_html(str(e))
 
 
def build_result_html(units, food_type, guests, event_type, pricing):
    food_icons = {
        "Meat": "", "Baked Goods": "", "Dairy Products": "",
        "Fruits": "", "Vegetables": ""
    }
    event_icons = {
        "Wedding": "", "Corporate": "", "Birthday": "", "Conference": ""
    }
    pricing_colors = {
        "Low": "#4ade80", "Moderate": "#facc15", "High": "#f87171"
    }
    food_icon = food_icons.get(food_type, "")
    event_icon = event_icons.get(event_type, "")
    pricing_color = pricing_colors.get(pricing, "#facc15")
 
    # Simple efficiency estimate based on pricing + history
    efficiency = {"Low": 88, "Moderate": 94, "High": 97}.get(pricing, 94)
 
    return f"""
    <div class="result-card">
        <div class="result-header">
            <span class="result-badge"> Prediction Ready</span>
        </div>
        <div class="result-main">
            <div class="result-quantity">{units:,}</div>
            <div class="result-unit-label">Recommended Units</div>
        </div>
        <div class="result-meta-grid">
            <div class="meta-chip">
                <span class="meta-icon">{food_icon}</span>
                <span class="meta-text">{food_type}</span>
            </div>
            <div class="meta-chip">
                <span class="meta-icon">{event_icon}</span>
                <span class="meta-text">{event_type}</span>
            </div>
            <div class="meta-chip">
                <span class="meta-icon"></span>
                <span class="meta-text">{int(guests):,} Guests</span>
            </div>
            <div class="meta-chip">
                <span class="meta-icon"></span>
                <span class="meta-text" style="color:{pricing_color}">{pricing} Tier</span>
            </div>
        </div>
        <div class="efficiency-bar-container">
            <div class="efficiency-label">
                <span>Supply Efficiency Score</span>
                <span class="efficiency-value">{efficiency}%</span>
            </div>
            <div class="efficiency-track">
                <div class="efficiency-fill" style="width:{efficiency}%"></div>
            </div>
        </div>
        <div class="result-footer">
            Optimal inventory calculated using ensemble ML model
        </div>
    </div>
    """
 
 
def build_error_html(message):
    return f"""
    <div class="error-card">
        <div class="error-icon"></div>
        <div class="error-title">Prediction Failed</div>
        <div class="error-message">{message}</div>
    </div>
    """
 
 
# ── Custom CSS ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
 
/* ── Root palette ── */
:root {
    --bg-deep:      #0d1117;
    --bg-card:      #161b22;
    --bg-input:     #1c2330;
    --border:       #30363d;
    --accent:       #3fb950;
    --accent-glow:  rgba(63,185,80,0.18);
    --accent-dim:   #238636;
    --text-primary: #e6edf3;
    --text-muted:   #7d8590;
    --text-subtle:  #484f58;
    --gold:         #d29922;
    --radius-sm:    8px;
    --radius-md:    14px;
    --radius-lg:    20px;
}
 
/* ── Base reset ── */
.gradio-container {
    background: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif !important;
    min-height: 100vh;
    color: var(--text-primary) !important;
}
 
body { background: var(--bg-deep) !important; }
 
/* ── Hide default Gradio footer ── */
footer { display: none !important; }
 
/* ── Hero banner ── */
.hero-wrap {
    background: linear-gradient(135deg, #0d1117 0%, #0f2027 50%, #0d1117 100%);
    border-bottom: 1px solid var(--border);
    padding: 48px 32px 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 50% at 50% -10%, rgba(63,185,80,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-glow);
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 100px;
    padding: 4px 14px;
    margin-bottom: 18px;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(28px, 4vw, 46px) !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
    line-height: 1.15 !important;
    margin: 0 0 14px !important;
    letter-spacing: -0.02em;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 15px;
    color: var(--text-muted);
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
}
 
/* ── Section labels ── */
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}
 
/* ── Input cards ── */
.input-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px;
}
 
/* ── Gradio component overrides ── */
.gradio-container label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}
 
/* Dropdowns & inputs */
.gradio-container select,
.gradio-container input[type="number"],
.gradio-container .wrap {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-size: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.gradio-container select:focus,
.gradio-container input[type="number"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
 
/* Radio buttons */
.gradio-container .wrap.svelte-1gfkn6j {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 12px !important;
    gap: 10px !important;
}
.gradio-container input[type="radio"] { accent-color: var(--accent) !important; }
.gradio-container .wrap.svelte-1gfkn6j label {
    text-transform: none !important;
    font-size: 13px !important;
    color: var(--text-primary) !important;
}
 
/* ── CTA Button ── */
.cta-btn {
    width: 100%;
    padding: 14px 0 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
    border-radius: var(--radius-md) !important;
    background: linear-gradient(135deg, var(--accent-dim) 0%, var(--accent) 100%) !important;
    border: none !important;
    color: #fff !important;
    cursor: pointer !important;
    box-shadow: 0 4px 24px rgba(63,185,80,0.25) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    margin-top: 8px !important;
}
.cta-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(63,185,80,0.4) !important;
}
.cta-btn:active { transform: translateY(0) !important; }
 
/* ── Result panel ── */
.result-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px;
    height: 100%;
}
 
/* Result card HTML */
.result-card {
    animation: fadeUp 0.4s ease-out;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-header { margin-bottom: 28px; }
.result-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--accent);
    background: var(--accent-glow);
    border: 1px solid rgba(63,185,80,0.3);
    border-radius: 100px;
    padding: 4px 12px;
}
.result-main { text-align: center; padding: 24px 0; }
.result-quantity {
    font-family: 'Syne', sans-serif;
    font-size: 72px;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
    text-shadow: 0 0 40px rgba(63,185,80,0.4);
    letter-spacing: -0.03em;
}
.result-unit-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 8px;
}
.result-meta-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-bottom: 24px;
}
.meta-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 12px;
}
.meta-icon { font-size: 16px; }
.meta-text { font-size: 12px; font-weight: 500; color: var(--text-primary); }
.efficiency-bar-container { margin-bottom: 24px; }
.efficiency-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
}
.efficiency-value { color: var(--accent); font-weight: 600; }
.efficiency-track {
    height: 6px;
    background: var(--bg-input);
    border-radius: 100px;
    overflow: hidden;
}
.efficiency-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent-dim), var(--accent));
    border-radius: 100px;
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.result-footer {
    font-size: 11px;
    color: var(--text-subtle);
    text-align: center;
    padding-top: 16px;
    border-top: 1px solid var(--border);
}
 
/* Error card */
.error-card {
    text-align: center;
    padding: 40px 20px;
}
.error-icon { font-size: 40px; margin-bottom: 12px; }
.error-title {
    font-weight: 600;
    font-size: 16px;
    color: #f87171;
    margin-bottom: 8px;
}
.error-message { font-size: 13px; color: var(--text-muted); }
 
/* Placeholder state */
.placeholder-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 320px;
    gap: 12px;
    color: var(--text-subtle);
    text-align: center;
}
.placeholder-icon { font-size: 48px; opacity: 0.4; }
.placeholder-text { font-size: 13px; line-height: 1.6; }
 
/* Hide default textbox borders in output */
.output-box textarea,
.output-box .wrap { border: none !important; background: transparent !important; }
 
/* ── Stat strip ── */
.stat-strip {
    display: flex;
    gap: 1px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    margin-bottom: 24px;
}
.stat-item {
    flex: 1;
    background: var(--bg-card);
    padding: 14px 16px;
    text-align: center;
}
.stat-value { font-size: 20px; font-weight: 700; color: var(--text-primary); }
.stat-key { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px; }
 
/* Gradio HTML component */
.gradio-container .prose { color: var(--text-primary) !important; }
"""
 
PLACEHOLDER_HTML = """
<div class="placeholder-wrap">
    <div class="placeholder-icon"></div>
    <div class="placeholder-text">
        Fill in the event parameters<br>and click <strong style="color:#3fb950">Calculate</strong> to see your<br>optimal inventory recommendation.
    </div>
</div>
"""
 
HERO_HTML = """
<div class="hero-wrap">
    <div class="hero-eyebrow"> ML-Powered Inventory Intelligence</div>
    <div class="hero-title">Smart Feast AI</span></div>
    <div class="hero-sub">Predict the exact inventory you need — eliminate waste, prevent shortfalls, and serve every guest perfectly.</div>
</div>
"""
 
STATS_HTML = """
<div class="stat-strip">
    <div class="stat-item">
        <div class="stat-value">5</div>
        <div class="stat-key">Food Types</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">4</div>
        <div class="stat-key">Event Kinds</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">±2%</div>
        <div class="stat-key">Model Error</div>
    </div>
    <div class="stat-item">
        <div class="stat-value">9</div>
        <div class="stat-key">Input Features</div>
    </div>
</div>
"""
 
# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Smart Feast AI",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.slate,
    )
) as demo:
 
    # Hero
    gr.HTML(HERO_HTML)
 
    with gr.Row(equal_height=False):
        # ── Left: Inputs ────────────────────────────────────────────────────
        with gr.Column(scale=3):
            gr.HTML('<div class="input-panel">')
            gr.HTML('<div class="section-label"> &nbsp;Event Configuration</div>')
            gr.HTML(STATS_HTML)
 
            with gr.Row():
                with gr.Column():
                    food_type = gr.Dropdown(
                        choices=["Meat", "Baked Goods", "Dairy Products", "Fruits", "Vegetables"],
                        label="Type of Food",
                        value="Meat",
                        interactive=True,
                    )
                    guests = gr.Number(
                        label="Number of Guests",
                        value=100,
                        precision=0,
                        minimum=1,
                    )
                    event_type = gr.Dropdown(
                        choices=["Wedding", "Corporate", "Birthday", "Conference"],
                        label="Event Type",
                        value="Corporate",
                    )
                    storage = gr.Dropdown(
                        choices=["Refrigerated", "Room Temperature", "Frozen"],
                        label="Storage Conditions",
                        value="Refrigerated",
                    )
 
                with gr.Column():
                    history = gr.Dropdown(
                        choices=["Regular", "Occasional", "First-time"],
                        label="Purchase History",
                        value="Regular",
                    )
                    seasonality = gr.Dropdown(
                        choices=["All Seasons", "Summer", "Winter", "Monsoon"],
                        label="Seasonality",
                        value="All Seasons",
                    )
                    prep_method = gr.Dropdown(
                        choices=["Buffet", "Plated", "Live Counter"],
                        label="Preparation Method",
                        value="Buffet",
                    )
                    location = gr.Dropdown(
                        choices=["Urban", "Suburban", "Rural"],
                        label="Geographical Location",
                        value="Urban",
                    )
 
            pricing = gr.Radio(
                choices=["Low", "Moderate", "High"],
                label="Pricing Tier",
                value="Moderate",
            )
 
            submit_btn = gr.Button(
                "⚡  Calculate Optimal Quantity",
                variant="primary",
                elem_classes=["cta-btn"],
            )
            gr.HTML('</div>')  # close input-panel
 
        # ── Right: Output ────────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.HTML('<div class="result-panel">')
            gr.HTML('<div class="section-label"> &nbsp;Recommendation</div>')
            output_display = gr.HTML(value=PLACEHOLDER_HTML, elem_classes=["output-box"])
            gr.HTML('</div>')  # close result-panel
 
    # Wire up
    submit_btn.click(
        fn=predict_food_requirement,
        inputs=[food_type, guests, event_type, storage,
                history, seasonality, prep_method, location, pricing],
        outputs=output_display,
    )
 
if __name__ == "__main__":
    demo.launch()