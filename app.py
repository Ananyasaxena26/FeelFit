import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="FeelFit — AI Wellness", layout="wide", page_icon="⚡")

# ---------------------------------------------------
# Inject Custom CSS — Dark Athletic Aesthetic
# ---------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #f0ede8;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 10%, #1a0a2e 0%, #0a0a0f 50%),
                radial-gradient(ellipse at 80% 90%, #0d1f0d 0%, transparent 50%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #0a0a0f 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #c8c4be !important; }
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 2px !important;
    color: #e8ff5a !important;
    font-size: 1.1rem !important;
}
[data-testid="stSidebar"] .stSlider > label {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888 !important;
}

/* Slider accent */
[data-testid="stSidebar"] [data-baseweb="slider"] [data-testid="stThumbValue"] {
    background: #e8ff5a !important;
    color: #0a0a0f !important;
}
div[data-baseweb="slider"] div[role="slider"] {
    background: #e8ff5a !important;
    border-color: #e8ff5a !important;
}

/* ── Page title area ── */
.hero-banner {
    background: linear-gradient(135deg, #e8ff5a 0%, #b8ff00 50%, #78e830 100%);
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: 'VITALFLOW';
    position: absolute;
    right: -10px;
    top: -20px;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 9rem;
    color: rgba(0,0,0,0.07);
    letter-spacing: 4px;
    pointer-events: none;
}
.hero-banner h1 {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.2rem;
    letter-spacing: 4px;
    color: #0a0a0f;
    margin: 0;
    line-height: 1;
}
.hero-banner p {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    color: rgba(0,0,0,0.6);
    margin: 8px 0 0 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ── Section headers ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e8ff5a;
    margin-bottom: 12px;
    opacity: 0.9;
}

/* ── Stat cards ── */
.stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #e8ff5a, transparent);
}
.stat-card .card-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #666;
    margin-bottom: 8px;
    font-family: 'Space Mono', monospace;
}
.stat-card .card-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    line-height: 1;
    color: #f0ede8;
    letter-spacing: 1px;
}
.stat-card .card-unit {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: #666;
    margin-top: 4px;
}
.stat-card .card-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.65rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 12px;
    font-family: 'Space Mono', monospace;
}
.badge-green  { background: rgba(120,232,48,0.15);  color: #78e830; border: 1px solid rgba(120,232,48,0.3); }
.badge-yellow { background: rgba(232,255,90,0.15);  color: #e8ff5a; border: 1px solid rgba(232,255,90,0.3); }
.badge-red    { background: rgba(255,80,80,0.12);   color: #ff6060; border: 1px solid rgba(255,80,80,0.2); }
.badge-blue   { background: rgba(80,160,255,0.12);  color: #60aaff; border: 1px solid rgba(80,160,255,0.2); }

/* ── HR row ── */
.hr-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}
.hr-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.hr-card .hr-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    color: #e8ff5a;
    letter-spacing: 1px;
    line-height: 1;
}
.hr-card .hr-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #555;
    margin-top: 6px;
    font-family: 'Space Mono', monospace;
}

/* ── Productivity meter ── */
.prod-bar-wrap {
    background: rgba(255,255,255,0.04);
    border-radius: 100px;
    height: 10px;
    width: 100%;
    margin: 14px 0 8px;
    overflow: hidden;
}
.prod-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #e8ff5a, #78e830);
    transition: width 0.6s ease;
}

/* ── Food cards ── */
.food-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.food-name {
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: #f0ede8;
}
.food-meta {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: #555;
    margin-top: 4px;
    letter-spacing: 1px;
}
.food-kcal {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.6rem;
    color: #e8ff5a;
    letter-spacing: 1px;
}
.food-kcal-label {
    font-size: 0.55rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: right;
    font-family: 'Space Mono', monospace;
}

/* ── Insight banner ── */
.insight-banner {
    border-radius: 14px;
    padding: 18px 24px;
    margin-top: 24px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 14px;
}
.insight-success { background: rgba(120,232,48,0.08);  border: 1px solid rgba(120,232,48,0.2); color: #78e830; }
.insight-info    { background: rgba(80,160,255,0.08);  border: 1px solid rgba(80,160,255,0.2); color: #60aaff; }
.insight-warning { background: rgba(255,180,50,0.08);  border: 1px solid rgba(255,180,50,0.2); color: #ffb432; }

/* ── Divider ── */
.vf-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 28px 0;
}

/* ── Streamlit metric override ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] { 
    font-family: 'Space Mono', monospace !important;
    font-size: 0.6rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #555 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
    color: #e8ff5a !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load datasets
# ---------------------------------------------------

df       = pd.read_csv("data/unified_wellness_dataset.csv")
food_df  = pd.read_csv("data/indian_food_nutrition.csv")

# ---------------------------------------------------
# Train ML Model
# ---------------------------------------------------

@st.cache_resource
def train_model(df):
    candidates = [
        "TotalSteps", "VeryActiveMinutes", "FairlyActiveMinutes",
        "LightlyActiveMinutes", "SedentaryMinutes", "Calories",
        "avg_hr", "hr_std", "sleep_hours"
    ]
    feature_cols = [c for c in candidates if c in df.columns]
    df = df.copy().dropna(subset=feature_cols)

    df["target_productivity"] = (
        0.35 * (df["VeryActiveMinutes"] + df.get("FairlyActiveMinutes", 0)) / 90 +
        0.25 * (df["sleep_hours"] / 8 if "sleep_hours" in df.columns else 0.5) +
        0.20 * (df["TotalSteps"] / 12000 if "TotalSteps" in df.columns else 0.5) +
        0.10 * (1 - df["SedentaryMinutes"] / 1440 if "SedentaryMinutes" in df.columns else 0.5) +
        0.10 * (1 / (1 + df["hr_std"]) if "hr_std" in df.columns else 0.5)
    ).clip(0, 1) * 10

    X = df[feature_cols]
    y = df["target_productivity"]
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model   = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler, feature_cols

model, scaler, feature_cols = train_model(df)

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.markdown("## ⚡ ACTIVITY")
steps          = st.sidebar.slider("Total Steps",             0,    20000, 7000)
very_active    = st.sidebar.slider("Very Active Minutes",     0,    120,   30)
fairly_active  = st.sidebar.slider("Fairly Active Minutes",   0,    120,   20)
lightly_active = st.sidebar.slider("Lightly Active Minutes",  0,    300,   60)
sedentary      = st.sidebar.slider("Sedentary Minutes",       0,    900,   500)
calories       = st.sidebar.slider("Calories Burned",         0,    5000,  2000)
sleep_hours    = st.sidebar.slider("Sleep Hours",             0.0,  12.0,  7.0)

st.sidebar.markdown("## ❤️ HEART RATE")
user_avg_hr = st.sidebar.slider("Avg Heart Rate (bpm)",  40,   180,   75)
user_max_hr = st.sidebar.slider("Max Heart Rate (bpm)",  60,   220,   150)
user_min_hr = st.sidebar.slider("Min Heart Rate (bpm)",  30,   100,   55)
user_hr_std = st.sidebar.slider("HR Variability (std)",  0.0,  30.0,  5.0)

# ---------------------------------------------------
# Compute scores
# ---------------------------------------------------

input_map = {
    "TotalSteps": steps, "VeryActiveMinutes": very_active,
    "FairlyActiveMinutes": fairly_active, "LightlyActiveMinutes": lightly_active,
    "SedentaryMinutes": sedentary, "Calories": calories,
    "avg_hr": user_avg_hr, "hr_std": user_hr_std, "sleep_hours": sleep_hours,
}
input_row  = {col: input_map[col] for col in feature_cols}
input_df   = pd.DataFrame([input_row])
input_scaled = scaler.transform(input_df)
productivity_score = float(np.clip(model.predict(input_scaled)[0], 0, 10))

mood_score = float(np.clip((
    0.4 * (sleep_hours / 8) +
    0.3 * (steps / 12000) +
    0.3 * (1 / (1 + user_hr_std))
) * 10, 0, 10))

estimated_calories = int(0.04 * steps + 5 * very_active + 3 * fairly_active)

if mood_score < 4:     mood_label, mood_badge = "Stressed",  "badge-red"
elif mood_score < 7:   mood_label, mood_badge = "Neutral",   "badge-yellow"
else:                  mood_label, mood_badge = "Happy",     "badge-green"

if steps < 4000:       activity_level, act_badge = "Low Activity",      "badge-red"
elif steps < 10000:    activity_level, act_badge = "Moderate Activity",  "badge-yellow"
else:                  activity_level, act_badge = "High Activity",      "badge-green"

if productivity_score >= 7:   prod_badge, prod_color = "badge-green",  "#78e830"
elif productivity_score >= 4: prod_badge, prod_color = "badge-yellow", "#e8ff5a"
else:                         prod_badge, prod_color = "badge-red",    "#ff6060"

if activity_level == "Low Activity":      diet_type = "Light Diet"
elif activity_level == "Moderate Activity": diet_type = "Balanced Diet"
else:                                      diet_type = "High Protein Diet"

food_filtered    = food_df[(food_df["Fats (g)"] < 30) & (food_df["Protein (g)"] > 3)]
recommended_food = food_filtered.sample(3)

# ---------------------------------------------------
# Hero Banner
# ---------------------------------------------------

st.markdown("""
<div class="hero-banner">
    <h1>⚡ FeelFit</h1>
    <p>AI-Powered Wellness Intelligence — Real-Time Body Analytics</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Row 1 — 3 stat cards
# ---------------------------------------------------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="card-label">Daily Steps</div>
        <div class="card-value">{steps:,}</div>
        <div class="card-unit">steps today</div>
        <span class="card-badge {act_badge}">{activity_level}</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="card-label">Mood Score</div>
        <div class="card-value">{round(mood_score, 1)}</div>
        <div class="card-unit">out of 10</div>
        <span class="card-badge {mood_badge}">😊 {mood_label}</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    bar_width = int(productivity_score * 10)
    st.markdown(f"""
    <div class="stat-card">
        <div class="card-label">ML Productivity Score</div>
        <div class="card-value">{round(productivity_score, 1)}</div>
        <div class="card-unit">predicted by AI model</div>
        <div class="prod-bar-wrap">
            <div class="prod-bar-fill" style="width:{bar_width}%; background: linear-gradient(90deg, {prod_color}, #e8ff5a);"></div>
        </div>
        <span class="card-badge {prod_badge}">{'On Fire 🔥' if productivity_score >= 7 else 'Moderate ⚡' if productivity_score >= 4 else 'Rest Up 💤'}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="vf-divider">', unsafe_allow_html=True)

# ---------------------------------------------------
# Row 2 — Calories + Sleep + Active Minutes
# ---------------------------------------------------

st.markdown('<div class="section-label">// Energy & Recovery</div>', unsafe_allow_html=True)

e1, e2, e3, e4 = st.columns(4)

with e1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="card-label">Calories Burned</div>
        <div class="card-value">{estimated_calories}</div>
        <div class="card-unit">kcal estimated</div>
    </div>
    """, unsafe_allow_html=True)

with e2:
    sleep_pct = int((sleep_hours / 9) * 100)
    sleep_color = "#78e830" if sleep_hours >= 7 else "#ffb432" if sleep_hours >= 5 else "#ff6060"
    st.markdown(f"""
    <div class="stat-card">
        <div class="card-label">Sleep</div>
        <div class="card-value">{sleep_hours:.1f}<span style="font-size:1.2rem; color:#555"> hrs</span></div>
        <div class="prod-bar-wrap">
            <div class="prod-bar-fill" style="width:{sleep_pct}%; background:{sleep_color};"></div>
        </div>
        <div class="card-unit">{'Great recovery 🌙' if sleep_hours >= 7 else 'Needs improvement' if sleep_hours >= 5 else 'Sleep deprived ⚠️'}</div>
    </div>
    """, unsafe_allow_html=True)

with e3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="card-label">Very Active</div>
        <div class="card-value">{very_active}</div>
        <div class="card-unit">minutes of intensity</div>
    </div>
    """, unsafe_allow_html=True)

with e4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="card-label">Sedentary Time</div>
        <div class="card-value">{sedentary}</div>
        <div class="card-unit">minutes sitting</div>
        <span class="card-badge {'badge-red' if sedentary > 600 else 'badge-yellow' if sedentary > 300 else 'badge-green'}">
            {'Too much sitting' if sedentary > 600 else 'Moderate' if sedentary > 300 else 'Active day'}
        </span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="vf-divider">', unsafe_allow_html=True)

# ---------------------------------------------------
# Heart Rate Section
# ---------------------------------------------------

st.markdown('<div class="section-label">// Heart Rate Metrics</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="hr-grid">
    <div class="hr-card">
        <div class="hr-val">{user_avg_hr}</div>
        <div class="hr-label">Avg BPM</div>
    </div>
    <div class="hr-card">
        <div class="hr-val">{user_max_hr}</div>
        <div class="hr-label">Peak BPM</div>
    </div>
    <div class="hr-card">
        <div class="hr-val">{user_min_hr}</div>
        <div class="hr-label">Resting BPM</div>
    </div>
    <div class="hr-card">
        <div class="hr-val" style="color:#c8ff90;">{round(user_hr_std, 1)}</div>
        <div class="hr-label">HRV (Variability)</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="vf-divider">', unsafe_allow_html=True)

# ---------------------------------------------------
# Diet Section
# ---------------------------------------------------

st.markdown('<div class="section-label">// Nutrition Plan</div>', unsafe_allow_html=True)

d1, d2 = st.columns([1, 2])

with d1:
    st.markdown(f"""
    <div class="stat-card" style="height:100%;">
        <div class="card-label">Recommended Plan</div>
        <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:2px; color:#e8ff5a; line-height:1.2; margin-top:8px;">
            {diet_type}
        </div>
        <div class="card-unit" style="margin-top:10px;">Based on your {activity_level.lower()} profile</div>
    </div>
    """, unsafe_allow_html=True)

with d2:
    st.markdown('<div style="margin-bottom:8px; font-family:\'Space Mono\',monospace; font-size:0.6rem; letter-spacing:2px; color:#555;">SUGGESTED FOODS FOR TODAY</div>', unsafe_allow_html=True)
    for _, row in recommended_food.iterrows():
        st.markdown(f"""
        <div class="food-card">
            <div>
                <div class="food-name">{row['Dish Name']}</div>
                <div class="food-meta">Protein: {row['Protein (g)']}g &nbsp;·&nbsp; Fats: {row['Fats (g)']}g</div>
            </div>
            <div style="text-align:right;">
                <div class="food-kcal">{row['Calories (kcal)']}</div>
                <div class="food-kcal-label">kcal</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr class="vf-divider">', unsafe_allow_html=True)

# ---------------------------------------------------
# Insight Banner
# ---------------------------------------------------

st.markdown('<div class="section-label">// Daily Insight</div>', unsafe_allow_html=True)

if steps > 10000:
    st.markdown('<div class="insight-banner insight-success">🔥 &nbsp; Exceptional movement today — your activity is well above the daily target. Your body is performing optimally.</div>', unsafe_allow_html=True)
elif steps > 5000:
    st.markdown('<div class="insight-banner insight-info">⚡ &nbsp; Solid movement today. You\'re in a healthy range — a short evening walk could push you to peak performance.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="insight-banner insight-warning">⚠️ &nbsp; Activity is below target today. Even 20 minutes of movement can significantly improve your mood and energy levels.</div>', unsafe_allow_html=True)