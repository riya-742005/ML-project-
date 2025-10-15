# ======================================================
# 🏏 Streamlit T20 Cricket Score Predictor Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="🏏 T20 Score Predictor", layout="wide")

st.title("🏏 T20 Cricket Score Predictor")

# ----------------------
# Load dataset
# ----------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

# ----------------------
# Detect target
# ----------------------
target_candidates = ["score", "runs", "total_runs", "predicted_score", "target"]
target = None
for t in target_candidates:
    matches = [c for c in df.columns if t in c.lower()]
    if matches:
        target = matches[0]
        break
if target is None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    target = max(numeric_cols, key=lambda c: df[c].std())

df[target] = pd.to_numeric(df[target], errors="coerce")
df = df[df[target].notna()].reset_index(drop=True)

y = df[target]

# ----------------------
# Feature selection
# ----------------------
drop_cols = ["id", "match_id", "player_id", "date", "time"]
features = [c for c in df.columns if c != target and c not in drop_cols]

numeric_feats = df[features].select_dtypes(include=[np.number]).columns.tolist()
cat_feats = [c for c in features if c not in numeric_feats]
safe_cat_feats = [c for c in cat_feats if df[c].nunique() <= 50]
final_feats = numeric_feats + safe_cat_feats
X = df[final_feats]

# ----------------------
# Preprocessing + Model
# ----------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_feats),
    ("cat", categorical_transformer, safe_cat_feats)
])

xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
pipeline = Pipeline([
    ("pre", preprocessor),
    ("xgb", xgb)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred)**2))

# ----------------------
# Streamlit Inputs
# ----------------------
st.sidebar.header("Match Inputs")
overs = st.sidebar.number_input("Overs played (1-20)", min_value=1.0, max_value=20.0, value=10.0)
wickets = st.sidebar.number_input("Wickets lost (0-10)", min_value=0, max_value=10, value=2)
run_rate = st.sidebar.number_input("Current Run Rate", min_value=0.0, value=8.0)
opponent_strength = st.sidebar.slider("Opponent Strength (1-10)", 1, 10, 5)
home_away = st.sidebar.selectbox("Home/Away", ["Home", "Away"])
pitch = st.sidebar.selectbox("Pitch Condition", ["Batting-friendly", "Bowling-friendly", "Balanced"])
weather = st.sidebar.selectbox("Weather", ["Sunny", "Cloudy", "Overcast"])

input_df = pd.DataFrame({
    "Overs Played": [overs],
    "Wickets Lost": [wickets],
    "Run Rate": [run_rate],
    "Opponent Strength": [opponent_strength],
    "Home/Away": [home_away],
    "Pitch Condition": [pitch],
    "Weather": [weather]
})

pred_score = pipeline.predict(input_df)[0]
st.success(f"🏆 Predicted Final T20 Score: **{pred_score:.1f} runs**")

# ----------------------
# Visual Dashboard
# ----------------------
st.subheader("📊 Dashboard Visuals")

# Run Meter
st.bar_chart(pd.DataFrame({"Predicted Score": [pred_score]}))

# Confidence Range
sim_range = np.linspace(pred_score - rmse, pred_score + rmse, 200)
fig, ax = plt.subplots()
sns.kdeplot(sim_range, fill=True, color="gold", alpha=0.6, ax=ax)
ax.axvline(pred_score, color="green", linestyle="--", label="Predicted Score")
ax.axvspan(pred_score - rmse, pred_score + rmse, color="lightgreen", alpha=0.3, label="±RMSE")
ax.set_title("🎯 Confidence Range")
ax.set_xlabel("Possible Scores")
ax.legend()
st.pyplot(fig)

# Projected Run Curve
overs_list = np.arange(overs, 21)
avg_rpo = pred_score / 20
curve = [run_rate * overs] + [avg_rpo * o for o in range(int(overs)+1, 21)]
fig2, ax2 = plt.subplots()
ax2.plot(range(1, len(curve)+1), curve, color="orange", marker="o")
ax2.set_title("📈 Projected Run Curve")
ax2.set_xlabel("Overs")
ax2.set_ylabel("Runs")
st.pyplot(fig2)

# ----------------------
# AI Commentary
# ----------------------
st.subheader("🎤 Match Insights")

def commentary(pred):
    if pred < 100:
        tone = "😬 Struggling innings! The batting side is in trouble."
    elif pred < 150:
        tone = "⚖️ Balanced play — acceleration needed."
    elif pred < 190:
        tone = "🔥 Strong innings! Competitive total expected."
    else:
        tone = "💥 Explosive batting show incoming!"

    return f"""
🏆 Projected Total: **{pred:.1f} runs**
📊 Current Run Rate: {run_rate} | Required Momentum: ~{round((pred / 20), 1)} RPO
🌦️ Pitch: {pitch} | Weather: {weather} | Opponent Strength: {opponent_strength}/10
⚾ Wickets Lost: {wickets} | Overs Played: {overs}

🎤 Commentary: {tone}
"""

st.markdown(commentary(pred_score))






