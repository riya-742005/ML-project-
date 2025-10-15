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
import joblib
import os

st.set_page_config(page_title="🏏 T20 Score Predictor", layout="wide")
st.title("🏏 T20 Cricket Score Predictor")

# Initialize
pipeline = None
final_feats = []

# ----------------------
# Load or train model
# ----------------------
MODEL_PATH = "t20_pipeline.joblib"
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
    transformers = pipeline.named_steps['pre'].transformers_
    final_feats = transformers[0][2] + transformers[1][2]
else:
    st.warning("No trained model found. Please upload dataset to train.")

uploaded_file = st.file_uploader("Upload CSV to train model", type="csv")
if uploaded_file and not os.path.exists(MODEL_PATH):
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
    # Features
    # ----------------------
    drop_cols = ["id", "match_id", "player_id", "date", "time"]
    features = [c for c in df.columns if c != target and c not in drop_cols]
    numeric_feats = df[features].select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = [c for c in features if c not in numeric_feats]
    safe_cat_feats = [c for c in cat_feats if df[c].nunique() <= 50]
    final_feats = numeric_feats + safe_cat_feats
    X = df[final_feats]

    # ----------------------
    # Preprocessor + model
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
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODEL_PATH)
    st.success("✅ Model trained and saved.")

# ----------------------
# Centered Inputs & Prediction
# ----------------------
if pipeline:
    st.subheader("🏏 Enter Match Details")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        overs = st.number_input("Overs played (1-20)", min_value=1.0, max_value=20.0, value=10.0)
        wickets = st.number_input("Wickets lost (0-10)", min_value=0, max_value=10, value=2)
        run_rate = st.number_input("Current Run Rate", min_value=0.0, value=8.0)
        opponent_strength = st.slider("Opponent Strength (1-10)", 1, 10, 5)
        home_away = st.selectbox("Home/Away", ["Home", "Away"])
        pitch = st.selectbox("Pitch Condition", ["Batting-friendly", "Bowling-friendly", "Balanced"])
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Overcast"])

        input_df = pd.DataFrame({
            "Overs Played": [overs],
            "Wickets Lost": [wickets],
            "Run Rate": [run_rate],
            "Opponent Strength": [opponent_strength],
            "Home/Away": [home_away],
            "Pitch Condition": [pitch],
            "Weather": [weather]
        })
        # Fill missing columns
        for col in final_feats:
            if col not in input_df.columns:
                input_df[col] = 0

        pred_score = pipeline.predict(input_df)[0]

        st.markdown(f"<h2 style='text-align: center;'>🏆 Predicted Final T20 Score: {pred_score:.1f} runs</h2>", unsafe_allow_html=True)

        # ----------------------
        # Fast Dashboard Visuals
        # ----------------------
        st.subheader("📊 Dashboard Visuals")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            # Run meter
            st.bar_chart(pd.DataFrame({"Predicted Score": [pred_score]}))

            # Confidence Range
            y_pred = pipeline.predict(pd.DataFrame(X)) if 'X' in locals() else np.array([pred_score])
            rmse = np.sqrt(np.mean((y_pred - np.mean(y_pred))**2))
            sim_range = np.linspace(pred_score - rmse, pred_score + rmse, 100)
            fig, ax = plt.subplots()
            sns.kdeplot(sim_range, fill=True, color="gold", alpha=0.6, ax=ax)
            ax.axvline(pred_score, color="green", linestyle="--", label="Predicted Score")
            st.pyplot(fig)

            # Projected Run Curve
            overs_list = np.arange(int(overs), 21)
            avg_rpo = pred_score / 20
            curve = [run_rate * overs] + [avg_rpo * o for o in range(int(overs)+1, 21)]
            fig2, ax2 = plt.subplots()
            ax2.plot(range(1, len(curve)+1), curve, color="orange", marker="o")
            ax2.set_xlabel("Overs")
            ax2.set_ylabel("Runs")
            st.pyplot(fig2)

