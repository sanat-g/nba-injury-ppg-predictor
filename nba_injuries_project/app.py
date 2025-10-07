import streamlit as st
import pandas as pd
import os
import sys
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
from predict_ppg import predict_post_injury_ppg_from_raw

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "scripts", "trained_rf_model.pkl")

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="NBA Post-Injury PPG Predictor", layout="centered")
st.title("🏀 NBA Post-Injury PPG Predictor")
st.write("Enter pre-injury stats to estimate a player’s expected **post-injury PPG**.")


col1, col2 = st.columns(2)

with col1:
    pts = st.number_input("PTS", value=0.0)
    ast = st.number_input("AST", value=0.0)
    trb = st.number_input("TRB", value=0.0)
    mp = st.number_input("MP", value=0.0)
    fga = st.number_input("FGA", value=0.0)
    fg = st.number_input("FG", value=0.0)

with col2:
    three_pa = st.number_input("3PA", value=0.0)
    three_p = st.number_input("3P", value=0.0)
    fta = st.number_input("FTA", value=0.0)
    ft = st.number_input("FT", value=0.0)
    tov = st.number_input("TOV", value=0.0)
    age = st.number_input("Age", value=20.0)

raw_stats_future = {
    "PTS": pts,
    "AST": ast,
    "TRB": trb,
    "MP": mp,
    "FGA": fga,
    "FG": fg,
    "3PA": three_pa,
    "3P": three_p,
    "FTA": fta,
    "FT": ft,
    "TOV": tov,
    "Age": age,
}

st.markdown("###")
col_left, col_center, col_right = st.columns([1,2,1])

with col_center:
    if st.button("🔮 Predict PPG", use_container_width=True):
        predicted_ppg = predict_post_injury_ppg_from_raw(model, raw_stats_future)
        st.success(f"Predicted Post-Injury PPG: **{predicted_ppg:.1f}**")
