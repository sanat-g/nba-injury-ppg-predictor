import pandas as pd
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_rf_model.pkl")


def predict_post_injury_ppg_from_raw(model, raw_stats):
    stats = raw_stats.copy()
    
    MP = stats["MP"]
    FGA = stats["FGA"]
    FTA = stats["FTA"]
    FGA_safe = FGA if FGA > 0 else 1
    MP_safe = MP if MP > 0 else 1 

    # Derived features
    stats["pre_PTS"] = stats["PTS"]
    stats["pre_AST"] = stats["AST"]
    stats["pre_TRB"] = stats["TRB"]
    stats["pre_MP"] = MP
    stats["pre_Age"] = stats["Age"]
    
    stats["pre_PPM"] = stats["PTS"] / MP_safe
    stats["pre_FGA_per_min"] = FGA / MP_safe
    stats["pre_3PA_rate"] = stats["3PA"] / FGA_safe
    stats["pre_FT_rate"] = FTA / FGA_safe
    stats["pre_Age_squared"] = stats["Age"] ** 2
    
    stats["pre_FG_pct"] = stats["FG"] / FGA_safe
    stats["pre_3P_pct"] = stats["3P"] / stats["3PA"] if stats["3PA"] > 0 else 0
    stats["pre_FT_pct"] = stats["FT"] / FTA if FTA > 0 else 0
    
    stats["pre_TS_pct"] = stats["PTS"] / (2 * (FGA + 0.44 * FTA)) if (FGA + FTA) > 0 else 0
    stats["pre_Usage_proxy"] = (FGA + 0.44 * FTA + stats["TOV"]) / MP_safe
    
    stats["pre_AST_per_min"] = stats["AST"] / MP_safe
    stats["pre_TRB_per_min"] = stats["TRB"] / MP_safe

    # Age bucket
    age = stats["Age"]
    if age < 23:
        stats["pre_age_bucket"] = 0
    elif age <= 28:
        stats["pre_age_bucket"] = 1
    elif age <= 32:
        stats["pre_age_bucket"] = 2
    else:
        stats["pre_age_bucket"] = 3
    
    # Select features used by model
    features = [
        "pre_PTS", "pre_AST", "pre_TRB", "pre_MP", "pre_Age",
        "pre_PPM", "pre_FGA_per_min", "pre_3PA_rate", "pre_FT_rate",
        "pre_Age_squared", "pre_FG_pct", "pre_3P_pct", "pre_FT_pct",
        "pre_TS_pct", "pre_Usage_proxy", "pre_AST_per_min",
        "pre_TRB_per_min", "pre_age_bucket"
    ]

    df = pd.DataFrame([{f: stats[f] for f in features}])
    predicted_ppg = model.predict(df)[0]

    return predicted_ppg
