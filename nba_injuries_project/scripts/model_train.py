import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
from predict_ppg import predict_post_injury_ppg_from_raw
import numpy as np
import matplotlib.pyplot as plt




BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset.csv")

# Load dataset
dataset = pd.read_csv(DATA_PATH)

# Feature engineering
dataset["pre_PPM"] = (dataset["pre_PTS"] / dataset["pre_MP"]).where(dataset["pre_MP"] > 0, 0)
dataset["pre_FGA_per_min"] = (dataset["pre_FGA"] / dataset["pre_MP"]).where(dataset["pre_MP"] > 0, 0)
dataset["pre_3PA_rate"] = (dataset["pre_3PA"] / dataset["pre_FGA"]).where(dataset["pre_FGA"] > 0, 0)
dataset["pre_FT_rate"] = (dataset["pre_FTA"] / dataset["pre_FGA"]).where(dataset["pre_FGA"] > 0, 0)
dataset["pre_Age_squared"] = dataset["pre_Age"] ** 2
dataset["pre_FG_pct"] = (dataset["pre_FG"] / dataset["pre_FGA"]).where(dataset["pre_FGA"] > 0, 0)
dataset["pre_3P_pct"] = (dataset["pre_3P"] / dataset["pre_3PA"]).where(dataset["pre_3PA"] > 0, 0)
dataset["pre_FT_pct"] = (dataset["pre_FT"] / dataset["pre_FTA"]).where(dataset["pre_FTA"] > 0, 0)
dataset["pre_TS_pct"] = (dataset["pre_PTS"] / (2 * (dataset["pre_FGA"] + 0.44 * dataset["pre_FTA"]))).where((dataset["pre_FGA"] + dataset["pre_FTA"]) > 0, 0)
dataset["pre_Usage_proxy"] = ((dataset["pre_FGA"] + 0.44 * dataset["pre_FTA"] + dataset["pre_TOV"]) / dataset["pre_MP"]).where(dataset["pre_MP"] > 0, 0)
dataset["pre_AST_per_min"] = (dataset["pre_AST"] / dataset["pre_MP"]).where(dataset["pre_MP"] > 0, 0)
dataset["pre_TRB_per_min"] = (dataset["pre_TRB"] / dataset["pre_MP"]).where(dataset["pre_MP"] > 0, 0)

#Age bucket
def age_bucket(age):
    if age < 23:
        return 0
    elif age <= 28: 
        return 1
    elif age <= 32: 
        return 2
    else: 
        return 3

dataset["pre_age_bucket"] = dataset["pre_Age"].apply(age_bucket)

dataset_filtered = dataset[dataset["pre_PTS"].notna()]

# Features and label
features = ["pre_PTS", "pre_AST", "pre_TRB", "pre_MP", "pre_Age",
            "pre_PPM", "pre_FGA_per_min", "pre_3PA_rate", "pre_FT_rate",
            "pre_Age_squared", "pre_FG_pct", "pre_3P_pct", "pre_FT_pct",
            "pre_TS_pct", "pre_Usage_proxy", "pre_AST_per_min",
            "pre_TRB_per_min", "pre_age_bucket"]

label = "post_PTS"

data = dataset_filtered.dropna(subset=features + [label])
X = data[features]
y = data[label]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest + GridSearch
rf = RandomForestRegressor(random_state=42)
param_grid = {"n_estimators":[100,200], "max_depth":[None,10], "min_samples_leaf":[1,2]}

grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)
print("Training R2:", grid_search.best_score_)

# Test evaluation
y_pred = model.predict(X_test)
print("Test R2:", r2_score(y_test, y_pred))
print("Test MSE:", mean_squared_error(y_test, y_pred))

# Save model
joblib.dump(model, os.path.join(BASE_DIR, "trained_rf_model.pkl"))
print("✅ Model saved as trained_rf_model.pkl")


'''

#random test from 20% unseen data
test_indices = X_test.index.tolist()
random_index = random.choice(test_indices)
selected_row = dataset_filtered.loc[random_index]

print("\nTesting on unseen player from 20% test set")
print(f"Player: {selected_row['player']}")
print(f"Date of injury: {selected_row['date']}")
print(f"Actual Post-Injury PPG: {selected_row['post_PTS']:.1f}")

# Create raw_stats dict
raw_stats = {
    "PTS": selected_row["pre_PTS"],
    "AST": selected_row["pre_AST"],
    "TRB": selected_row["pre_TRB"],
    "MP": selected_row["pre_MP"],
    "FGA": selected_row["pre_FGA"],
    "FG": selected_row["pre_FG"],
    "3PA": selected_row["pre_3PA"],
    "3P": selected_row["pre_3P"],
    "FTA": selected_row["pre_FTA"],
    "FT": selected_row["pre_FT"],
    "TOV": selected_row["pre_TOV"],
    "Age": selected_row["pre_Age"]
}

# Predict
predicted_ppg = predict_post_injury_ppg_from_raw(model, raw_stats)
print(f"Predicted Post-Injury PPG: {predicted_ppg:.1f}")


#Test: Kobe 2013-04-12 ACL injury (unseen)
kobe_row = dataset_filtered[(dataset_filtered['player'] == 'Kobe Bryant') &(dataset_filtered['date'] == '2013-04-12')]

if kobe_row.empty:
    print("kobe 2013-04-12 injury not found in dataset_filtered")
else:
    kobe_row = kobe_row.iloc[0] 

    print("\nTesting on Kobe 2013-04-12 ACL injury (unseen)")
    print(f"Player: {kobe_row['player']}")
    print(f"Date of injury: {kobe_row['date']}")
    print(f"Actual Post-Injury PPG: {kobe_row['post_PTS']:.1f}")

    # Create raw_stats dict
    raw_stats = {
        "PTS": kobe_row["pre_PTS"],
        "AST": kobe_row["pre_AST"],
        "TRB": kobe_row["pre_TRB"],
        "MP": kobe_row["pre_MP"],
        "FGA": kobe_row["pre_FGA"],
        "FG": kobe_row["pre_FG"],
        "3PA": kobe_row["pre_3PA"],
        "3P": kobe_row["pre_3P"],
        "FTA": kobe_row["pre_FTA"],
        "FT": kobe_row["pre_FT"],
        "TOV": kobe_row["pre_TOV"],
        "Age": kobe_row["pre_Age"]
    }

    # Predict
    predicted_ppg = predict_post_injury_ppg_from_raw(model, raw_stats)
    print(f"Predicted Post-Injury PPG: {predicted_ppg:.1f}")


'''

#Evaluate model on full 20% unseen test set
X_test_values = X_test.copy()
y_test_values = y_test.copy()

# Predict
y_pred_test = model.predict(X_test_values)

r2 = r2_score(y_test_values, y_pred_test)

mse = mean_squared_error(y_test_values, y_pred_test)
rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test_values, y_pred_test)

print("\nModel performance on 20% unseen test set")
print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")


# Scatter Plot: Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test_values, y_pred_test, alpha=0.6, edgecolors="k", label="Predicted Vs. Actual")
max_val = max(max(y_test_values), max(y_pred_test))
plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
plt.xlabel("Actual Post-Injury PPG")
plt.ylabel("Predicted Post-Injury PPG")
plt.title("Predicted vs Actual Post-Injury PPG")
plt.legend()
plt.grid(True)
plt.show()

# Residuals vs Predicted
residuals = y_test_values.to_numpy() - y_pred_test  # y_test_values is pandas Series

plt.figure(figsize=(8,6))
plt.scatter(y_pred_test, residuals, alpha=0.6, edgecolors="k")
plt.axhline(0, color='r', linestyle="--")
plt.xlabel("Predicted Post-Injury PPG")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs Predicted PPG")
plt.grid(True)

# Annotate top 5 largest residuals
abs_residuals = np.abs(residuals)
top5_pos = np.argsort(-abs_residuals)[:5]

for pos in top5_pos:
    dataset_idx = X_test.iloc[pos].name  # original index in dataset_filtered
    player_name = dataset_filtered.loc[dataset_idx, "player"]
    injury_date = dataset_filtered.loc[dataset_idx, "date"]
    
    plt.annotate(f"{player_name}\n{injury_date}",
                 (y_pred_test[pos], residuals[pos]),
                 textcoords="offset points", xytext=(5,5), ha='left', fontsize=9)

plt.show()


#Predicted vs. actual scatter plot graph (what it tells us): predicted post injury ppg (y) and actual post injury ppg (x). The red dashed line is the perfect prediction line. it's equation is y = x, Any point on this line means the model predicted that player’s post-injury PPG exactly correctly.
# - points above this line - model overpredicted (predicted>actual)
# - points below the line - model underpredicted (predicted<actual)

#Residuals vs predicted ppg plot: residual = actual ppg - predicted ppg. The horizontal line at zero represents the perfect predictions. residual > 0 - model underpredicted, <0, overpredicted. 
