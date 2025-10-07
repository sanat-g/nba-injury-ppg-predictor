import pandas as pd
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Load raw data
inj = pd.read_csv(os.path.join(BASE_DIR, "data", "injuries.csv"))
stats = pd.read_csv(os.path.join(BASE_DIR, "data", "stats.csv"))

inj = inj.rename(columns={
    "Date": "date",
    "Team": "team",
    "Relinquished": "player",
    "Notes": "injury"
})

# Convert dates and compute season_start (start year of NBA season)
inj["date"] = pd.to_datetime(inj["date"], errors="coerce")
inj["season_start"] = inj["date"].apply(lambda d: d.year if d.month >= 8 else d.year - 1)

# Pre/post season for merging
inj["pre_season_start"] = inj["season_start"] - 1
inj["post_season_start"] = inj["season_start"] + 1

def normalize_name(name):
    return re.sub(r'[^a-z]', '', str(name).lower())

inj["player_normal"] = inj["player"].apply(normalize_name)

# Clean stats: normalize names and derive season_start from "Year" column
stats["Player"] = stats["Player"].astype(str)
stats["player_normal"] = stats["Player"].apply(normalize_name)
stats["season_start"] = stats["Year"].apply(lambda x: int(str(x).split("-")[0]))

# Prepare pre/post stats for merging
pre_stats = stats.copy().add_prefix("pre_")
post_stats = stats.copy().add_prefix("post_")

# Merge pre-injury stats
dataset = inj.merge(
    pre_stats,
    left_on=["player_normal", "pre_season_start"],
    right_on=["pre_player_normal", "pre_season_start"],
    how="left"
)

# Merge post-injury stats
dataset = dataset.merge(
    post_stats,
    left_on=["player_normal", "post_season_start"],
    right_on=["post_player_normal", "post_season_start"],
    how="left"
)

out_path = os.path.join(BASE_DIR, "data", "clean_dataset.csv")
dataset.to_csv(out_path, index=False)
print(f"Saved cleaned dataset to {out_path}")
