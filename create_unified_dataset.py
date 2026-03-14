import pandas as pd
import os

# ---------------------------------------------------
# 1. Ensure data folder exists
# ---------------------------------------------------

os.makedirs("data", exist_ok=True)

# ---------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------

activity = pd.read_csv("data/dailyActivity_merged.csv")
hr = pd.read_csv("data/heartrate_seconds_merged.csv")
sleep = pd.read_csv("data/sleepDay_merged.csv")

print("Datasets loaded successfully")

# ---------------------------------------------------
# 3. Convert timestamps to date
# ---------------------------------------------------

activity["ActivityDate"] = pd.to_datetime(activity["ActivityDate"]).dt.date

hr["Time"] = pd.to_datetime(hr["Time"])
hr["date"] = hr["Time"].dt.date

sleep["SleepDay"] = pd.to_datetime(sleep["SleepDay"]).dt.date

# ---------------------------------------------------
# 4. Heart Rate Feature Engineering
# ---------------------------------------------------

hr_features = hr.groupby(["Id", "date"]).agg(

    avg_hr=("Value", "mean"),
    max_hr=("Value", "max"),
    min_hr=("Value", "min"),
    hr_std=("Value", "std")

).reset_index()

print("Heart rate features created")

# ---------------------------------------------------
# 5. Sleep Features
# ---------------------------------------------------

sleep["sleep_hours"] = sleep["TotalMinutesAsleep"] / 60

sleep_features = sleep[[
    "Id",
    "SleepDay",
    "sleep_hours",
    "TotalTimeInBed"
]]

sleep_features = sleep_features.rename(columns={
    "SleepDay": "date"
})

print("Sleep features created")

# ---------------------------------------------------
# 6. Merge Activity + Heart Rate
# ---------------------------------------------------

df = pd.merge(
    activity,
    hr_features,
    left_on=["Id", "ActivityDate"],
    right_on=["Id", "date"],
    how="left"
)

# remove duplicate date column
df.drop(columns=["date"], inplace=True)

print("Activity and HR merged")

# ---------------------------------------------------
# 7. Merge Sleep Data
# ---------------------------------------------------

df = pd.merge(
    df,
    sleep_features,
    left_on=["Id", "ActivityDate"],
    right_on=["Id", "date"],
    how="left"
)

df.drop(columns=["date"], inplace=True)

print("Sleep merged")

# ---------------------------------------------------
# 8. Handle Missing Values
# ---------------------------------------------------

df["hr_std"] = df["hr_std"].fillna(df["hr_std"].mean())
df["sleep_hours"] = df["sleep_hours"].fillna(df["sleep_hours"].mean())

# ---------------------------------------------------
# 9. Mood Score
# ---------------------------------------------------

df["mood_score"] = (
    0.4 * (df["sleep_hours"] / 8)
    + 0.3 * (df["TotalSteps"] / 12000)
    + 0.3 * (1 / (1 + df["hr_std"]))
)

df["mood_score"] = (df["mood_score"] * 10).clip(0, 10)

# ---------------------------------------------------
# 10. Productivity Score
# ---------------------------------------------------

df["productivity_score"] = (
    0.5 * (df["VeryActiveMinutes"] + df["FairlyActiveMinutes"]) / 90
    + 0.3 * (df["sleep_hours"] / 8)
    + 0.2 * (1 - df["SedentaryMinutes"] / 1440)
)

df["productivity_score"] = (df["productivity_score"] * 10).clip(0, 10)

# ---------------------------------------------------
# 11. Select Final Columns
# ---------------------------------------------------

final_df = df[[
    "Id",
    "ActivityDate",
    "TotalSteps",
    "VeryActiveMinutes",
    "FairlyActiveMinutes",
    "LightlyActiveMinutes",
    "SedentaryMinutes",
    "Calories",
    "avg_hr",
    "max_hr",
    "min_hr",
    "hr_std",
    "sleep_hours",
    "mood_score",
    "productivity_score"
]]

# ---------------------------------------------------
# 12. Save Unified Dataset
# ---------------------------------------------------

output_path = "data/unified_wellness_dataset.csv"

final_df.to_csv(output_path, index=False)

print("\nUnified dataset created successfully!")
print("Saved at:", output_path)

print("\nDataset Shape:", final_df.shape)

print("\nSample Data:")
print(final_df.head())