import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("data/unified_wellness_dataset.csv")


# -----------------------------
# Features
# -----------------------------

X = df[[
"VeryActiveMinutes",
"FairlyActiveMinutes",
"SedentaryMinutes",
"avg_hr"
]]

# Target

y = df["mood_score"]


# -----------------------------
# Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)


# -----------------------------
# Train model
# -----------------------------

model = RandomForestRegressor(
n_estimators=100,
random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Evaluation
# -----------------------------

r2 = r2_score(y_test, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("R2 Score:", r2)

print("RMSE:", rmse)


# -----------------------------
# Cross Validation
# -----------------------------

cv_scores = cross_val_score(
model,
X,
y,
cv=5,
scoring="r2"
)

print("Cross Validation R2 Scores:", cv_scores)

print("Average CV R2:", cv_scores.mean())
