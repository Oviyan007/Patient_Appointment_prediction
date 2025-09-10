# train.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

np.random.seed(42)
n = 2000
data = pd.DataFrame({
    "no_show_history": np.random.randint(0, 5, n),
    "attended_history": np.random.randint(1, 20, n),
    "day_of_week": np.random.randint(0, 7, n),
    "hour": np.random.choice([9, 10, 11, 14, 15, 16], n),
    "lead_time_days": np.random.randint(0, 30, n),
})

# Simulate target
data["target"] = (
    (data["no_show_history"] * 0.5 +
     (data["hour"] >= 15) * 0.3 +
     (data["lead_time_days"] < 2) * 0.2 +
     np.random.rand(n) * 0.3) > 0.6
).astype(int)

X = data.drop("target", axis=1)
y = data["target"]

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,    # number of trees
    max_depth=None,      # let trees grow fully
    random_state=42,
    n_jobs=-1            # use all CPU cores
)
rf.fit(X, y)

joblib.dump(rf, "models/noshow_model.joblib")
print("Random Forest model trained and saved")

# ploting featire
importances = rf.feature_importances_
features = X.columns
for f, imp in zip(features, importances):
    print(f"{f}: {imp:.2f}")

plt.bar(features, importances)
plt.title("Feature importance in no-show prediction")
plt.show()