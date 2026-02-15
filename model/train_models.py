import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ===============================
# 1. READ CSV (same format as app upload)
# ===============================

# Put your training CSV inside project root
DATA_FILE = "training_data.csv"

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"{DATA_FILE} not found in project root folder"
    )

data = pd.read_csv(DATA_FILE)

# ===============================
# 2. SPLIT FEATURES & TARGET
# ===============================

if "target" not in data.columns:
    raise ValueError("CSV must contain a 'target' column")

X = data.drop("target", axis=1)
y = data["target"]

feature_names = X.columns.tolist()

# ===============================
# 3. TRAIN TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. SCALING
# ===============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# ===============================
# 5. MODELS
# ===============================

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}

os.makedirs("model", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.pkl")

# Save scaler & feature names
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_names, "model/feature_names.pkl")

print("Models trained successfully!")
