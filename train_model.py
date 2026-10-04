import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

FILE = "network_data.xlsx"

# 1. Data load කරන්න
df = pd.read_excel(FILE, sheet_name="network_data").dropna()
df["Overlap"] = df["Overlap"].map({"Low": 0, "High": 1})

features = ["RSRP", "SINR", "Throughput", "Traffic_Load", "Overlap"]
X = df[features]
y = df["Problem"]

# 2. Train / Test split (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Model train කරන්න
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. Accuracy බලන්න
pred = model.predict(X_test)
print(classification_report(y_test, pred))

# 5. Feature importance
for name, score in sorted(zip(features, model.feature_importances_),
                          key=lambda t: -t[1]):
    print(f"{name:15s} {score:.3f}")

# 6. Model එක save කරන්න
joblib.dump(model, "network_model.pkl")

# 7. අලුත් data එකක් predict කරන්න
new = pd.DataFrame([{"RSRP": -108, "SINR": 3, "Throughput": 10,
                     "Traffic_Load": 88, "Overlap": 1}])
print("Prediction:", model.predict(new)[0])

# 8. Confusion matrix chart එක
ConfusionMatrixDisplay.from_predictions(y_test, pred, xticks_rotation=30)
plt.tight_layout()
plt.show()