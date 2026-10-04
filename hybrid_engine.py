import pandas as pd
import joblib

# ---------- Settings (මෙතන අගයන් වෙනස් කරන්න පුළුවන්) ----------
CONFIDENCE_MIN = 0.60      # මීට අඩු confidence නම් manual check
RSRP_VERY_WEAK = -115      # dBm
SINR_CRITICAL = 0          # dB
LOAD_CRITICAL = 90         # %

FEATURES = ["RSRP", "SINR", "Throughput", "Traffic_Load", "Overlap"]
model = joblib.load("network_model.pkl")

# ---------- Rule-based part ----------
def select_action(row, problem):
    """return (action, priority)"""
    if problem == "Good":
        return "No action needed - keep monitoring", "None"

    if problem == "Poor_Coverage":
        if row["RSRP"] < RSRP_VERY_WEAK:
            return "Increase Tx power and uptilt antenna by 1-2 degrees", "High"
        return "Uptilt antenna by 1 degree; check for nearby obstruction", "Medium"

    if problem == "Interference":
        if row["Overlap"] == "High":
            action = "Increase downtilt by 1-2 degrees and reduce Tx power on overlapping cell"
        else:
            action = "Check PCI / frequency plan and external interference sources"
        return action, ("High" if row["SINR"] < SINR_CRITICAL else "Medium")

    if problem == "Congestion":
        if row["Overlap"] == "High":
            action = "Load balancing: adjust CIO to offload users to neighbour cell"
        else:
            action = "No neighbour to offload: add carrier / capacity expansion"
        return action, ("High" if row["Traffic_Load"] > LOAD_CRITICAL else "Medium")

    return "Unknown problem - manual check", "Low"

# ---------- Hybrid engine ----------
def run_engine(input_file, output_file):
    df = pd.read_excel(input_file, sheet_name="network_data").dropna(subset=FEATURES)

    X = df[FEATURES].copy()
    X["Overlap"] = X["Overlap"].map({"Low": 0, "High": 1})

    proba = model.predict_proba(X)
    df["Predicted_Problem"] = model.classes_[proba.argmax(axis=1)]
    df["Confidence"] = proba.max(axis=1).round(2)

    actions, priorities = [], []
    for _, r in df.iterrows():
        if r["Confidence"] < CONFIDENCE_MIN:
            a, p = "Low confidence - manual engineer check", "Review"
        else:
            a, p = select_action(r, r["Predicted_Problem"])
        actions.append(a); priorities.append(p)
    df["Action"] = actions
    df["Priority"] = priorities

    # High priority මුලින්ම පේන්න sort කරනවා
    order = {"High": 0, "Medium": 1, "Review": 2, "Low": 3, "None": 4}
    df = df.sort_values("Priority", key=lambda s: s.map(order))
    df.to_excel(output_file, index=False)
    return df

if __name__ == "__main__":
    out = run_engine("network_data.xlsx", "optimization_results.xlsx")
    print(out[["Predicted_Problem", "Confidence", "Priority"]].head(8))
    print("\nPriority summary:")
    print(out["Priority"].value_counts())
    print("\nSaved: optimization_results.xlsx")