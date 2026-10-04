import pandas as pd
import matplotlib.pyplot as plt

# ---------- උපකල්පන (assumed effect of each action) ----------
# මේවා simulation සඳහා දාපු අගයන්. Real network එකේ test කරලා තමයි ඇත්ත අගය හොයාගන්න ඕන.
EFFECTS = {
    "Poor_Coverage": {"RSRP": +4,  "SINR": +1, "Throughput_x": 1.25, "Traffic_Load": 0},
    "Interference":  {"RSRP": 0,   "SINR": +5, "Throughput_x": 1.30, "Traffic_Load": 0},
    "Congestion":    {"RSRP": 0,   "SINR": 0,  "Throughput_x": 1.40, "Traffic_Load": -20},
}

df = pd.read_excel("optimization_results.xlsx")
after = df.copy()

for problem, e in EFFECTS.items():
    m = (df["Predicted_Problem"] == problem) & (df["Priority"].isin(["High", "Medium"]))
    after.loc[m, "RSRP"] = df.loc[m, "RSRP"] + e["RSRP"]
    after.loc[m, "SINR"] = df.loc[m, "SINR"] + e["SINR"]
    after.loc[m, "Throughput"] = (df.loc[m, "Throughput"] * e["Throughput_x"]).round(1)
    after.loc[m, "Traffic_Load"] = (df.loc[m, "Traffic_Load"] + e["Traffic_Load"]).clip(lower=0)

cols = ["RSRP", "SINR", "Throughput", "Traffic_Load"]
before_mean = df.groupby("Predicted_Problem")[cols].mean().round(1)
after_mean = after.groupby("Predicted_Problem")[cols].mean().round(1)

summary = before_mean.join(after_mean, lsuffix="_before", rsuffix="_after")
print(summary.T)

overall_b = df["Throughput"].mean()
overall_a = after["Throughput"].mean()
print(f"\nOverall avg Throughput: {overall_b:.1f} -> {overall_a:.1f} Mbps "
      f"({(overall_a/overall_b-1)*100:+.1f}%)")

# ---------- Chart ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, col in zip(axes, ["Throughput", "SINR", "Traffic_Load"]):
    x = range(len(before_mean))
    ax.bar([i - 0.2 for i in x], before_mean[col], width=0.4, label="Before")
    ax.bar([i + 0.2 for i in x], after_mean[col], width=0.4, label="After")
    ax.set_xticks(list(x))
    ax.set_xticklabels(before_mean.index, rotation=25)
    ax.set_title(col)
    ax.legend()
fig.suptitle("Before vs After optimization (SIMULATED)")
plt.tight_layout()
plt.savefig("before_after.png", dpi=150)

summary.to_excel("before_after_summary.xlsx")
print("\nSaved: before_after.png, before_after_summary.xlsx")
plt.show()