import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. Load your survey data ----------
# Adjust filename/path if needed
df = pd.read_csv("data_Survey-Borhan_2026-02-04_19-02.csv", encoding="utf-16", sep="\t")

# Keep only completed responses
df = df[df["FINISHED"] == 1].copy()
print("Completed responses:", len(df))


# ---------- 2. Define groups of questions ----------
# A204–A212 are the 1–5 Likert items
groups = {
    "Emotional understanding":      ["A204", "A205"],
    "Confidence & transparency":    ["A206", "A207"],
    "Empathy & usefulness":         ["A208", "A209"],
    "Usability":                    ["A210"],
    "Safety":                       ["A211", "A212"],
}

# ---------- 3. Compute group mean scores ----------
group_scores = {}
for name, cols in groups.items():
    vals = df[cols].astype(float).mean(axis=1)  # average per participant
    group_scores[name] = vals.mean()            # average across participants

# Put into a table dataframe
table_df = pd.DataFrame({
    "Dimension": list(group_scores.keys()),
    "Mean (1-5)": [round(v, 2) for v in group_scores.values()]
})

print("\n=== Group scores table ===")
print(table_df)

# (Optional) save table for your report
table_df.to_csv("user_study_group_scores.csv", index=False)


# ---------- 4. Plot bar chart ----------
labels = list(group_scores.keys())
scores = [group_scores[g] for g in labels]

plt.figure(figsize=(8, 4))
plt.bar(labels, scores)
plt.ylim(0, 5)
plt.ylabel("Average Score (1-5)")
plt.title("User Study Results (N = {})".format(len(df)))
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("user_study_group_scores.png", dpi=300)  # image for slides
plt.show()
