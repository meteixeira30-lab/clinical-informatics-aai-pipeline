import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set visual style for corporate/healthcare reporting
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"

# 1. SQL-like Clinical Data Simulation
# Simulating a dataset extracted via SQL query from an Electronic Health Record (EHR) system
np.random.seed(42)
n_patients = 60

clinical_data = {
    "patient_id": range(101, 101 + n_patients),
    "age": np.random.randint(4, 12, size=n_patients),
    "diagnosis": np.random.choice(
        ["Cerebral Palsy", "Autism Spectrum Disorder (ASD)", "Down Syndrome"],
        size=n_patients,
        p=[0.5, 0.3, 0.2],
    ),  # Based on highly prevalent demographics in Latin American AAI literature[cite: 2]
    "co_therapist_species": np.random.choice(
        ["Equine", "Canine"], size=n_patients, p=[0.7, 0.3]
    ),  # Replicating the real proportion found in historical scoping data[cite: 2]
    "sessions_completed": np.random.randint(12, 24, size=n_patients),
}

df_clinical = pd.DataFrame(clinical_data)

# 2. Advanced Metrics Processing (Python Layer)
# Simulating longitudinal biomedical sensor telemetry (Baseline vs Endline)
# Activating continuous variables to measure stress reduction via Heart Rate Variability (HRV)[cite: 2]


def simulate_hrv_outcomes(row):
    # Simulating a physiological regulatory response triggered by animal interaction[cite: 2]
    baseline_hrv = np.random.normal(50, 5)

    # Simulating clinical gains mediated by session engagement levels
    treatment_effect = (
        row["sessions_completed"] * 0.4
        if row["co_therapist_species"] == "Equine"
        else row["sessions_completed"] * 0.3
    )

    endline_hrv = baseline_hrv + treatment_effect + np.random.normal(2, 2)
    return pd.Series([baseline_hrv, endline_hrv])


# Applying the function to inject biostatistical variables into the clinical pipeline
df_clinical[["hrv_baseline", "hrv_endline"]] = df_clinical.apply(
    simulate_hrv_outcomes, axis=1
)

# 3. Biostatistics Layer: Evaluating Statistical Significance
# Overcoming historical literature limitations regarding the lack of quantitative validation[cite: 2]
t_stat, p_value = stats.ttest_rel(
    df_clinical["hrv_endline"], df_clinical["hrv_baseline"]
)

# 4. Data Visualization Layer (Healthcare BI Insights)
# Restructuring data into a long-form format for advanced statistical plotting
df_melted = pd.melt(
    df_clinical,
    id_vars=["patient_id", "diagnosis", "co_therapist_species"],
    value_vars=["hrv_baseline", "hrv_endline"],
    var_name="Timeline",
    value_name="HRV (ms)",
)

df_melted["Timeline"] = df_melted["Timeline"].map(
    {"hrv_baseline": "Baseline", "hrv_endline": "Endline"}
)

# Initialize the matplotlib figure
fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]})
fig.suptitle(
    "Clinical Data Visualization: HRV Stress Recovery Pipeline via AAI",
    fontsize=16,
    fontweight="bold",
    color="#1a252c",
)

# Chart 1: Boxenplot showing overall clinical progression & distribution
sns.boxenplot(
    data=df_melted,
    x="Timeline",
    y="HRV (ms)",
    hue="co_therapist_species",
    palette=["#2b5c8f", "#4682b4"],
    ax=axes[0],
)
axes[0].set_title(
    f"Physiological Progression (Paired t-test p-value: {p_value:.5f})",
    fontsize=12,
    fontweight="semibold",
)
axes[0].set_xlabel("Clinical Evaluation Timeline", fontsize=10)
axes[0].set_ylabel("Heart Rate Variability (ms)", fontsize=10)
axes[0].legend(title="Co-Therapist")

# Chart 2: Linear correlation between completed sessions and endline HRV improvements
sns.regplot(
    data=df_clinical,
    x="sessions_completed",
    y="hrv_endline",
    scatter_kws={"alpha": 0.6, "color": "#2b5c8f"},
    line_kws={"color": "#e74c3c", "fontweight": "bold"},
    ax=axes[1],
)
axes[1].set_title(
    "Dose-Response Effect: Dose of Sessions vs. Physiological Output",
    fontsize=12,
    fontweight="semibold",
)
axes[1].set_xlabel("Number of Therapy Sessions Completed", fontsize=10)
axes[1].set_ylabel("Endline HRV Assessment (ms)", fontsize=10)

plt.tight_layout()

# Save the dashboard plot for GitHub repository display
plt.savefig("aai_healthcare_dashboard.png", dpi=300)
plt.show()

# Printing pipeline execution results
print("--- HEALTHCARE CLINICAL DATA PIPELINE OUTCOMES ---")
print(f"Total Cohort Analysed: {len(df_clinical)} patients")
print(f"Paired t-test Statistics: t = {t_stat:.3f}, p-value = {p_value:.5f}")
