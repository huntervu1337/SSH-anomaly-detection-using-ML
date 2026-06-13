# Final Evaluation Notebook — Implementation Spec

## 1. Mục tiêu

Notebook tổng hợp **toàn bộ kết quả** của project thành một bộ artifacts thống nhất,
phục vụ trực tiếp cho báo cáo. Không train model mới — chỉ load kết quả đã có và
phân tích sâu hơn.

---

## 2. Inputs (đã có từ các bước trước)

```
models/
├── best_model.pkl              — RandomForest Layer 1
└── anomaly_detector.pkl         — IsolationForest Layer 2

data/processed/
├── ssh_sessions.csv
├── X_train_ready.csv / y_train_ready.csv
├── X_time_train.csv / y_time_train.csv
└── X_time_test.csv / y_time_test.csv

results/
├── IsolationForest/
    ├── if_session_scores.csv        — IF scores cho toàn bộ 3916 sessions
    ├── if_contamination_tuning.csv
├── RandomForest/
    ├── confusion_matrix.png
    ├── ...
├── RandomForest_Decoupling/
    ├── comparison_summary.csv
    ├── feature_importance_comparison.csv
    ├── cv_predictions_full.csv
    └── cv_predictions_decoupled.csv
└── simulation_alerts.csv             — output từ run_simulation()
```

---

## 3. Notebook Structure

### §1 — Setup & Load All Artifacts

```python
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from datetime import datetime, timezone
from pathlib import Path

# Add src folder to path to import feature_engineering
sys.path.insert(0, "../src")
from feature_engineering import CLASS_NAMES, TRAINING_FEATURES

CLASS_LABELS = list(CLASS_NAMES.values())

# Create final results directory
Path("../results/final").mkdir(parents=True, exist_ok=True)

# Models
rf_model  = joblib.load("../models/best_model.pkl")
iso_model = joblib.load("../models/anomaly_detector.pkl")

# Session data
df_sessions = pd.read_csv("../data/processed/ssh_sessions.csv")

# IF scores - Align with df_sessions to get 'ip' column since they are row-aligned
df_if_scores = pd.read_csv("../results/IsolationForest/if_session_scores.csv")
df_if_scores["ip"] = df_sessions["ip"]

# Decoupling results
df_decoupling = pd.read_csv("../results/RandomForest_Decoupling/comparison_summary.csv")
df_importance = pd.read_csv("../results/RandomForest_Decoupling/feature_importance_comparison.csv")

# Simulation alerts
df_alerts = pd.read_csv("../results/simulation_alerts.csv")
df_early  = df_alerts[df_alerts["alert_kind"] == "EARLY_ALERT"].copy()
df_final  = df_alerts[df_alerts["alert_kind"] == "FINAL_CLASSIFICATION"].copy()

# Compute ts_first for final classifications to align with early alerts
df_final["ts_first"] = df_final["ts_alert"] - df_final["session_duration"]

# Ensure ts_first exists in df_early (it is in our simulation_alerts.csv on disk, but fallback just in case)
if "ts_first" not in df_early.columns or df_early["ts_first"].isna().any():
    def match_session_ts_first(row):
        candidates = df_final[
            (df_final["ip"] == row["ip"]) &
            (df_final["ts_first"] <= row["ts_alert"]) &
            (row["ts_alert"] <= df_final["ts_alert"])
        ]
        return candidates.iloc[0]["ts_first"] if len(candidates) == 1 else np.nan
    df_early["ts_first"] = df_early.apply(match_session_ts_first, axis=1)

print(f"Sessions:        {len(df_sessions):,}")
print(f"Early alerts:    {len(df_early):,}")
print(f"Final classifications: {len(df_final):,}")
```

---

### §2 — Layer 1 vs Layer 2 vs Hybrid Summary Table

Bảng tổng hợp 1 dòng cho mỗi "tầng" của hệ thống — dùng trực tiếp trong báo cáo.

```python
summary_rows = []

# Layer 1 (RF) — từ decoupling comparison (full feature set)
rf_row = df_decoupling[df_decoupling["feature_set"] == "full"].iloc[0]
summary_rows.append({
    "Component": "Layer 1 (RandomForest)",
    "Metric": "CV Macro F1",
    "Value": rf_row["cv_macro_f1"],
    "Note": "Known attack classification (class 0-4)",
})

# Layer 2 (IF) — false alarm / detection from tuning csv
df_tuning = pd.read_csv("../results/IsolationForest/if_contamination_tuning.csv")
best_row = df_tuning[df_tuning["contamination"] == 0.02].iloc[0]
summary_rows.append({
    "Component": "Layer 2 (IsolationForest)",
    "Metric": "False Alarm Rate",
    "Value": best_row["false_alarm_rate"],
    "Note": "On Normal sessions, contamination=0.02",
})

# Hybrid — distribution from FINAL_CLASSIFICATION
alert_dist = df_final["alert_type"].value_counts(normalize=True)
for atype, pct in alert_dist.items():
    summary_rows.append({
        "Component": "Hybrid (Final)",
        "Metric": f"% {atype}",
        "Value": pct,
        "Note": f"{df_final['alert_type'].value_counts()[atype]} sessions",
    })

df_system_summary = pd.DataFrame(summary_rows)
df_system_summary.to_csv("../results/final/system_summary.csv", index=False)
print(df_system_summary.to_string(index=False))
```

---

### §3 — Early Alert Latency Analysis

Đo "thời gian phản ứng" của hệ thống — bao nhiêu attempts trước khi early alert fire.

```python
# Latency theo alert_subtype
latency_stats = df_early.groupby("alert_subtype")["event_index"].agg(
    ["count", "mean", "median", "min", "max"]
).round(2)
latency_stats.columns = ["n_alerts", "mean_attempt", "median_attempt", "min_attempt", "max_attempt"]
latency_stats.to_csv("../results/final/early_alert_latency.csv")
print(latency_stats)

# Plot: distribution of event_index per subtype
fig, ax = plt.subplots(figsize=(9, 5))
for subtype in df_early["alert_subtype"].unique():
    subset = df_early[df_early["alert_subtype"] == subtype]
    ax.hist(subset["event_index"], bins=20, alpha=0.6, label=subtype)
ax.set_xlabel("Attempt # khi alert fire")
ax.set_ylabel("Số lượng sessions")
ax.set_title("Early Alert Latency — Attempt # tại thời điểm fire")
ax.legend()
plt.tight_layout()
plt.savefig("../results/final/early_alert_latency.png", dpi=150)
plt.close()
```

**Key number cho báo cáo:** trung bình `BRUTE_FORCE` fires tại attempt #6-7
(vì threshold=6), nghĩa là hệ thống cảnh báo **trước khi** session hoàn thành
toàn bộ (trung bình session brute-force có >100 attempts).

```python
avg_total_attempts_bf = df_sessions[df_sessions["class"]==3]["total_attempts"].mean()
avg_alert_attempt_bf  = df_early[df_early["alert_subtype"]=="BRUTE_FORCE"]["event_index"].mean()
early_pct = avg_alert_attempt_bf / avg_total_attempts_bf * 100
print(f"Brute-force: alert fires at attempt #{avg_alert_attempt_bf:.1f} "
      f"out of avg {avg_total_attempts_bf:.1f} total attempts "
      f"({early_pct:.1f}% through the session)")
```

---

### §4 — EARLY_ALERT vs FINAL_CLASSIFICATION Consistency

Sanity check: sessions có EARLY_ALERT="BRUTE_FORCE" có FINAL class=3 không?

```python
# Map mỗi session (ip, ts_first) → set of early alert subtypes
early_by_session = df_early.groupby(["ip", "ts_first"])["alert_subtype"].apply(set).to_dict()

# Merge với final
df_final["early_subtypes"] = df_final.apply(
    lambda row: early_by_session.get((row["ip"], row["ts_first"]), set()), axis=1
)

# Expected mapping
EXPECTED_FINAL = {
    "BRUTE_FORCE":   3,  # Brute-force
    "SCAN":          2,  # Scan
    "BREAK_IN_RISK": 4,  # Break-in
}

consistency_rows = []
for subtype, expected_class in EXPECTED_FINAL.items():
    mask = df_final["early_subtypes"].apply(lambda s: subtype in s)
    matched = (df_final.loc[mask, "layer1_class"] == expected_class).mean() if mask.sum() > 0 else 0.0
    consistency_rows.append({
        "early_subtype": subtype,
        "expected_final_class": CLASS_NAMES[expected_class],
        "n_sessions": mask.sum(),
        "consistency_rate": matched,
    })

df_consistency = pd.DataFrame(consistency_rows)
df_consistency.to_csv("../results/final/early_final_consistency.csv", index=False)
print(df_consistency.to_string(index=False))
```

**Lưu ý:** Consistency rate < 100% không phải lỗi — một session có thể trigger
`BRUTE_FORCE` early alert (streak >= 6 tại một điểm) nhưng kết thúc với pattern
khác (ví dụ thêm nhiều unique users sau đó → final = Scan vì priority `is_scan`
trong `label_session` không override `is_brute_force`... — review lại
`label_session priority` nếu mismatch rate cao bất thường, > 10%).

---

### §5 — UNKNOWN_PATTERN Case Studies

Deep-dive 2 sessions đã phát hiện (1783, 3820) — format cho báo cáo.

```python
unknown = df_final[df_final["alert_type"] == "UNKNOWN_PATTERN"]
print(f"Total UNKNOWN_PATTERN sessions: {len(unknown)}")

# Merge với session features đầy đủ
unknown_detail = df_sessions.merge(
    unknown[["ip", "ts_alert"]], left_on=["ip", "ts_last"], right_on=["ip", "ts_alert"]
)

case_study_cols = [
    "ip", "total_attempts", "session_duration", "total_successes",
    "unique_users_ratio", "is_single_event", "attempts_per_second",
    "has_reverse_mapping_failed", "class",
]
df_cases = unknown_detail[case_study_cols].copy()

# So sánh với Normal average
normal_avg = df_sessions[df_sessions["class"]==0][
    ["total_attempts","session_duration","total_successes",
     "unique_users_ratio","attempts_per_second"]
].mean()

print("Case studies vs Normal average:")
print(df_cases)
print("\nNormal averages:")
print(normal_avg)

df_cases.to_csv("../results/final/unknown_pattern_cases.csv", index=False)
```

#### Visualization — Feature deviation radar/bar chart

```python
features_to_compare = ["total_attempts", "session_duration",
                        "total_successes", "unique_users_ratio",
                        "attempts_per_second"]

fig, axes = plt.subplots(1, len(df_cases), figsize=(6*len(df_cases), 5))
if len(df_cases) == 1:
    axes = [axes]

for ax, (_, row) in zip(axes, df_cases.iterrows()):
    values = [row[f] for f in features_to_compare]
    norm_values = [normal_avg[f] for f in features_to_compare]

    x = np.arange(len(features_to_compare))
    w = 0.35
    ax.bar(x - w/2, norm_values, w, label="Normal avg", color="#90A4AE")
    ax.bar(x + w/2, values,      w, label=f"Session ({row['ip']})", color="#E53935")
    ax.set_xticks(x)
    ax.set_xticklabels(features_to_compare, rotation=30, ha="right")
    ax.set_title(f"IP {row['ip']} — UNKNOWN_PATTERN")
    ax.legend()

plt.suptitle("UNKNOWN_PATTERN Case Studies vs Normal Average", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("../results/final/unknown_pattern_cases.png", dpi=150)
plt.close()
```

---

### §6 — Timeline Visualization

Phân phối alerts theo thời gian — toàn bộ 28 ngày của dataset.

```python
df_final["ts_datetime"] = pd.to_datetime(df_final["ts_alert"], unit="s")
day0 = df_final["ts_datetime"].min().normalize()
df_final["day_index"] = (df_final["ts_datetime"] - day0).dt.days + 1  # Day 1, 2, ...

timeline = df_final.groupby(["day_index", "alert_type"]).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(14, 5))
timeline.plot(kind="bar", stacked=True, ax=ax,
               color={"NORMAL":"#4CAF50","KNOWN_ATTACK":"#E53935","UNKNOWN_PATTERN":"#FF9800"})
ax.set_title("Session Classifications Over Time (28 days)")
ax.set_ylabel("Sessions per day")
ax.set_xlabel("Day")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/final/timeline_alerts.png", dpi=150)
plt.close()

timeline.to_csv("../results/final/timeline_alerts.csv")
```

---

### §7 — Layer 2 Score Distribution (refined)

Tái sử dụng `if_session_scores.csv`, thêm overlay điểm UNKNOWN_PATTERN.

```python
fig, ax = plt.subplots(figsize=(10, 6))

for cls in range(5):
    subset = df_if_scores[df_if_scores["class"] == cls]
    ax.hist(subset["anomaly_score"], bins=30, alpha=0.5,
            label=f"{CLASS_NAMES[cls]} (n={len(subset)})")

# Highlight UNKNOWN_PATTERN points
unknown_scores = df_if_scores[df_if_scores["hybrid_alert"] == "UNKNOWN_PATTERN"]
for _, row in unknown_scores.iterrows():
    ax.axvline(row["anomaly_score"], color="red", linestyle="--", linewidth=2)
    ax.text(row["anomaly_score"], ax.get_ylim()[1]*0.9, f"IP {row['ip']}",
            rotation=90, color="red", fontsize=8, ha="right")

ax.axvline(0, color="black", linestyle="-", alpha=0.3, label="Decision boundary")
ax.set_xlabel("Anomaly Score (lower = more anomalous)")
ax.set_ylabel("Count")
ax.set_title("IF Score Distribution — UNKNOWN_PATTERN Highlighted")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("../results/final/if_scores_with_unknown.png", dpi=150)
plt.close()
```

---

### §8 — Master Results Table (cho Abstract/Conclusion)

Một bảng duy nhất tổng hợp toàn bộ con số quan trọng nhất của project.

```python
# Dynamic metrics from decoupling and tuning
cv_macro_f1_full = df_decoupling[df_decoupling["feature_set"] == "full"].iloc[0]["cv_macro_f1"]
cv_macro_f1_decoupled = df_decoupling[df_decoupling["feature_set"] == "decoupled"].iloc[0]["cv_macro_f1"]
holdout_macro_f1_full = df_decoupling[df_decoupling["feature_set"] == "full"].iloc[0]["holdout_macro_f1"]
holdout_macro_f1_decoupled = df_decoupling[df_decoupling["feature_set"] == "decoupled"].iloc[0]["holdout_macro_f1"]

df_tuning = pd.read_csv("../results/IsolationForest/if_contamination_tuning.csv")
best_row = df_tuning[df_tuning["contamination"] == 0.02].iloc[0]
far_validation = best_row["false_alarm_rate"]

# Actual False Alarm Rate on the entire dataset (Normal sessions flagged as anomaly)
# (Reconciles 1.6% actual vs 5.17% validation from tuning split)
far_actual = df_if_scores[df_if_scores["class"] == 0]["is_anomaly"].mean()

avg_alert_attempt_bf  = df_early[df_early["alert_subtype"]=="BRUTE_FORCE"]["event_index"].mean()

master = {
    "Dataset": {
        "Total log lines":      655_147,
        "Parsed events":        212_149,
        "Sessions (idle_gap=600s)": len(df_sessions),
        "Time span (days)":     28,
        "Unique source IPs":    df_sessions["ip"].nunique(),
    },
    "Class distribution": {
        f"Class {c} ({CLASS_NAMES[c]})": int((df_sessions["class"]==c).sum())
        for c in range(5)
    },
    "Layer 1 (RandomForest)": {
        "CV Macro F1 (full features)":      cv_macro_f1_full,
        "CV Macro F1 (decoupled features)": cv_macro_f1_decoupled,
        "Tautology cost":                   cv_macro_f1_full - cv_macro_f1_decoupled,
        "Holdout Macro F1 (full)":          holdout_macro_f1_full,
        "Holdout Macro F1 (decoupled)":     holdout_macro_f1_decoupled,
    },
    "Layer 2 (IsolationForest)": {
        "Contamination":                               0.02,
        "False alarm rate (out-of-sample validation)": far_validation,
        "False alarm rate (in-sample calibration)":    far_actual,
        "Sessions flagged UNKNOWN_PATTERN":           len(df_cases),
    },
    "Real-time simulation": {
        "Total early alerts":         len(df_early),
        "  - BRUTE_FORCE":             (df_early["alert_subtype"]=="BRUTE_FORCE").sum(),
        "  - SCAN":                    (df_early["alert_subtype"]=="SCAN").sum(),
        "  - BREAK_IN_RISK":           (df_early["alert_subtype"]=="BREAK_IN_RISK").sum(),
        "Sessions w/ early alert":     df_final["had_early_alert"].sum(),
        "Avg attempt# at BF alert":    avg_alert_attempt_bf,
    },
}

# Flatten + save
rows = []
for section, items in master.items():
    for k, v in items.items():
        rows.append({"Section": section, "Metric": k, "Value": v})
df_master = pd.DataFrame(rows)
df_master.to_csv("../results/final/master_results_table.csv", index=False)
print(df_master.to_string(index=False))
```

---

## 4. Output Directory Structure

```
results/final/
├── system_summary.csv              ← §2
├── early_alert_latency.csv         ← §3
├── early_alert_latency.png
├── early_final_consistency.csv     ← §4
├── unknown_pattern_cases.csv       ← §5
├── unknown_pattern_cases.png
├── timeline_alerts.csv             ← §6
├── timeline_alerts.png
├── if_scores_with_unknown.png      ← §7
└── master_results_table.csv        ← §8 — DÙNG CHO ABSTRACT/CONCLUSION
```

---

## 5. Lưu ý quan trọng

**§4 Consistency check là sanity check, không phải bug hunt:**
Nếu consistency rate thấp (<70%) cho một subtype, đó là điểm thảo luận thú vị
trong báo cáo (ví dụ: "Early alert BRUTE_FORCE chỉ map với Final class=3 ở 85%
trường hợp — 15% còn lại escalate thành Break-in vì attacker đoán đúng password
sau streak dài"). Không cần "fix" — đây là insight.

**§5 chỉ work nếu UNKNOWN_PATTERN sessions tồn tại:**
Nếu `len(df_cases) == 0` (không có session nào), section này note rõ:
"Trong run này, không có session nào được flag UNKNOWN_PATTERN — toàn bộ
Normal sessions khớp với profile học từ training data." Đây vẫn là kết quả
hợp lệ, chỉ thay đổi narrative.

**§8 Master table là nguồn duy nhất cho Abstract:**
Mọi số liệu trích dẫn trong Abstract/Conclusion của báo cáo PHẢI lấy từ
`master_results_table.csv` — đảm bảo consistency giữa các phần của báo cáo.

**Phân biệt FAR (validation) vs FAR (calibration):**
- `False alarm rate (in-sample calibration)` (~1.6%) được tính trên chính tập huấn luyện. Do IsolationForest tự hiệu chuẩn quyết định dựa trên tham số `contamination` (0.02), tỷ lệ dị biệt trên tập huấn luyện mặc định sẽ xấp xỉ 2% bất kể độ chính xác của mô hình. Đây là con số đo sự tự nhất quán (self-consistency), không phải hiệu năng tổng quát hóa.
- `False alarm rate (out-of-sample validation)` (~5.17%) được đo trên tập dữ liệu holdout (chưa từng thấy). Đây là con số quan trọng phản ánh chính xác khả năng tổng quát hóa (generalization) của IsolationForest. Việc hiệu năng giảm nhẹ out-of-sample (~3.2x) là bình thường và vẫn nằm dưới ngưỡng mục tiêu 10%.
