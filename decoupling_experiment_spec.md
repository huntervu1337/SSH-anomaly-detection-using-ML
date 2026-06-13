# Feature Decoupling Experiment — Implementation Spec

## 1. Mục tiêu

Định lượng phần F1 đến từ tautology (feature-label coupling) vs generalization thực sự.
Tạo toàn bộ artifacts cần thiết để so sánh, đối chiếu trong evaluation notebook và báo cáo.

## 2. Background

Model RandomForest trong pipeline hiện tại đạt Macro F1 = 0.9943 trên CV.
Tuy nhiên, 4 trong 11 features **trực tiếp encode labeling rules**:

| Feature bị loại | Rule encode trực tiếp |
|---|---|
| `max_failure_streak` | `streak > 5 → class 3 (Brute-force)` |
| `unique_users_ratio` | `ratio > 0.5 → class 2 (Scan)` |
| `failure_ratio` | `ratio == 0 → class 0 (Normal)` |
| `has_valid_user_attempt` | condition cho class 4 (Break-in) |

**Hypothesis:** Bỏ 4 features này sẽ làm F1 giảm, phần giảm đó chính là "tautology cost".

---

## 3. Feature Sets

```python
# Full — 11 features (pipeline hiện tại)
FEATURES_FULL = [
    "is_private", "session_duration", "attempts_per_second", "is_single_event",
    "failure_ratio", "unique_users_ratio", "has_root_attempt",
    "has_valid_user_attempt", "max_failure_streak", "invalid_user_attempts",
    "has_reverse_mapping_failed",
]

# Decoupled — 7 features (loại bỏ rule-encoding features)
FEATURES_DECOUPLED = [
    "is_private", "session_duration", "attempts_per_second", "is_single_event",
    "has_root_attempt", "invalid_user_attempts", "has_reverse_mapping_failed",
]

LEAKY_FEATURES = [
    "max_failure_streak", "unique_users_ratio",
    "failure_ratio", "has_valid_user_attempt",
]
```

---

## 4. Experiment Setup

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from feature_engineering import load_session_csv, prepare_features, CLASS_NAMES
CLASS_LABELS = list(CLASS_NAMES.values())  # ["Normal", "Single failure", ...]

# --- Load data ---
df            = load_session_csv("data/processed/ssh_sessions.csv")
X_full, y     = prepare_features(df)
X_decoupled   = X_full.drop(columns=LEAKY_FEATURES)

X_time_train  = pd.read_csv("data/processed/X_time_train.csv")
y_time_train  = pd.read_csv("data/processed/y_time_train.csv").squeeze()
X_time_test   = pd.read_csv("data/processed/X_time_test.csv")
y_time_test   = pd.read_csv("data/processed/y_time_test.csv").squeeze()

X_time_train_dec = X_time_train.drop(columns=LEAKY_FEATURES)
X_time_test_dec  = X_time_test.drop(columns=LEAKY_FEATURES)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

RF_PARAMS = dict(n_estimators=300, class_weight="balanced",
                 random_state=42, n_jobs=-1)
```

---

## 5. Training & CV Evaluation

```python
# Train + CV predict cho cả hai models
rf_full = RandomForestClassifier(**RF_PARAMS)
rf_dec  = RandomForestClassifier(**RF_PARAMS)

y_pred_full = cross_val_predict(rf_full, X_full,      y, cv=cv)
y_pred_dec  = cross_val_predict(rf_dec,  X_decoupled, y, cv=cv)

# Fit trên toàn bộ data (cho feature importance)
rf_full.fit(X_full,      y)
rf_dec.fit(X_decoupled,  y)

# Holdout evaluation: fit model trên time train only (tránh leakage)
rf_full_ho = RandomForestClassifier(**RF_PARAMS)
rf_dec_ho  = RandomForestClassifier(**RF_PARAMS)

X_time_train_full = X_time_train.reindex(columns=FEATURES_FULL, fill_value=0)
X_time_test_full  = X_time_test.reindex(columns=FEATURES_FULL, fill_value=0)
rf_full_ho.fit(X_time_train_full, y_time_train)
rf_dec_ho.fit(X_time_train_dec, y_time_train)

# Holdout predictions
y_pred_full_holdout = rf_full_ho.predict(X_time_test_full)
y_pred_dec_holdout  = rf_dec_ho.predict(X_time_test_dec)
```

---

## 6. Output Artifacts

### 6.1 CSV Outputs

#### `comparison_summary.csv`
Bảng tổng hợp metrics chính:

| column | nội dung |
|---|---|
| `model` | "Full (11 features)" / "Decoupled (7 features)" |
| `feature_set` | "full" / "decoupled" |
| `n_features` | 11 / 7 |
| `cv_macro_f1` | CV Macro F1 |
| `cv_weighted_f1` | CV Weighted F1 |
| `holdout_macro_f1` | Time holdout Macro F1 |
| `f1_class_{0..4}` | Per-class CV F1 cho từng class |
| `f1_drop_vs_full` | = cv_macro_f1_full - cv_macro_f1 (0 cho full) |

```python
rows = []
for name, feat_set, y_pred_cv, y_pred_ho in [
    ("Full (11 features)",      "full",      y_pred_full,  y_pred_full_holdout),
    ("Decoupled (7 features)",  "decoupled", y_pred_dec,   y_pred_dec_holdout),
]:
    f1_per_class = f1_score(y, y_pred_cv, average=None)
    row = {
        "model":            name,
        "feature_set":      feat_set,
        "n_features":       11 if feat_set == "full" else 7,
        "cv_macro_f1":      f1_score(y, y_pred_cv, average="macro"),
        "cv_weighted_f1":   f1_score(y, y_pred_cv, average="weighted"),
        "holdout_macro_f1": f1_score(y_time_test, y_pred_ho, average="macro"),
    }
    for cls in range(5):
        row[f"f1_class_{cls}"] = f1_per_class[cls]
    rows.append(row)

df_summary = pd.DataFrame(rows)
# Tính delta so với full
full_macro = df_summary.loc[df_summary["feature_set"]=="full", "cv_macro_f1"].values[0]
df_summary["f1_drop_vs_full"] = full_macro - df_summary["cv_macro_f1"]
df_summary.to_csv("models/results/RandomForest_Decoupling/comparison_summary.csv", index=False)
```

#### `feature_importance_comparison.csv`
Feature importance của cả 2 models (full model có NaN cho 4 leaky features trong cột decoupled):

| column | nội dung |
|---|---|
| `feature` | tên feature |
| `importance_full` | importance trong full model |
| `importance_decoupled` | importance trong decoupled model (NaN nếu feature bị loại) |
| `is_leaky` | 1 nếu feature thuộc LEAKY_FEATURES |

```python
imp_full = pd.Series(rf_full.feature_importances_, index=FEATURES_FULL)
imp_dec  = pd.Series(rf_dec.feature_importances_,  index=FEATURES_DECOUPLED)

df_imp = pd.DataFrame({
    "feature":              FEATURES_FULL,
    "importance_full":      imp_full.values,
    "importance_decoupled": [imp_dec.get(f, np.nan) for f in FEATURES_FULL],
    "is_leaky":             [int(f in LEAKY_FEATURES) for f in FEATURES_FULL],
}).sort_values("importance_full", ascending=False)
df_imp.to_csv("models/results/RandomForest_Decoupling/feature_importance_comparison.csv", index=False)
```

#### `cv_predictions_full.csv` và `cv_predictions_decoupled.csv`
Cross-val predictions để downstream analysis:

```python
pd.DataFrame({"y_true": y, "y_pred": y_pred_full}).to_csv(
    "models/results/RandomForest_Decoupling/cv_predictions_full.csv", index=False)
pd.DataFrame({"y_true": y, "y_pred": y_pred_dec}).to_csv(
    "models/results/RandomForest_Decoupling/cv_predictions_decoupled.csv", index=False)
```

---

### 6.2 Text Reports

#### `cv_report_full.txt` và `cv_report_decoupled.txt`

```python
for name, y_pred in [("full", y_pred_full), ("decoupled", y_pred_dec)]:
    report = classification_report(y, y_pred,
                target_names=CLASS_LABELS, digits=4)
    with open(f"models/results/RandomForest_Decoupling/cv_report_{name}.txt", "w") as f:
        f.write(f"Model: RandomForest — {name.upper()} features\n")
        f.write(f"CV Macro F1: {f1_score(y, y_pred, average='macro'):.4f}\n\n")
        f.write(report)
```

---

### 6.3 Visualizations

#### Plot 1: `macro_f1_comparison.png`
Bar chart so sánh CV Macro F1 và Holdout Macro F1 của 2 models.
Highlight phần chênh lệch (tautology cost).

```python
fig, ax = plt.subplots(figsize=(8, 5))

models   = ["Full\n(11 features)", "Decoupled\n(7 features)"]
cv_f1    = [f1_score(y, y_pred_full, average="macro"),
            f1_score(y, y_pred_dec,  average="macro")]
ho_f1    = [f1_score(y_time_test, y_pred_full_holdout, average="macro"),
            f1_score(y_time_test, y_pred_dec_holdout,  average="macro")]

x = np.arange(len(models))
w = 0.35
bars_cv = ax.bar(x - w/2, cv_f1, w, label="CV Macro F1",      color="#2196F3")
bars_ho = ax.bar(x + w/2, ho_f1, w, label="Holdout Macro F1", color="#FF9800")

# Annotate bars
for bar in list(bars_cv) + list(bars_ho):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=10)

# Annotate tautology cost
tautology_cost = cv_f1[0] - cv_f1[1]
ax.annotate(
    f"Tautology cost\n−{tautology_cost:.4f}",
    xy=(0 - w/2, cv_f1[0]), xytext=(0.5, cv_f1[0] + 0.02),
    arrowprops=dict(arrowstyle="->", color="red"),
    color="red", fontsize=9,
)

ax.set_ylim(0.85, 1.02)
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylabel("Macro F1")
ax.set_title("Full vs Decoupled — Macro F1 Comparison\n(Tautology Cost Highlighted)")
ax.legend(); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("results/decoupling/macro_f1_comparison.png", dpi=150)
plt.close()
```

#### Plot 2: `per_class_f1_comparison.png`
Grouped bar chart — F1 mỗi class cho Full vs Decoupled.
Highlight class có drop lớn nhất (Break-in).

```python
f1_full_cls = f1_score(y, y_pred_full, average=None)
f1_dec_cls  = f1_score(y, y_pred_dec,  average=None)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(5)
w = 0.35
ax.bar(x - w/2, f1_full_cls, w, label="Full (11 feat)", color="#2196F3")
ax.bar(x + w/2, f1_dec_cls,  w, label="Decoupled (7 feat)", color="#FF9800")

# Annotate delta cho từng class
for i in range(5):
    delta = f1_dec_cls[i] - f1_full_cls[i]
    color = "red" if delta < -0.01 else "green" if delta > 0.01 else "gray"
    ax.text(i, max(f1_full_cls[i], f1_dec_cls[i]) + 0.005,
            f"{delta:+.3f}", ha="center", fontsize=9, color=color, fontweight="bold")

# Threshold lines
ax.axhline(0.75, color="gray",   linestyle="--", alpha=0.5, label="Threshold 0.75")
ax.axhline(0.50, color="orange", linestyle="--", alpha=0.5, label="Threshold 0.50")

ax.set_xticks(x)
ax.set_xticklabels([f"class={i}\n{CLASS_NAMES[i]}" for i in range(5)])
ax.set_ylim(0.5, 1.05)
ax.set_ylabel("F1 Score (CV)")
ax.set_title("Per-Class F1: Full vs Decoupled Features\n(delta shown above bars)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/decoupling/per_class_f1_comparison.png", dpi=150)
plt.close()
```

#### Plot 3: `feature_importance_comparison.png`
Horizontal bar chart — importance Full vs Decoupled side by side.
Leaky features highlighted màu khác.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel trái: Full model
imp_full_sorted = pd.Series(rf_full.feature_importances_,
                             index=FEATURES_FULL).sort_values()
colors_full = ["#E53935" if f in LEAKY_FEATURES else "#1565C0"
               for f in imp_full_sorted.index]
axes[0].barh(imp_full_sorted.index, imp_full_sorted.values, color=colors_full)
axes[0].set_title("Full Model (11 features)\n(red = leaky/rule-encoding)")
axes[0].set_xlabel("Importance")

# Panel kanan: Decoupled model
imp_dec_sorted = pd.Series(rf_dec.feature_importances_,
                            index=FEATURES_DECOUPLED).sort_values()
axes[1].barh(imp_dec_sorted.index, imp_dec_sorted.values, color="#1565C0")
axes[1].set_title("Decoupled Model (7 features)\n(no rule-encoding features)")
axes[1].set_xlabel("Importance")

plt.suptitle("Feature Importance: Full vs Decoupled", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("models/results/RandomForest_Decoupling/feature_importance_comparison.png", dpi=150)
plt.close()
```

#### Plot 4: `confusion_matrices_comparison.png`
2×2 grid: CV confusion matrix cho Full (normalized) | Decoupled (normalized)
+ raw count cho từng cái.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, y_pred, title in [
    (axes[0], y_pred_full, f"Full (11 feat) — CV\nMacro F1={f1_score(y,y_pred_full,average='macro'):.4f}"),
    (axes[1], y_pred_dec,  f"Decoupled (7 feat) — CV\nMacro F1={f1_score(y,y_pred_dec,average='macro'):.4f}"),
]:
    cm = confusion_matrix(y, y_pred, normalize="true")
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS,
                ax=ax, vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")

plt.suptitle("Confusion Matrices: Full vs Decoupled (CV, normalized)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("models/results/RandomForest_Decoupling/confusion_matrices_comparison.png", dpi=150)
plt.close()
```

---

## 7. Output Directory Structure

```
models/results/RandomForest_Decoupling/
├── comparison_summary.csv              ← bảng tổng hợp chính
├── feature_importance_comparison.csv   ← importance full vs decoupled
├── cv_predictions_full.csv             ← y_true, y_pred từ CV
├── cv_predictions_decoupled.csv
├── cv_report_full.txt                  ← classification report
├── cv_report_decoupled.txt
├── macro_f1_comparison.png             ← bar chart F1 + tautology cost
├── per_class_f1_comparison.png         ← grouped bar per class
├── feature_importance_comparison.png   ← side-by-side importance
└── confusion_matrices_comparison.png   ← confusion matrix grid
```

---

## 8. Key Numbers để Highlight trong Báo cáo

Sau khi chạy experiment, điền vào bảng này (từ `comparison_summary.csv`):

| Metric | Full | Decoupled | Tautology cost |
|---|---|---|---|
| CV Macro F1 | 0.9943 | 0.9268 | **−0.0675** |
| F1 Normal | 1.0000 | 0.9500 | −0.0500 |
| F1 Single failure | 1.0000 | 0.9971 | −0.0029 |
| F1 Scan | 0.9974 | 0.9596 | −0.0378 |
| F1 Brute-force | 1.0000 | **1.0000** | **0.0000** ← insight |
| F1 Break-in | 0.9744 | 0.7273 | −0.2471 |

**Insight cho báo cáo:**
- Brute-force F1 = 1.0 ngay cả khi decoupled → signal tự nhiên trong data, không cần rule
- Break-in bị ảnh hưởng nhất (−0.2471) → phụ thuộc rule + ít data (19 samples)
- 92.68% F1 là generalization thật, 7.32% là tautology

---

## 9. Lưu ý

**Decoupled model không thay thế Full model trong production:**
Full model vẫn là model chính trong pipeline. Decoupling experiment chỉ phục vụ
mục đích nghiên cứu — định lượng tautology và chứng minh generalization capacity.

**Framing cho báo cáo:**
> *"To quantify the contribution of feature-label coupling to model performance,
> we conducted a decoupling experiment by removing the four features that directly
> encode the labeling rules. The resulting model achieved Macro F1 = 0.9268,
> suggesting that approximately 6.75% of the original score is attributable to
> tautological learning, while the remaining 92.68% reflects genuine generalization
> from indirect features. Notably, Brute-force classification remained perfect
> (F1 = 1.0) even without its rule-encoding feature, demonstrating that this
> attack class exhibits sufficiently distinctive behavioral patterns in the
> remaining feature space."*
