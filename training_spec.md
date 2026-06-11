# SSH Anomaly Detection — Training & Evaluation Spec

## 1. Context

### Pipeline đã hoàn thành
```
SSH.log (655K lines)
  → SSHLogParser          [log_processing.py]    → 212,149 records
  → build_session_dataset [data_labeling.py]     → 3,916 sessions
  → prepare_features      [feature_engineering.py] → X (3916×11), y (0–4)
```

### Dữ liệu đầu vào
| File | Shape | Dùng cho |
|------|-------|----------|
| `X_train_ready.csv` | 3916 × 11 | CV — toàn bộ dataset |
| `y_train_ready.csv` | 3916 × 1  | CV — toàn bộ dataset |
| `X_time_train.csv`  | 3107 × 11 | Time holdout — train |
| `y_time_train.csv`  | 3107 × 1  | Time holdout — train |
| `X_time_test.csv`   |  809 × 11 | Time holdout — test  |
| `y_time_test.csv`   |  809 × 1  | Time holdout — test  |

### Features (11 cột, đã log1p transform nơi cần)
```python
FEATURES = [
    "is_private",              # bool
    "session_duration",        # log1p(seconds)
    "attempts_per_second",     # log1p; =0 khi is_single_event=1
    "is_single_event",         # bool
    "failure_ratio",           # [0.0, 1.0]
    "unique_users_ratio",      # [0.0, 1.0]; =0 khi is_single_event=1
    "has_root_attempt",        # bool
    "has_valid_user_attempt",  # bool
    "max_failure_streak",      # log1p(count)
    "invalid_user_attempts",   # log1p(count)
    "has_reverse_mapping_failed", # bool
]
```

### Class distribution
| Class | Label | Count | % |
|-------|-------|------:|---:|
| 0 | Normal | 123 | 3.14% |
| 1 | Single failure | 2542 | 64.91% |
| 2 | Scan | 192 | 4.90% |
| 3 | Brute-force | 1040 | 26.56% |
| 4 | Break-in | 19 | 0.49% |

---

## 2. Chiến lược đánh giá

### Nguyên tắc
- **CV (5-fold stratified)**: dùng để chọn model và tune hyperparameters.
- **Time holdout**: dùng để report kết quả cuối cho class 0/1/2/3.
- **Break-in (class 4)**: chỉ 7 samples trong holdout test — **báo cáo từ CV**, không dùng holdout cho class này.

### Primary metric: Macro F1
```python
from sklearn.metrics import f1_score
macro_f1 = f1_score(y_true, y_pred, average='macro')
```
Macro F1 trọng số đều nhau cho 5 class — phù hợp với imbalanced data, không bị dominated bởi class 1 (64.91%).

### Secondary metrics
- Per-class Precision, Recall, F1 (`classification_report`)
- Confusion matrix (normalized by true label)
- Weighted F1 (để so sánh với baseline)

---

## 3. Xử lý Class Imbalance

**Không dùng SMOTE** — Break-in chỉ có 19 samples, SMOTE sẽ tạo synthetic samples không đáng tin cậy.

**Dùng `class_weight='balanced'`** cho tất cả models:
```python
# sklearn tự tính: weight[c] = n_samples / (n_classes * n_samples_c)
# → class 4 (Break-in) nhận weight ~103x so với class 1
```

---

## 4. Model Candidates

### 4.1 Baseline
```python
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy="most_frequent")
# Expected macro F1: ~0.10 (luôn predict class 1)
# Bất kỳ model nào cũng phải vượt qua ngưỡng này rõ ràng
```

### 4.2 Random Forest (primary candidate)
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

param_grid_rf = {
    "max_depth":        [None, 10, 20],
    "min_samples_leaf": [1, 2, 5],
    "max_features":     ["sqrt", "log2"],
}
```

### 4.3 Gradient Boosting — LightGBM (expected best)
```python
import lightgbm as lgb

lgbm = lgb.LGBMClassifier(
    n_estimators=500,
    class_weight="balanced",
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

param_grid_lgbm = {
    "max_depth":    [4, 6, 8],
    "num_leaves":   [15, 31, 63],
    "min_child_samples": [5, 10, 20],
    "subsample":    [0.8, 1.0],
}
```

### 4.4 Logistic Regression (interpretable baseline)
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
    multi_class="multinomial",
)
# Không cần StandardScaler — features đã log1p, range tương đương
```

---

## 5. Training Pipeline

```python
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# --- Load data ---
X = pd.read_csv("X_train_ready.csv")
y = pd.read_csv("y_train_ready.csv").squeeze()
X_time_train = pd.read_csv("X_time_train.csv")
y_time_train = pd.read_csv("y_time_train.csv").squeeze()
X_time_test  = pd.read_csv("X_time_test.csv")
y_time_test  = pd.read_csv("y_time_test.csv").squeeze()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- CV tuning cho mỗi model ---
def tune_model(estimator, param_distributions, X, y, cv, n_iter=20):
    search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_score_, search.best_params_

# --- Collect results ---
results = {}
for name, model, params in [
    ("RandomForest", rf, param_grid_rf),
    ("LightGBM",     lgbm, param_grid_lgbm),
    ("LogisticReg",  lr, {}),
]:
    best_model, cv_score, best_params = tune_model(model, params, X, y, cv)
    results[name] = {
        "model":     best_model,
        "cv_macro_f1": cv_score,
        "params":    best_params,
    }
    print(f"{name}: CV macro F1 = {cv_score:.4f}")
```

---

## 6. Evaluation Framework

### 6.1 Cross-Validation — per-class metrics
```python
from sklearn.model_selection import cross_val_predict

# Dùng cross_val_predict để lấy per-sample predictions
y_pred_cv = cross_val_predict(best_model, X, y, cv=cv)

print(classification_report(y, y_pred_cv,
      target_names=["Normal","Single failure","Scan","Brute-force","Break-in"]))
```

### 6.2 Time holdout — final evaluation
```python
# Train lại trên toàn bộ time_train, test trên time_test
best_model.fit(X_time_train, y_time_train)
y_pred_holdout = best_model.predict(X_time_test)

print(classification_report(y_time_test, y_pred_holdout,
      target_names=["Normal","Single failure","Scan","Brute-force","Break-in"]))

# ⚠️ Break-in (class 4): chỉ 7 samples trong holdout
# Kết quả holdout cho class 4 chỉ tham khảo, tin vào CV
```

### 6.3 Confusion matrix
```python
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_time_test, y_pred_holdout, normalize="true")
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES.values(),
            yticklabels=CLASS_NAMES.values(), ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
```

### 6.4 Feature importance (Random Forest / LightGBM)
```python
importances = pd.Series(
    best_model.feature_importances_, index=FEATURES
).sort_values(ascending=False)
print(importances)
```

---

## 7. Model Selection Criteria

Chọn model dựa trên **thứ tự ưu tiên**:

1. **CV Macro F1** cao nhất — primary metric
2. **CV F1 class 2 (Scan)** và **class 4 (Break-in)** — hai class khó nhất
3. Nếu macro F1 chênh ≤ 0.01 → ưu tiên model đơn giản hơn (Random Forest > LightGBM)
4. Training time là yếu tố phụ, không phải tiêu chí chính

### Ngưỡng chấp nhận được
| Metric | Ngưỡng tối thiểu |
|--------|-----------------|
| CV Macro F1 | ≥ 0.75 |
| F1 class 0 (Normal) | ≥ 0.85 |
| F1 class 3 (Brute-force) | ≥ 0.90 |
| F1 class 2 (Scan) | ≥ 0.60 |
| F1 class 4 (Break-in) | ≥ 0.50 (từ CV) |

---

## 8. Output Artifacts

| File | Nội dung |
|------|----------|
| `model_comparison.csv` | CV macro F1 + per-class F1 cho từng model |
| `best_model.pkl` | Model tốt nhất (joblib.dump) |
| `confusion_matrix.png` | Heatmap trên time holdout |
| `feature_importance.csv` | Feature importances (nếu tree-based) |
| `cv_classification_report.txt` | Full report từ cross_val_predict |
| `holdout_classification_report.txt` | Full report từ time holdout |

```python
import joblib
joblib.dump(best_model, "best_model.pkl")
```

---

## 9. Lưu ý quan trọng

**Break-in (class 4):**
- CV: ~4 samples/fold trong val set — F1 sẽ dao động lớn giữa các fold, dùng mean ± std
- Holdout: 7 samples — chỉ báo cáo, không dùng để so sánh models
- Nếu F1 class 4 = 0.0 trên holdout: không phải model fail, là do sample size quá nhỏ

**Không StandardScaler:**
Features đã được log1p transform và range tương đương nhau. Logistic Regression hoạt động đủ tốt mà không cần scale thêm. Nếu muốn thêm SVM sau này thì cần scale.

**Không GridSearchCV toàn diện:**
Dùng `RandomizedSearchCV(n_iter=20)` — với dataset 3916 rows, full GridSearch tốn thời gian không đáng.
