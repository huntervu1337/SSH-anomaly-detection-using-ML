# SSH Anomaly Detection — Real-time Hybrid Detection Spec

## 1. Tổng quan

### Những gì đã có
```
log_processing.py      — SSHLogParser
data_labeling.py       — build_session_dataset, label_session
feature_engineering.py — prepare_features, load_session_csv
best_model.pkl         — RandomForest Layer 1 (Macro F1 = 0.9943)
ssh_sessions.csv       — 3,916 sessions đã labeled
X_train_ready.csv / y_train_ready.csv
```

### Những gì cần build
```
anomaly_detector.py    — Layer 2: Isolation Forest (train + score)
realtime_simulator.py  — Simulate streaming detection trên SSH.log
hybrid_detector.py     — Kết hợp toàn bộ pipeline
Evaluation notebook    — Đánh giá hệ thống hybrid
```

---

## 2. Kiến trúc

### Tại sao cần tách Early Alert và Final Classification

Model RandomForest được train trên **closed session features** (giá trị cuối khi session
đã kết thúc hoàn toàn). Nếu gọi `model.predict()` trên partial session (session đang mở),
các features có giá trị khác về bản chất — đây là distribution shift:

| Feature | Sau 6 events | Sau 200 events (closed) |
|---|---|---|
| `max_failure_streak` | 6 | 198 |
| `attempts_per_second` | 2.0 | 0.8 |
| `unique_users_ratio` | 0.17 | 0.01 |
| `session_duration` | 5s | 247s |

Kết quả predict trên partial session không đáng tin cậy. Giải pháp: dùng
**rule-based Early Alert** cho cảnh báo tức thì, **model-based Final Classification**
khi session đóng.

### Sơ đồ tổng thể

```
Log stream (từng dòng)
       │
       ▼
SSHLogParser.parse_line()           ← per-event
       │
       ▼
SessionBuffer.add_event(record)     ← in-memory state tracking
       │
       ├──► [Mỗi event] EarlyAlertEngine.check(session_state)
       │         │
       │         └─ Trigger → EARLY_ALERT (rule-based, latency < 1s)
       │
       └──► [Khi session đóng] classify_closed_session()
                 │
                 ├──► aggregate_session() → prepare_features()
                 │
                 ├──► Layer 1: RandomForest.predict()
                 │         └─ class 1–4 → KNOWN_ATTACK
                 │
                 └──► Layer 2: IsolationForest.predict()
                           └─ class=0 + anomaly → UNKNOWN_PATTERN
                           └─ class=0 + normal  → NORMAL
```

---

## 3. In-Memory State Tracking

### File: `realtime_simulator.py` — Class `SessionBuffer`

Duy trì trạng thái tất cả active sessions theo IP. Mỗi event mới kích hoạt
4 bước xử lý (State Check → Gap Evaluation → Incremental Update → Early Alert).

```python
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class SessionState:
    """Trạng thái của một active session."""
    ip: str
    records: List[dict] = field(default_factory=list)
    ts_first: int = 0
    ts_last: int = 0
    total_attempts: int = 0
    total_failures: int = 0
    max_failure_streak: int = 0
    cur_failure_streak: int = 0
    unique_users: set = field(default_factory=set)
    has_root_attempt: int = 0
    has_valid_user_attempt: int = 0
    early_alert_fired: set = field(default_factory=set)  # tránh alert trùng lặp


class SessionBuffer:
    """
    In-memory state tracking cho active sessions.

    Một session đóng khi:
      - Event mới từ cùng IP có gap > idle_gap giây
      - flush() được gọi (end of stream)
    """

    def __init__(self, idle_gap: int = 600):
        self.idle_gap = idle_gap
        self.sessions: Dict[str, SessionState] = {}

    def add_event(self, record: dict) -> Tuple[Optional[List[dict]], SessionState]:
        """
        Bước 1–2: State Check + Gap Evaluation.
        Bước 3: Incremental Update.

        Returns:
            closed_records: list records của session vừa đóng (nếu có), else None
            current_state:  SessionState sau khi update (dùng cho Early Alert)
        """
        ip = str(record["ip"])
        ts = int(record["ts"])

        # Bước 1 & 2: Gap Evaluation
        closed_records = None
        if ip in self.sessions:
            state = self.sessions[ip]
            if ts - state.ts_last > self.idle_gap:
                # Trường hợp 2a: gap lớn → đóng session cũ
                closed_records = list(state.records)
                del self.sessions[ip]
                # Khởi tạo session mới bên dưới
            # else: Trường hợp 2b → tiếp tục session cũ (update bên dưới)

        # Bước 3: Incremental Update
        if ip not in self.sessions:
            self.sessions[ip] = SessionState(ip=ip, ts_first=ts)

        state = self.sessions[ip]
        state.records.append(record)
        state.ts_last = ts
        state.total_attempts += 1

        if int(record["is_failure"]) == 1:
            state.total_failures += 1
            state.cur_failure_streak += 1
            state.max_failure_streak = max(
                state.max_failure_streak, state.cur_failure_streak
            )
        else:
            state.cur_failure_streak = 0

        state.unique_users.add(str(record["user"]))
        state.has_root_attempt = max(state.has_root_attempt, int(record["is_root"]))
        state.has_valid_user_attempt = max(
            state.has_valid_user_attempt, int(record["is_valid"])
        )

        return closed_records, state

    def flush(self) -> List[List[dict]]:
        """Đóng tất cả open sessions (end of stream)."""
        remaining = [s.records for s in self.sessions.values() if s.records]
        self.sessions.clear()
        return remaining
```

---

## 4. Early Alert Engine

### Rule-based, chạy sau mỗi Incremental Update (Bước 4)

```python
class EarlyAlertEngine:
    """
    Cảnh báo tức thì dựa trên ngưỡng rule — không dùng model.predict().
    Mỗi alert type chỉ fire 1 lần per session (dùng state.early_alert_fired).
    """

    # Ngưỡng cảnh báo
    BRUTE_FORCE_STREAK   = 6    # failures liên tiếp
    SCAN_UNIQUE_USERS    = 4    # unique usernames thử trong session
    BREAK_IN_RISK        = True # valid user fail rồi success → ngay lập tức

    def check(self, state: SessionState, record: dict) -> Optional[dict]:
        """
        Kiểm tra điều kiện early alert sau mỗi event.
        Returns alert dict nếu trigger, else None.
        """

        # Trigger 1: Brute-force in progress
        if (
            "BRUTE_FORCE" not in state.early_alert_fired
            and state.max_failure_streak >= self.BRUTE_FORCE_STREAK
        ):
            state.early_alert_fired.add("BRUTE_FORCE")
            return self._make_early_alert(
                state, record,
                alert_subtype="BRUTE_FORCE",
                reason=f"max_failure_streak={state.max_failure_streak} ≥ {self.BRUTE_FORCE_STREAK}",
            )

        # Trigger 2: Scan in progress
        if (
            "SCAN" not in state.early_alert_fired
            and len(state.unique_users) >= self.SCAN_UNIQUE_USERS
            and state.total_failures > 0
        ):
            state.early_alert_fired.add("SCAN")
            return self._make_early_alert(
                state, record,
                alert_subtype="SCAN",
                reason=f"unique_users={len(state.unique_users)} ≥ {self.SCAN_UNIQUE_USERS}",
            )

        # Trigger 3: Break-in risk (valid user từng fail, vừa success)
        if (
            "BREAK_IN_RISK" not in state.early_alert_fired
            and int(record["is_failure"]) == 0          # vừa success
            and int(record["is_valid"]) == 1             # với valid user
            and state.total_failures > 0                  # đã từng fail trước đó
        ):
            state.early_alert_fired.add("BREAK_IN_RISK")
            return self._make_early_alert(
                state, record,
                alert_subtype="BREAK_IN_RISK",
                reason="valid user succeeded after previous failures",
            )

        return None

    def _make_early_alert(
        self, state: SessionState, record: dict,
        alert_subtype: str, reason: str,
    ) -> dict:
        return {
            "alert_kind":    "EARLY_ALERT",
            "alert_subtype": alert_subtype,
            "ts_alert":      int(record["ts"]),
            "ip":            state.ip,
            "attempts_so_far": state.total_attempts,
            "failures_so_far": state.total_failures,
            "reason":        reason,
        }
```

---

## 5. Layer 2 — Isolation Forest

### File: `anomaly_detector.py`

#### 5.1 Training

```python
from sklearn.ensemble import IsolationForest
import joblib

def train_anomaly_detector(
    X, y,
    contamination: float = 0.05,
    random_state: int = 42,
) -> IsolationForest:
    """Train Isolation Forest trên Normal sessions only (class=0)."""

    X_normal = X[y == 0]   # 123 sessions

    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
    )
    iso.fit(X_normal)
    return iso
```

#### 5.2 Scoring

```python
def score_session(iso, x) -> dict:
    """
    Returns:
        anomaly_score: float  — thấp hơn = bất thường hơn
        is_anomaly:    int    — 1 = anomaly, 0 = normal
    """
    return {
        "anomaly_score": round(float(iso.decision_function(x)[0]), 4),
        "is_anomaly":    int(iso.predict(x)[0] == -1),
    }
```

#### 5.3 Threshold tuning

```python
# Chọn contamination thấp nhất mà detection rate vẫn chấp nhận được
for c in [0.01, 0.02, 0.05, 0.10]:
    iso_c = IsolationForest(contamination=c, n_estimators=200, random_state=42)
    iso_c.fit(X[y == 0])
    fa = (iso_c.predict(X_time_test[y_time_test == 0]) == -1).mean()
    dr = (iso_c.predict(X_time_test[y_time_test > 0])  == -1).mean()
    print(f"contamination={c:.2f} → false_alarm={fa:.1%}, detection={dr:.1%}")
# Target: false_alarm < 10%
```

---

## 6. Alert Schema

```python
# EARLY_ALERT — phát ngay trong session đang mở
EARLY_ALERT_SCHEMA = {
    "alert_kind":      str,   # "EARLY_ALERT"
    "alert_subtype":   str,   # "BRUTE_FORCE" | "SCAN" | "BREAK_IN_RISK"
    "ts_alert":        int,
    "ip":              str,
    "ts_first":        int,
    "attempts_so_far": int,
    "failures_so_far": int,
    "reason":          str,   # mô tả trigger condition
}

# FINAL_CLASSIFICATION — phát khi session đóng
FINAL_ALERT_SCHEMA = {
    "alert_kind":      str,   # "FINAL_CLASSIFICATION"
    "alert_type":      str,   # "KNOWN_ATTACK" | "UNKNOWN_PATTERN" | "NORMAL"
    "ts_alert":        int,
    "ip":              str,
    "session_duration": int,
    "total_attempts":  int,
    "layer1_class":    int,   # 0–4
    "layer1_label":    str,
    "layer2_score":    float,
    "layer2_anomaly":  int,
    "had_early_alert": int,   # 1 nếu session này đã có EARLY_ALERT trước đó
}
```

---

## 7. Main Simulation Loop

```python
def run_simulation(log_path: str, rf_model, iso_model) -> List[dict]:
    parser = SSHLogParser()
    parser.valid_users = parser.infer_valid_users_from_file(log_path)
    parser.ip_state = {}

    buffer  = SessionBuffer(idle_gap=IDLE_GAP)
    early   = EarlyAlertEngine()
    alerts  = []
    ips_with_early_alert = set()

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = parser.parse_line(line.strip())
            if record is None:
                continue

            # Bước 1–3: State check + gap eval + incremental update
            closed_records, state = buffer.add_event(record)

            # Bước 4: Early alert check
            early_alert = early.check(state, record)
            if early_alert:
                alerts.append(early_alert)
                ips_with_early_alert.add(state.ip)

            # Final classification khi session đóng
            if closed_records:
                alert = _final_classify(
                    closed_records, rf_model, iso_model,
                    had_early=closed_records[0]["ip"] in ips_with_early_alert,
                )
                alerts.append(alert)

    # Flush
    for session_records in buffer.flush():
        ip = session_records[0]["ip"]
        alert = _final_classify(
            session_records, rf_model, iso_model,
            had_early=ip in ips_with_early_alert,
        )
        alerts.append(alert)

    return alerts


def _final_classify(session_records, rf_model, iso_model, had_early: bool) -> dict:
    row   = aggregate_session(session_records)
    df    = pd.DataFrame([row])
    X_row, _ = prepare_features(df.assign(**{"class": 0}))

    l1_class  = int(rf_model.predict(X_row)[0])
    l2_result = score_session(iso_model, X_row)

    alert_type = (
        "KNOWN_ATTACK"     if l1_class > 0 else
        "UNKNOWN_PATTERN"  if l2_result["is_anomaly"] else
        "NORMAL"
    )

    return {
        "alert_kind":       "FINAL_CLASSIFICATION",
        "alert_type":       alert_type,
        "ts_alert":         row["ts_last"],
        "ip":               row["ip"],
        "session_duration": row["session_duration"],
        "total_attempts":   row["total_attempts"],
        "layer1_class":     l1_class,
        "layer1_label":     CLASS_NAMES[l1_class],
        "layer2_score":     l2_result["anomaly_score"],
        "layer2_anomaly":   l2_result["is_anomaly"],
        "had_early_alert":  int(had_early),
    }
```

---

## 8. Output Artifacts

| File | Nội dung |
|------|----------|
| `anomaly_detector.pkl` | Trained Isolation Forest |
| `simulation_alerts.csv` | Tất cả alerts (EARLY + FINAL) theo thứ tự thời gian |
| `alert_summary.txt` | Thống kê tổng hợp |

---

## 9. Evaluation Notebook

| Cell | Nội dung |
|------|----------|
| §1 Layer 2 tuning | False alarm rate + Detection rate theo contamination |
| §2 Alert overview | Phân phối EARLY vs FINAL, timeline theo giờ/ngày |
| §3 Early alert latency | Thời gian từ session_start → first EARLY_ALERT (trung bình theo subtype) |
| §4 EARLY vs FINAL consistency | % sessions có EARLY_ALERT → FINAL cùng class? (sanity check) |
| §5 UNKNOWN_PATTERN deep-dive | Feature profile của các sessions bị flag |
| §6 Layer 2 score distribution | Histogram theo alert_type |

---

## 10. Lưu ý

**Early Alert không thay thế Final Classification:**
EARLY_ALERT là tín hiệu cảnh báo sớm. FINAL_CLASSIFICATION là kết quả chính xác.
Trong báo cáo, phân tích cả hai và so sánh consistency giữa chúng.

**`early_alert_fired` set tránh alert spam:**
Không có set này, mỗi event sau threshold đều tạo alert mới cho cùng 1 session.

**Isolation Forest với 123 normal samples:**
Đủ cho research scope. Ghi nhận giới hạn này trong báo cáo.

**Không cần streaming framework:**
`SessionBuffer` + vòng lặp Python là đủ cho simulation. Không cần Kafka hay Redis.

**Thứ tự import:**
```python
import joblib
from log_processing    import SSHLogParser
from data_labeling     import aggregate_session, IDLE_GAP
from feature_engineering import prepare_features, CLASS_NAMES

rf_model  = joblib.load("best_model.pkl")
iso_model = joblib.load("anomaly_detector.pkl")
```
