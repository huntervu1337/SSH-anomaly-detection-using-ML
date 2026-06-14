# Live IDS Demo — Implementation Spec

## 1. Tổng quan

Chuyển từ batch simulation (`realtime_simulator.py`, đọc file tĩnh) sang **live daemon**
chạy trên Ubuntu VM, đọc `/var/log/auth.log` real-time bằng `tail -f`, trong khi Kali VM
thực hiện các kịch bản tấn công.

### Kiến trúc Dual-Thread

```
                    ┌─── Thread 1: Log Tailer (blocking I/O) ───┐
/var/log/auth.log ──┤    tail -f → parse_line → buffer.add_event│
                    │    → EarlyAlertEngine.check()             ├──► Console (rich)
                    │    → nếu session đóng: _final_classify()  │    + alerts.csv
                    └────────────────────────────────────────────┘    + alerts.jsonl
                    ┌─── Thread 2: Timeout Poller ──────────────┐
                    │    every 5s: buffer.check_expired(now)    │
                    │    → session idle quá IDLE_GAP            │
                    │    → _final_classify() ngay               │
                    └────────────────────────────────────────────┘

  buffer.sessions (dict) — shared, bảo vệ bằng threading.Lock()
```

---

## 2. Setup Ubuntu VM (Victim)

```bash
# Cài sshd nếu chưa có
sudo apt install openssh-server -y

# /etc/ssh/sshd_config — đảm bảo các dòng sau:
PasswordAuthentication yes
PermitRootLogin no          # giữ nguyên, hydra sẽ thử root nhưng fail — đúng ý

sudo systemctl restart ssh

# Tạo tài khoản demo
sudo useradd -m -s /bin/bash alice
echo "alice:CorrectPass123" | sudo chpasswd

# Cài Python deps cho live_ids.py
pip install rich joblib pandas scikit-learn --break-system-packages
```

Lấy IP của Ubuntu VM (`ip addr`) — dùng cho Kali scripts.

---

## 3. `live_ids.py` — Main Program

### 3.1 Config

```python
import threading, time, json, csv
from datetime import datetime
from pathlib import Path
import joblib
import pandas as pd
from rich.console import Console

from log_processing import SSHLogParser
from data_labeling import aggregate_session
from feature_engineering import prepare_features, CLASS_NAMES
from realtime_simulator import SessionState, EarlyAlertEngine

# ---- Demo config (khác offline research config) ----
AUTH_LOG    = "/var/log/auth.log"
IDLE_GAP    = 30          # giây — RÚT NGẮN cho demo (offline research dùng 600s)
POLL_EVERY  = 5           # giây — tần suất polling thread quét session hết hạn
DEMO_VALID_USERS = {"alice"}   # hardcode — KHÔNG dùng infer_valid_users_from_file
                                # (file đang stream, chưa "đầy đủ")

OUT_CSV   = "alerts.csv"
OUT_JSONL = "alerts.jsonl"
```

> **Lưu ý IDLE_GAP=30s:** giá trị nghiên cứu chính thức (600s, đã validate qua toàn bộ
> pipeline offline) KHÔNG đổi. IDLE_GAP chỉ ảnh hưởng đến *thời điểm đóng session*,
> không ảnh hưởng đến cách tính features hay model. 30s là tham số **trình diễn**,
> giúp demo phản hồi nhanh — document rõ sự khác biệt này trong báo cáo.

---

### 3.2 LiveSessionBuffer — thread-safe

```python
class LiveSessionBuffer:
    """Giống SessionBuffer trong realtime_simulator, nhưng thread-safe và
    có check_expired() cho polling thread."""

    def __init__(self, idle_gap: int):
        self.idle_gap = idle_gap
        self.sessions: dict[str, SessionState] = {}
        self.lock = threading.Lock()

    def add_event(self, record: dict):
        """Gọi từ Tailer thread. Logic giống SessionBuffer.add_event gốc."""
        ip = str(record["ip"])
        ts = int(record["ts"])

        with self.lock:
            closed_state = None
            if ip in self.sessions:
                state = self.sessions[ip]
                if ts - state.ts_last > self.idle_gap:
                    closed_state = state
                    del self.sessions[ip]

            if ip not in self.sessions:
                self.sessions[ip] = SessionState(ip=ip, ts_first=ts)

            state = self.sessions[ip]
            state.records.append(record)
            state.ts_last = ts
            state.total_attempts += 1

            if int(record["is_failure"]) == 1:
                state.total_failures += 1
                state.cur_failure_streak += 1
                state.max_failure_streak = max(state.max_failure_streak, state.cur_failure_streak)
            else:
                state.cur_failure_streak = 0

            state.unique_users.add(str(record["user"]))
            state.has_root_attempt = max(state.has_root_attempt, int(record["is_root"]))
            state.has_valid_user_attempt = max(state.has_valid_user_attempt, int(record["is_valid"]))

            return closed_state, state

    def check_expired(self, now_ts: int):
        """Gọi từ Polling thread. Trả về list session đã idle quá idle_gap."""
        with self.lock:
            expired_ips = [
                ip for ip, state in self.sessions.items()
                if now_ts - state.ts_last > self.idle_gap
            ]
            return [self.sessions.pop(ip) for ip in expired_ips]
```

---

### 3.3 Final Classification — `is_private` override

```python
def final_classify(closed_state: SessionState, rf_model, iso_model, had_early: bool) -> dict:
    row = aggregate_session(closed_state.records)

    # ⚠️ FIX: is_private là CONSTANT=0 trong 100% training data (3916 sessions).
    # VM lab dùng IP private (192.168.x.x) → is_private=1 — giá trị OOD chưa
    # từng thấy. Vì feature này đã xác nhận uninformative (importance=0 ở cả
    # RF và decoupled model), hardcode về 0 để khớp training distribution và
    # tránh IsolationForest bias score do split ngẫu nhiên trên feature OOD.
    row["is_private"] = 0

    df = pd.DataFrame([row])
    df["class"] = 0  # dummy, prepare_features cần cột này
    X_row, _ = prepare_features(df)

    l1_class  = int(rf_model.predict(X_row)[0])
    score     = float(iso_model.decision_function(X_row)[0])
    is_anom   = int(iso_model.predict(X_row)[0] == -1)

    alert_type = (
        "KNOWN_ATTACK"    if l1_class > 0 else
        "UNKNOWN_PATTERN" if is_anom else
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
        "layer2_score":     round(score, 4),
        "layer2_anomaly":   is_anom,
        "had_early_alert":  int(had_early),
    }
```

---

### 3.4 Console Output (rich)

```python
def emit_alert(alert: dict, console: Console, csv_writer, jsonl_fh, fieldnames):
    ts_str = datetime.fromtimestamp(alert["ts_alert"]).strftime("%H:%M:%S")

    if alert["alert_kind"] == "EARLY_ALERT":
        console.print(
            f"[{ts_str}] ⚠️  EARLY_ALERT [{alert['alert_subtype']}] "
            f"IP={alert['ip']} attempt#{alert['event_index']} — {alert['reason']}",
            style="bold yellow",
        )
    else:
        atype = alert["alert_type"]
        style, icon = {
            "KNOWN_ATTACK":    ("bold red",     "🔴"),
            "UNKNOWN_PATTERN": ("bold magenta", "🟣"),
            "NORMAL":          ("bold green",   "🟢"),
        }[atype]
        console.print(
            f"[{ts_str}] {icon} FINAL [{atype}] {alert['layer1_label']} "
            f"IP={alert['ip']} attempts={alert['total_attempts']} "
            f"dur={alert['session_duration']}s IF_score={alert['layer2_score']}",
            style=style,
        )
        if alert["layer1_class"] in (3, 4):  # Brute-force or Break-in
            console.bell()

    # Structured logging — dùng cho phân tích sau / downstream tools
    row = {k: alert.get(k, "") for k in fieldnames}
    csv_writer.writerow(row)
    jsonl_fh.write(json.dumps(alert) + "\n")
    jsonl_fh.flush()
```

---

### 3.5 Thread 1 — Log Tailer

```python
def tail_f(path: str):
    with open(path, "r") as f:
        f.seek(0, 2)  # seek to END — chỉ xử lý log MỚI từ lúc start
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            yield line


def tailer_thread(stop_event, parser, buffer, early_engine, rf_model, iso_model,
                   console, csv_writer, jsonl_fh, fieldnames):
    for line in tail_f(AUTH_LOG):
        if stop_event.is_set():
            break
        record = parser.parse_line(line.strip())
        if record is None:
            continue

        closed_state, current_state = buffer.add_event(record)

        early_alert = early_engine.check(current_state, record)
        if early_alert:
            emit_alert(early_alert, console, csv_writer, jsonl_fh, fieldnames)

        if closed_state is not None:
            had_early = len(closed_state.early_alert_fired) > 0
            final_alert = final_classify(closed_state, rf_model, iso_model, had_early)
            emit_alert(final_alert, console, csv_writer, jsonl_fh, fieldnames)
```

---

### 3.6 Thread 2 — Timeout Poller

```python
def poller_thread(stop_event, buffer, rf_model, iso_model, console, csv_writer, jsonl_fh, fieldnames):
    while not stop_event.is_set():
        time.sleep(POLL_EVERY)
        now_ts = int(time.time())
        for state in buffer.check_expired(now_ts):
            had_early = len(state.early_alert_fired) > 0
            final_alert = final_classify(state, rf_model, iso_model, had_early)
            emit_alert(final_alert, console, csv_writer, jsonl_fh, fieldnames)
```

---

### 3.7 Main

```python
def main():
    console = Console()
    parser  = SSHLogParser(valid_users=DEMO_VALID_USERS, year=datetime.now().year)
    buffer  = LiveSessionBuffer(idle_gap=IDLE_GAP)
    early_engine = EarlyAlertEngine()

    rf_model  = joblib.load("models/best_model.pkl")
    iso_model = joblib.load("models/anomaly_detector.pkl")

    fieldnames = [
        "alert_kind", "alert_subtype", "alert_type", "ts_alert", "ip",
        "total_attempts", "session_duration", "layer1_class", "layer1_label",
        "layer2_score", "layer2_anomaly", "had_early_alert",
        "attempts_so_far", "failures_so_far", "reason", "event_index",
    ]

    csv_fh = open(OUT_CSV, "w", newline="")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
    csv_writer.writeheader()
    jsonl_fh = open(OUT_JSONL, "w")

    stop_event = threading.Event()

    console.print(f"[bold cyan]Live IDS started — watching {AUTH_LOG}[/bold cyan]")
    console.print(f"[bold cyan]IDLE_GAP={IDLE_GAP}s (demo) | valid_users={DEMO_VALID_USERS}[/bold cyan]\n")

    t1 = threading.Thread(target=tailer_thread, args=(
        stop_event, parser, buffer, early_engine, rf_model, iso_model,
        console, csv_writer, jsonl_fh, fieldnames), daemon=True)
    t2 = threading.Thread(target=poller_thread, args=(
        stop_event, buffer, rf_model, iso_model,
        console, csv_writer, jsonl_fh, fieldnames), daemon=True)

    t1.start(); t2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[bold cyan]Shutting down...[/bold cyan]")
        stop_event.set()
        csv_fh.close(); jsonl_fh.close()


if __name__ == "__main__":
    main()
```

Run với: `sudo .venv/bin/python live_ids.py`

---

## 4. Kali Attack Scenarios

Tất cả scripts cần `TARGET=<Ubuntu_VM_IP>`. Cài `sshpass` và `hydra`:
```bash
sudo apt install sshpass hydra -y
```

### 4.1 `normal_login.sh` — Baseline NORMAL

```bash
#!/bin/bash
TARGET=$1
sshpass -p "CorrectPass123" ssh -o StrictHostKeyChecking=no alice@$TARGET "echo OK"
```
**Expected:** FINAL NORMAL, layer1_class=0.

---

### 4.2 `brute_force.sh` — BRUTE_FORCE

```bash
#!/bin/bash
TARGET=$1
hydra -l root -P /usr/share/wordlists/rockyou.txt -t 4 ssh://$TARGET
```
**Expected:** EARLY_ALERT[BRUTE_FORCE] tại attempt#6 (deterministic, đã validate offline),
sau đó FINAL KNOWN_ATTACK (Brute-force), beep.

---

### 4.3 `multi_user_scan.sh` — SCAN (kỳ vọng escalate thành Brute-force)

```bash
#!/bin/bash
TARGET=$1
hydra -L /usr/share/seclists/Usernames/top-usernames-shortlist.txt \
      -p "wrongpass" -t 4 ssh://$TARGET
```
**Expected behavior — đây là điểm thú vị để demo:**
- Nếu danh sách user đủ dài, `unique_users>=4` → EARLY_ALERT[SCAN]
- NHƯNG nếu cũng đạt `max_failure_streak>=6` → FINAL = Brute-force (3), KHÔNG phải Scan (2)
- Đây chính là **finding "SCAN consistency=0%"** đã phát hiện ở evaluation offline —
  live demo này **trực tiếp minh họa lại finding đó**, biến limitation thành
  một phần của câu chuyện trình bày.

---

### 4.4 `break_in.sh` — BREAK_IN_RISK

```bash
#!/bin/bash
TARGET=$1

# 2 lần sai password cho user hợp lệ
for i in 1 2; do
    sshpass -p "wrongpass$i" ssh -o StrictHostKeyChecking=no alice@$TARGET "exit" 2>/dev/null
    sleep 1
done

# Lần thứ 3: đúng password
sshpass -p "CorrectPass123" ssh -o StrictHostKeyChecking=no alice@$TARGET "echo Access granted"
```
**Expected:** EARLY_ALERT[BREAK_IN_RISK] ngay khi login thành công sau failures,
FINAL KNOWN_ATTACK (Break-in, class=4), beep.

---

### 4.5 `rapid_login.sh` — Thử trigger UNKNOWN_PATTERN (best-effort)

```bash
#!/bin/bash
TARGET=$1

# Nhiều login THÀNH CÔNG liên tiếp, gần như đồng thời — mô phỏng session 1783
# (2 successful logins, duration=0s, attempts_per_second cao bất thường)
for i in 1 2 3; do
    sshpass -p "CorrectPass123" ssh -o StrictHostKeyChecking=no alice@$TARGET "exit" &
done
wait
```

> **⚠️ KHÔNG đảm bảo kết quả:** session này có `failure_ratio=0` → Layer 1 chắc chắn
> predict Normal (đúng). Việc Layer 2 (IF) flag UNKNOWN_PATTERN hay không phụ thuộc
> vào việc feature vector (duration≈0, attempts_per_second cao, multi-success) có
> nằm ngoài decision boundary học từ 123 Normal sessions của dataset gốc hay không.
>
> **Cả 2 outcome đều là kết quả hợp lệ cho báo cáo:**
> - Nếu FLAG → minh chứng trực tiếp Layer 2 hoạt động trên live traffic
> - Nếu KHÔNG flag → minh chứng cho đúng limitation đã note: IF train trên 123
>   samples, decision boundary có thể không generalize sang traffic pattern
>   của môi trường demo (khác hẳn dataset gốc 2018, single-host LogHub)

---

## 5. Demo Runbook (thứ tự trình diễn)

```
1. Terminal Ubuntu:  sudo .venv/bin/python live_ids.py
   → "Live IDS started — watching /var/log/auth.log"

2. Kali:  ./normal_login.sh $TARGET
   → 🟢 FINAL [NORMAL] Normal | attempts=1

3. Kali:  ./brute_force.sh $TARGET
   → ⚠️ EARLY_ALERT [BRUTE_FORCE] attempt#6   (~vài giây sau khi bắt đầu)
   → (chờ IDLE_GAP=30s sau khi hydra dừng)
   → 🔴 FINAL [KNOWN_ATTACK] Brute-force  (+ beep)

4. Kali:  ./break_in.sh $TARGET
   → ⚠️ EARLY_ALERT [BREAK_IN_RISK]
   → 🔴 FINAL [KNOWN_ATTACK] Break-in  (+ beep)

5. Kali:  ./multi_user_scan.sh $TARGET
   → ⚠️ EARLY_ALERT [SCAN] (nếu unique_users>=4 đạt trước)
   → ⚠️ EARLY_ALERT [BRUTE_FORCE] (nếu streak>=6 cũng đạt)
   → 🔴 FINAL [KNOWN_ATTACK] Brute-force  ← minh họa SCAN consistency=0% finding

6. Kali:  ./rapid_login.sh $TARGET
   → 🟢 hoặc 🟣 FINAL [NORMAL / UNKNOWN_PATTERN]  ← best-effort, cả 2 outcome OK

7. Ctrl+C → in tổng kết alerts.csv
```

---

## 6. Output Files

| File | Nội dung |
|---|---|
| `alerts.csv` | Tất cả alerts (EARLY + FINAL), structured |
| `alerts.jsonl` | Tương tự, JSON Lines — dễ feed vào tool khác |

---

## 7. Known Limitations (ghi trong báo cáo)

1. **IDLE_GAP=30s (demo) khác 600s (research)** — không ảnh hưởng feature computation,
   chỉ ảnh hưởng thời điểm đóng session. Document rõ cả 2 giá trị.

2. **`is_private` hardcode=0** — fix cho OOD do VM lab dùng private IP. Feature này
   đã xác nhận uninformative (importance=0) nên không mất thông tin.

3. **`has_reverse_mapping_failed`** — KHÔNG hardcode (feature có ý nghĩa, không
   constant trong training). Nếu Ubuntu có `UseDNS yes` và không resolve được
   reverse DNS cho IP Kali, giá trị này có thể =1 cho mọi session demo — khác
   với Normal training mean=0.033. Nếu thấy IF score thấp bất thường cho session
   NORMAL trong demo, đây là nguyên nhân khả dĩ — note lại, không cần fix.

4. **UNKNOWN_PATTERN không đảm bảo xuất hiện** — xem §4.5.

5. **`best_model.pkl` / `anomaly_detector.pkl` train trên dataset 2018, single-host**
   — live demo là cross-environment generalization test, chưa từng được claim
   trong phần evaluation offline. Kết quả demo là **minh họa kiến trúc**, không
   thay thế cho Macro F1 / FAR đã report.
