import joblib
from realtime_simulator import run_simulation

# Load models
rf_model = joblib.load("D:/SSH-anomaly-detection-using-ML/models/best_model.pkl")
if_model = joblib.load("D:/SSH-anomaly-detection-using-ML/models/anomaly_detector.pkl")

alerts = run_simulation("D:/SSH-anomaly-detection-using-ML/data/raw/SSH.log", rf_model, if_model)

finals = [a for a in alerts if a["alert_kind"] == "FINAL_CLASSIFICATION"]
early  = [a for a in alerts if a["alert_kind"] == "EARLY_ALERT"]

# Kiểm tra bug đã fix: mỗi session chỉ có had_early_alert=1 nếu thực sự fire
had_early_finals = [a for a in finals if a["had_early_alert"]]
print(f"Finals with early alert: {len(had_early_finals)}")

# Số sessions unique có early alert
early_session_keys = {(a["ip"], a["ts_first"]) for a in early}
print(f"Unique sessions with early alert: {len(early_session_keys)}")
# Expected: 1060 (Hai số này phải bằng nhau)

# Breakdown theo subtype
from collections import Counter
subtype_counts = Counter(a["alert_subtype"] for a in early)
print("Subtype breakdown:", dict(subtype_counts))