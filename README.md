# SSH Anomaly Detection Using Machine Learning

Hệ thống phát hiện xâm nhập SSH theo thời gian thực sử dụng Machine Learning, kết hợp mô hình phân loại có giám sát (Random Forest) và phát hiện bất thường không giám sát (Isolation Forest) theo kiến trúc hai lớp.

## Tổng quan kiến trúc

```
SSH auth.log  →  Log Parser  →  Session Buffer  →  EarlyAlertEngine (rule-based)
                                      ↓ (session closed)
                                 Layer 1: Random Forest (5-class)
                                      ↓
                                 Layer 2: Isolation Forest (anomaly scoring)
                                      ↓
                                 Console + alerts.csv + alerts.jsonl
```

### Phân loại đa lớp (Multi-Level Classification)

| Class | Tên | Mô tả |
|:---:|---|---|
| 0 | Normal | Tất cả events đều là đăng nhập thành công |
| 1 | Single failure | Failure đơn lẻ, không có pattern tấn công rõ ràng |
| 2 | Scan | Dò nhiều username khác nhau từ cùng một IP |
| 3 | Brute-force | Tấn công dồn dập liên tiếp vào cùng tài khoản |
| 4 | Break-in | Đăng nhập thành công SAU các lần failure trong cùng session |

### Hệ thống hai lớp (Dual-Layer Detection)

- **Layer 1 — Random Forest Classifier**: Phân loại session vào 5 lớp tấn công đã biết (class 0–4) với `class_weight="balanced"` để xử lý mất cân bằng dữ liệu.
- **Layer 2 — Isolation Forest**: Phát hiện các pattern bất thường chưa từng thấy trong dữ liệu huấn luyện (zero-day / unknown attack patterns).
- **EarlyAlertEngine**: Cảnh báo sớm theo luật (rule-based) khi phát hiện brute-force streak ≥ 6, scan ≥ 4 username, hoặc break-in risk ngay trong lúc session đang mở.

## Cấu trúc thư mục

```
SSH-anomaly-detection-using-ML/
├── main.py                    # Pipeline huấn luyện mô hình end-to-end
├── requirements.txt           # Thư viện Python
├── Dockerfile                 # Container image cho live IDS
├── run_ids_demo.sh            # Script demo chạy IDS bằng Docker
├── install_docker.sh          # Script cài đặt Docker trên Ubuntu
│
├── src/                       # Mã nguồn chính
│   ├── log_processing.py      # Parser cho SSH auth.log (syslog + RFC3339)
│   ├── data_labeling.py       # Sessionization và gán nhãn heuristic
│   ├── feature_engineering.py # Tiền xử lý đặc trưng (log1p, drop leakage)
│   ├── anomaly_detector.py    # Huấn luyện và scoring Isolation Forest
│   ├── realtime_simulator.py  # Mô phỏng streaming offline trên log tĩnh
│   └── live_ids.py            # Daemon real-time (tail -f auth.log)
│
├── notebooks/                 # Jupyter Notebooks phân tích
│   ├── EDA.ipynb              # Khám phá dữ liệu
│   ├── Feature_engineering.ipynb
│   ├── Modeling_supervised.ipynb
│   ├── Modeling_unsupervised.ipynb
│   ├── Modeling_RF_decoupling.ipynb
│   └── Evaluation.ipynb       # Đánh giá tổng hợp
│
├── data/
│   ├── raw/                   # Log SSH thô (SSH.log, SSH_2k.log)
│   └── processed/             # Dữ liệu đã xử lý (CSV)
│
├── models/                    # Mô hình đã huấn luyện (.pkl)
│   ├── best_model.pkl         # Random Forest Classifier
│   └── anomaly_detector.pkl   # Isolation Forest
│
├── results/                   # Kết quả đánh giá và báo cáo
│   ├── RandomForest/
│   ├── RandomForest_Decoupling/
│   ├── IsolationForest/
│   ├── Model_Comparison/
│   └── final/
│
└── report_latex/              # Báo cáo LaTeX
```

## Cài đặt

### Yêu cầu hệ thống

- Python ≥ 3.11
- pip hoặc venv
- (Tùy chọn) Docker cho deployment

### Cài đặt môi trường

```bash
# Clone repository
git clone https://github.com/huntervu1337/SSH-anomaly-detection-using-ML.git
cd SSH-anomaly-detection-using-ML

# Tạo môi trường ảo và cài đặt thư viện
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Huấn luyện mô hình

Nếu thư mục `models/` chưa có file `.pkl`, hoặc muốn huấn luyện lại từ đầu:

```bash
# Chạy pipeline đầy đủ trên SSH.log (~73MB, ~15 giây)
python main.py

# Hoặc chạy nhanh trên tập mẫu nhỏ để kiểm tra (~1 giây)
python main.py --quick
```

Pipeline sẽ tự động:
1. Parse raw log → 212,149 event records
2. Gom nhóm thành 3,916 sessions và gán nhãn
3. Tiền xử lý đặc trưng và tạo train/test split
4. Huấn luyện Random Forest (300 estimators, balanced weights)
5. Huấn luyện Isolation Forest (contamination = 0.02)
6. Lưu mô hình vào `models/`

## Sử dụng

### Chạy IDS real-time (trên máy chủ Linux)

```bash
# Chạy trực tiếp — theo dõi /var/log/auth.log theo thời gian thực
sudo .venv/bin/python src/live_ids.py

# Chạy với chế độ verbose (debug)
sudo .venv/bin/python src/live_ids.py --verbose

# Đọc toàn bộ auth.log từ đầu file (xử lý log lịch sử)
sudo .venv/bin/python src/live_ids.py --read-all --verbose

# Tuỳ chỉnh đầy đủ
sudo .venv/bin/python src/live_ids.py \
  --log-file /var/log/auth.log \
  --idle-gap 30 \
  --valid-users alice,bob \
  --verbose
```

#### Tham số dòng lệnh

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--read-all` | `False` | Đọc auth.log từ đầu file thay vì chỉ theo dõi dòng mới |
| `--verbose` | `False` | In chi tiết debug cho mỗi dòng log đọc được |
| `--log-file` | `/var/log/auth.log` | Đường dẫn file log xác thực |
| `--idle-gap` | `30` | Khoảng cách idle (giây) để kết thúc session |
| `--valid-users` | `alice` | Danh sách user hợp lệ, cách nhau bằng dấu phẩy |

#### Đầu ra

Khi phát hiện sự kiện SSH, hệ thống sẽ:

- **Hiển thị trên console** với emoji và màu sắc (rich):
  ```
  [19:55:53] ⚠️  EARLY_ALERT [BREAK_IN_RISK] IP=192.168.119.1 attempt#3
  [19:55:53] 🔴 FINAL [KNOWN_ATTACK] Break-in IP=192.168.119.1 attempts=3 dur=18s IF_score=0.1174
  [19:44:13] 🟢 FINAL [NORMAL] Normal IP=192.168.119.1 attempts=1 dur=0s IF_score=0.332
  ```

- **Ghi file CSV** (`alerts.csv`) và **JSONL** (`alerts.jsonl`) — flush real-time sau mỗi alert.

### Chạy bằng Docker

```bash
# Kéo image từ Docker Hub
docker pull huntervu1035/ssh-anomaly-detection-using-ml:latest

# Chạy container với bind mount vào auth.log
docker run -it \
  --name ssh-ids-live \
  --network host \
  -v /var/log/auth.log:/var/log/auth.log \
  huntervu1035/ssh-anomaly-detection-using-ml:latest

# Hoặc sử dụng script demo có sẵn
sudo bash run_ids_demo.sh
```

### Build Docker image từ source

```bash
docker build -t huntervu1035/ssh-anomaly-detection-using-ml:latest .
docker push huntervu1035/ssh-anomaly-detection-using-ml:latest
```

## Notebooks

| Notebook | Mô tả |
|---|---|
| `EDA.ipynb` | Khám phá dữ liệu thô, phân bố event types, thống kê IP |
| `Feature_engineering.ipynb` | Thiết kế đặc trưng session-level, phân tích tương quan |
| `Modeling_supervised.ipynb` | Huấn luyện và đánh giá Random Forest với cross-validation |
| `Modeling_unsupervised.ipynb` | Huấn luyện Isolation Forest, tìm ngưỡng contamination tối ưu |
| `Modeling_RF_decoupling.ipynb` | Thí nghiệm decoupling — loại bỏ `is_private` khỏi mô hình |
| `Evaluation.ipynb` | Đánh giá tổng hợp hai lớp, confusion matrix, so sánh mô hình |

## Công nghệ sử dụng

- **Machine Learning**: scikit-learn (RandomForest, IsolationForest), pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Real-time Processing**: Python threading, rich console
- **Containerization**: Docker
- **Log Parsing**: Hỗ trợ cả định dạng syslog cổ điển (`Dec 10 06:55:46`) và RFC3339/ISO 8601 (`2026-06-16T19:02:24.083906+07:00`)
