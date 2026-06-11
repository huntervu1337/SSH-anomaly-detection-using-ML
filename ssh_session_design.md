# SSH Log — Session Aggregation & Multi-Level Labeling Spec

## 1. Đầu vào

Danh sách records đã parse từ `SSHLogParser.parse_file()`, mỗi record là một `Dict[str, object]` với các fields:

```
user, is_private, is_failure, is_root, is_valid,
not_valid_count, ip_failure, ip_success, no_failure,
first, td, ts
```

---

## 2. Định nghĩa Session

**Grouping key:** `ip` (địa chỉ nguồn của event, lấy từ bước parse).

> **Lưu ý:** Parser hiện tại không lưu `ip` vào record. Cần bổ sung field `ip` vào output của `parse_line()` trước khi aggregation.

**Tách session:** Trong cùng một IP, nếu khoảng cách giữa hai event liên tiếp `ts[i+1] - ts[i] > IDLE_GAP` thì bắt đầu session mới.

```python
IDLE_GAP = 600  # giây (10 phút)
```

Mỗi session là một danh sách các records liên tiếp của cùng một IP, không có khoảng im lặng nào vượt quá `IDLE_GAP`.

---

## 3. Feature Aggregation

| Nhóm | Tên Feature | Vai trò trong ML | Ghi chú / Cách xử lý |
| :--- | :--- | :--- | :--- |
| **Identity** | ip | Drop (Loại bỏ khi train) | Chỉ giữ lại làm Metadata để AI Agent tra cứu/block IP sau này. |
| **Identity** | is_private | Giữ lại để train | Phân biệt IP nội bộ (quét cấu hình) vs IP ngoài Internet. |
| **Volume & Time** | total_attempts | Cân nhắc Drop | Tránh mô hình học vẹt luật cứng. |
| **Volume & Time** | session_duration | Giữ lại để train | Đoạn thời gian diễn ra chiến dịch tấn công. |
| **Volume & Time** | attempts_per_second | Giữ lại để train | Tốc độ dồn dập của tool tự động. |
| **Volume & Time** | is_single_event | Giữ lại để train | Cờ quan trọng để mô hình cô lập các session chỉ có 1 log. |
| **Ratios** | total_failures | Drop (Loại bỏ khi train) | Đã có failure_ratio đại diện, giữ lại sẽ bị lộ luật gán nhãn. |
| **Ratios** | total_successes | Drop (Loại bỏ khi train) | Buộc phải drop để AI không ăn gian được luật của Class 4 (Break-In). |
| **Ratios** | failure_ratio | Giữ lại để train | Đặc trưng cốt lõi để bắt Brute-force/Scan. |
| **Ratios** | unique_users_ratio | Giữ lại để train | Đặc trưng cốt lõi để phân biệt Scan (User liên tục đổi) vs Brute-force. |
| **Boolean Agg.** | has_root_attempt | Giữ lại để train | Mức độ nguy hiểm (nhắm vào tài khoản tối cao). |
| **Boolean Agg.** | has_valid_user_attempt | Giữ lại để train | Kẻ tấn công có thông tin tình báo nội bộ hay không. |
| **Boolean Agg.** | max_failure_streak | Giữ lại để train | Chuỗi fail liên tục lớn nhất nội trong session. |
| **Boolean Agg.** | invalid_user_attempts | Giữ lại để train | Số lần thử thất bại vào tài khoản không hợp lệ (từ parser). |
| **Boolean Agg.** | has_reverse_mapping_failed | Giữ lại để train | Dấu hiệu bất thường từ tầng mạng (DNS). |

## 4. Multi-Level Label

### Thứ tự severity (dùng `max()` để gán nhãn session)

| Class | Tên | Định nghĩa |
|---|---|---|
| `0` | Normal | Tất cả events đều là success (`total_failures == 0`) |
| `1` | Single failure | Failure đơn lẻ, không có pattern tấn công rõ ràng |
| `2` | Scan | Dò nhiều username khác nhau từ 1 IP |
| `3` | Brute-force | Tấn công dồn dập vào cùng 1 username |
| `4` | Break-in | Có ít nhất 1 success sau các failures trong cùng session |

### Logic gán nhãn từng record (trước khi aggregate)

```python
def label_record_multilevel(record: Dict[str, object]) -> int:
    is_failure      = int(record["is_failure"])
    is_valid        = int(record["is_valid"])
    is_private      = int(record["is_private"])
    no_failure      = int(record["no_failure"])
    not_valid_count = int(record["not_valid_count"])
    td              = int(record["td"])

    if is_failure == 0:
        return 0  # Normal / success

    # Break-in: success đến sau failure trong cùng IP — xử lý ở session level
    # (không detect được ở record level đơn lẻ)

    if no_failure > 5 and td < 5:
        return 3  # Brute-force: nhanh, liên tiếp

    if not_valid_count > 3:
        return 2  # Scan: nhiều user khác nhau

    return 1  # Single failure
```

### Logic gán nhãn session (sau aggregation)

```python
def label_session(session_records: List[Dict[str, object]]) -> int:
    # Break-in: có success sau ít nhất 1 failure
    has_failure = any(int(r["is_failure"]) == 1 for r in session_records)
    has_success = any(int(r["is_failure"]) == 0 for r in session_records)
    if has_failure and has_success:
        return 4  # Break-in

    # Lấy nhãn cao nhất trong session
    return max(label_record_multilevel(r) for r in session_records)
```

---

## 5. Output Schema

Mỗi session được đại diện bởi 1 row với schema sau:

```
ip, is_private,
total_attempts, session_duration, attempts_per_second, is_single_event,
total_failures, total_successes, failure_ratio,
unique_users_count, unique_users_ratio,
has_root_attempt, has_valid_user_attempt,
max_no_failure, max_not_valid_count,
ts_first, ts_last,
class
```

---

## 6. Lưu ý triển khai

- **Thêm `ip` vào `parse_line()`** — field này hiện không có trong record output, cần bổ sung trước khi chạy aggregation.
- **Sort theo `ts` trước khi group** — đảm bảo idle gap được tính đúng.
- **`attempts_per_second` không dùng cho `is_single_event = 1`** — nên set về `NaN` hoặc `0` và loại khỏi features khi train nếu cần.
- **Class imbalance vẫn còn** — sau aggregation, class 0 (Normal) sẽ tăng tương đối nhưng vẫn thiểu số. Dùng `class_weight='balanced'` khi train.

## Tổng hợp fix qua các vòng review

| Vấn đề | Fix |
| :--- | :--- |
| DEFAULT_VALID_USERS hardcode từ dataset khác | Thay bằng `infer_valid_users_from_file()` |
| root trong valid users | Bỏ, Rule 2 xử lý riêng |
| Double-counting authentication failure | Bỏ khỏi `_event_type()` |
| no_failure global thay vì per-IP | Chuyển vào `ip_state` |
| not first dùng int như bool | Đổi thành `first == 0` |
| Rule 3 trùng Rule 4 trong labeler | Xóa Rule 3 |
| max_no_failure cross-session leakage | Tính lại `max_failure_streak` per-session |
| max_not_valid_count cross-session leakage | Tính lại `invalid_user_attempts` per-session |
| Break-in false positive (2 user khác nhau) | Check user in `seen_failure_users` |
| Scan chỉ có 2 sessions | Hạ ngưỡng + dùng `unique_users_ratio` |
| attempts_per_second vô nghĩa khi single event | Zero-out khi `is_single_event=1` |