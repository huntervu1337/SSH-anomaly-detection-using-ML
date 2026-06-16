FROM python:3.12-slim

WORKDIR /app

# Cài đặt các công cụ hệ thống cần thiết (nếu có thư viện nào yêu cầu gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt thư viện Python từ file requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn (.py) và mô hình (.pkl) vào image
COPY src/ ./src/
COPY models/ ./models/

# Khởi chạy ứng dụng live_ids
ENTRYPOINT ["python", "src/live_ids.py"]