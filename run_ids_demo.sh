#!/bin/bash

# Tên Container và Docker Image
CONTAINER_NAME="ssh-ids-live"
IMAGE_NAME="huntervu1035/ssh-anomaly-detection-using-ml:latest"
AUTH_LOG_PATH="/var/log/auth.log"

echo "=================================================="
echo "   KHỞI CHẠY KHÔNG GIAN DEMO SSH ANOMALY IDS      "
echo "=================================================="

# 1. Kiểm tra quyền sudo (vì cần đọc /var/log/auth.log và chạy docker)
if [ "$EUID" -ne 0 ]; then
  echo "❌ Vui lòng chạy script này với quyền sudo: sudo $0"
  exit 1
fi

# 2. Kiểm tra xem file auth.log có tồn tại không
if [ ! -f "$AUTH_LOG_PATH" ]; then
  echo "⚠️ Không tìm thấy $AUTH_LOG_PATH. Đang tự động tạo file trống..."
  touch "$AUTH_LOG_PATH"
  chmod 640 "$AUTH_LOG_PATH"
fi

# 3. Kiểm tra và dọn dẹp container cũ nếu đang chạy trùng tên
if [ "$(docker ps -aq -f name=^${CONTAINER_NAME}$)" ]; then
    echo "🔄 Phát hiện container cũ trùng tên. Đang dọn dẹp..."
    docker rm -f $CONTAINER_NAME > /dev/null 2>&1
fi

echo "🚀 Đang kéo Image mới nhất và khởi chạy IDS..."
echo "--------------------------------------------------"

# 4. Chạy Docker container
# Sử dụng -it để hứng giao diện Rich Console khi các kịch bản từ Kali bắn sang
docker run -it \
  --name "$CONTAINER_NAME" \
  --network host \
  -v "$AUTH_LOG_PATH":"$AUTH_LOG_PATH" \
  "$IMAGE_NAME"

# 5. Thông báo sau khi tắt thoát container
echo "--------------------------------------------------"
echo "🛑 Đã dừng chương trình IDS Demo."