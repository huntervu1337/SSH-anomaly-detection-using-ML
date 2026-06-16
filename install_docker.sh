#!/bin/bash

# Thoát script ngay lập tức nếu có lệnh nào bị lỗi
set -e

echo "=================================================="
echo "    BẮT ĐẦU CÀI ĐẶT DOCKER TRÊN UBUNTU            "
echo "=================================================="

# 1. Cập nhật danh sách gói và cài đặt các thư viện tiền đề
echo "🔄 Khởi tạo và cập nhật hệ thống..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 2. Thêm khóa GPG chính thức của Docker
echo "🔑 Thêm Docker GPG Key..."
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg --yes

# 3. Thiết lập Docker Repository cho Ubuntu
echo "📁 Cấu hình Docker Repository..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Cập nhật lại apt và cài đặt Docker Engine + Docker Compose
echo "📦 Đang tiến hành cài đặt Docker..."
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. Khởi động và kích hoạt Docker chạy cùng hệ thống
echo "🚀 Khởi chạy dịch vụ Docker..."
sudo systemctl start docker
sudo systemctl enable docker

# 6. Cấu hình phân quyền (Tùy chọn nhưng khuyến khích)
# Giúp bạn có thể chạy lệnh 'docker' trực tiếp mà không cần gõ 'sudo' sau này
echo "👤 Thêm user hiện tại vào nhóm docker..."
sudo usermod -aG docker $USER

echo "--------------------------------------------------"
echo "✅ Cài đặt thành công!"
echo "⚠️  LƯU Ý: Vui lòng ĐĂNG XUẤT (Log out) khởi máy ảo hoặc chạy lệnh: 'newgrp docker' để thay đổi phân quyền nhóm docker có hiệu lực."
echo "--------------------------------------------------"

# Kiểm tra phiên bản hiển thị
docker --version
docker compose version