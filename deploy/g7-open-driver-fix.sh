#!/bin/bash
# ============================================================
# g7 (RTX PRO 4500 / Blackwell) 节点 NVIDIA 驱动修复脚本
# 作用：把 EKS GPU AMI 的闭源内核模块切换为 open kernel modules
# 适用：Amazon Linux 2023 x86_64 NVIDIA AMI (R580 驱动)
# 用法：
#   方式 A（现有节点手动修复）：
#     sudo bash g7-open-driver-fix.sh
#   方式 B（新节点自动修复）：把脚本内容嵌入节点组 userdata（见
#     g7-userdata.sh 模板），新节点启动自动执行
# ============================================================
set -x
KVER=$(uname -r)
LOG=/var/log/g7-open-driver-fix.log
exec > >(tee -a $LOG) 2>&1

echo "================================================"
echo "[$(date)] 开始切换 open kernel modules, kernel=$KVER"
echo "================================================"

# 1. 卸载闭源内核模块，安装 open 内核模块 + 用户态组件（同版本 580.x）
#    --allowerasing 自动解决 kmod-nvidia-latest-dkms 与 kmod-nvidia-open-dkms 冲突
#    nvidia-open 提供 nvidia-smi / libnvidia-ml.so 等用户态工具（EKS AMI 同版本）
echo "[1/5] 卸载闭源 + 安装 open 内核模块与用户态组件..."
dnf install -y --allowerasing kmod-nvidia-open-dkms nvidia-open || {
  echo "[ERROR] dnf 安装 kmod-nvidia-open-dkms / nvidia-open 失败"
  exit 1
}

# 2. 手动编译 open 内核模块
#    不用 dkms（dkms 3.4.1 存在 MAKE 解析 bug: make module 而非 modules）；
#    直接用 AMI 自带的 /usr/src/kernels/<KVER> 内核源码树编译
SRC=/usr/src/nvidia-580.178.04
echo "[2/5] 编译 open 内核模块 ($SRC)..."
if [ ! -d "$SRC" ]; then
  echo "[ERROR] 未找到源码目录 $SRC"
  exit 1
fi
cd "$SRC" || exit 1
make -j"$(nproc)" KERNEL_UNAME="$KVER" \
  IGNORE_PREEMPT_RT_PRESENCE=1 IGNORE_XEN_PRESENCE=1 modules || {
  echo "[ERROR] make modules 失败，参见上面的编译输出"
  exit 1
}

# 3. 安装模块到内核模块目录 + depmod
echo "[3/5] 安装模块..."
for m in nvidia nvidia-modeset nvidia-drm nvidia-uvm nvidia-peermem; do
  cp -v "kernel-open/$m.ko" "/usr/lib/modules/$KVER/extra/$m.ko"
done
depmod -a

# 4. 开机自动加载 open 模块（重启后生效）
echo "[4/5] 写入 /etc/modules-load.d/nvidia.conf ..."
printf 'nvidia\nnvidia-modeset\nnvidia-uvm\nnvidia-drm\n' > /etc/modules-load.d/nvidia.conf
cat /etc/modules-load.d/nvidia.conf

# 5. 立即切换到 open 模块（无需重启；若 rmmod 失败可重启节点）
echo "[5/5] 重载内核模块为 open 版本..."
rmmod nvidia_uvm 2>/dev/null || true
rmmod nvidia_drm 2>/dev/null || true
rmmod nvidia_modeset 2>/dev/null || true
rmmod nvidia 2>/dev/null || true
modprobe nvidia && modprobe nvidia-modeset && modprobe nvidia-uvm && modprobe nvidia-drm
echo "------------------------------------------------"
echo "[$(date)] 完成。验证:"
nvidia-smi | head -12
echo "------------------------------------------------"
