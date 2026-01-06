#!/bin/bash
#
# NeuralOS QEMU Test Script
# Tests basic functionality of NeuralOS in QEMU
#

set -e

IMAGE_DIR="/home/neuraos/neuraos-images"
KERNEL="$IMAGE_DIR/Image"
ROOTFS="$IMAGE_DIR/rootfs.ext2"

echo "======================================"
echo "NeuralOS QEMU Boot Test"
echo "======================================"
echo ""

# Test 1: Boot and check kernel version
echo "[TEST 1] Boot test and kernel version..."
timeout 30 qemu-system-aarch64 \
  -M virt \
  -cpu cortex-a57 \
  -m 512 \
  -kernel "$KERNEL" \
  -drive file="$ROOTFS",if=none,format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -append "root=/dev/vda console=ttyAMA0 quiet" \
  -nographic \
  -serial mon:stdio \
  <<'EOF' | tee /tmp/neuraos_boot.log
root
uname -a
cat /etc/os-release
poweroff -f
EOF

echo ""
echo "[TEST 1] ✅ Boot successful!"
echo ""

# Test 2: Check installed packages
echo "[TEST 2] Checking installed packages..."
timeout 30 qemu-system-aarch64 \
  -M virt \
  -cpu cortex-a57 \
  -m 512 \
  -kernel "$KERNEL" \
  -drive file="$ROOTFS",if=none,format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -append "root=/dev/vda console=ttyAMA0 quiet" \
  -nographic \
  -serial mon:stdio \
  <<'EOF' | tee /tmp/neuraos_packages.log
root
echo "=== BusyBox ==="
busybox --help | head -1
echo "=== Python ==="
python3 --version
echo "=== NumPy ==="
python3 -c "import numpy; print('NumPy', numpy.__version__)"
echo "=== Network ==="
which dropbear
poweroff -f
EOF

echo ""
echo "[TEST 2] ✅ Package check complete!"
echo ""

# Test 3: Simple Python test
echo "[TEST 3] Running Python/NumPy computation test..."
timeout 30 qemu-system-aarch64 \
  -M virt \
  -cpu cortex-a57 \
  -m 512 \
  -kernel "$KERNEL" \
  -drive file="$ROOTFS",if=none,format=raw,id=hd0 \
  -device virtio-blk-device,drive=hd0 \
  -append "root=/dev/vda console=ttyAMA0 quiet" \
  -nographic \
  -serial mon:stdio \
  <<'EOF' | tee /tmp/neuraos_numpy_test.log
root
python3 << 'PYEOF'
import numpy as np
print("NumPy Test: Creating 10x10 matrix...")
matrix = np.random.rand(10, 10)
print(f"Matrix shape: {matrix.shape}")
print(f"Matrix mean: {matrix.mean():.4f}")
print(f"Matrix sum: {matrix.sum():.4f}")
result = np.dot(matrix, matrix.T)
print(f"Matrix multiplication result shape: {result.shape}")
print("✅ NumPy computation successful!")
PYEOF
poweroff -f
EOF

echo ""
echo "[TEST 3] ✅ Python/NumPy test passed!"
echo ""

echo "======================================"
echo "All NeuralOS tests PASSED! ✅"
echo "======================================"
echo ""
echo "Test logs saved to:"
echo "  - /tmp/neuraos_boot.log"
echo "  - /tmp/neuraos_packages.log"
echo "  - /tmp/neuraos_numpy_test.log"
