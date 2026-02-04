#!/bin/bash
#
# Run NeuralOS x86_64 with KVM acceleration
# Safe for production servers - fully isolated from host
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/neuraos-images-x86_64"

# Configuration (can be overridden via environment)
MEMORY="${MEMORY:-1024}"
CPUS="${CPUS:-2}"
SSH_PORT="${SSH_PORT:-2222}"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║       NeuralOS x86_64 with KVM Acceleration                  ║"
echo "║       Production-Safe Isolated Environment                    ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check KVM availability
if [ ! -r /dev/kvm ]; then
    echo "[ERROR] /dev/kvm not accessible"
    echo "  Check: ls -la /dev/kvm"
    echo "  Fix:   sudo chmod 666 /dev/kvm"
    echo "     or: sudo usermod -aG kvm $USER"
    exit 1
fi

# Find kernel image
KERNEL=""
if [ -f "$OUTPUT_DIR/bzImage" ]; then
    KERNEL="$OUTPUT_DIR/bzImage"
elif [ -f "$OUTPUT_DIR/vmlinux" ]; then
    KERNEL="$OUTPUT_DIR/vmlinux"
else
    echo "[ERROR] Kernel not found in $OUTPUT_DIR"
    echo "Run: ./scripts/build_x86_64.sh first"
    exit 1
fi

# Find rootfs
ROOTFS=""
if [ -f "$OUTPUT_DIR/rootfs.ext4" ]; then
    ROOTFS="$OUTPUT_DIR/rootfs.ext4"
elif [ -f "$OUTPUT_DIR/rootfs.ext2" ]; then
    ROOTFS="$OUTPUT_DIR/rootfs.ext2"
else
    echo "[ERROR] Rootfs not found in $OUTPUT_DIR"
    exit 1
fi

echo "[INFO] Configuration:"
echo "  Kernel:    $KERNEL"
echo "  Rootfs:    $ROOTFS"
echo "  Memory:    ${MEMORY}MB"
echo "  CPUs:      $CPUS"
echo "  SSH Port:  localhost:$SSH_PORT -> guest:22"
echo "  KVM:       Enabled (hardware acceleration)"
echo ""
echo "[INFO] Security:"
echo "  - Isolated from host filesystem"
echo "  - User-mode networking (no bridge)"
echo "  - No access to host Docker"
echo ""
echo "Credentials: root / neuraos"
echo "SSH: ssh -p $SSH_PORT root@localhost"
echo "Exit: Ctrl+A then X"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run QEMU with KVM
exec qemu-system-x86_64 \
    -enable-kvm \
    -cpu host \
    -m "$MEMORY" \
    -smp "$CPUS" \
    -kernel "$KERNEL" \
    -drive file="$ROOTFS",format=raw,if=virtio \
    -append "root=/dev/vda rw console=ttyS0 init=/sbin/init loglevel=5" \
    -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 \
    -device virtio-net-pci,netdev=net0 \
    -nographic \
    -no-reboot
