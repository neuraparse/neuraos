#!/bin/bash
#
# Run NeuralOS in QEMU (headless mode - no GUI)
# Updated: February 2026
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Try multiple image locations
if [ -d "$PROJECT_ROOT/neuraos-images" ] && [ -f "$PROJECT_ROOT/neuraos-images/Image" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/neuraos-images"
elif [ -d "$PROJECT_ROOT/buildroot-2024.02.9/output/images" ]; then
    OUTPUT_DIR="$PROJECT_ROOT/buildroot-2024.02.9/output/images"
else
    OUTPUT_DIR="$PROJECT_ROOT/output/images"
fi

# Memory size (default 1024MB)
MEMORY="${MEMORY:-1024}"
# CPU count (default 2)
CPUS="${CPUS:-2}"
# SSH port forward (default 2222)
SSH_PORT="${SSH_PORT:-2222}"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          NeuralOS QEMU Boot (Headless Mode)                   ║"
echo "║          Version: 1.0.0-alpha                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if images exist
if [ ! -f "$OUTPUT_DIR/Image" ]; then
    echo "Error: Kernel image not found"
    echo "Searched in: $OUTPUT_DIR/Image"
    echo "Please run ./scripts/build_neuraos.sh first"
    exit 1
fi

# Support both ext2 and ext4 rootfs
ROOTFS=""
if [ -f "$OUTPUT_DIR/rootfs.ext4" ]; then
    ROOTFS="$OUTPUT_DIR/rootfs.ext4"
elif [ -f "$OUTPUT_DIR/rootfs.ext2" ]; then
    ROOTFS="$OUTPUT_DIR/rootfs.ext2"
else
    echo "Error: Root filesystem not found"
    echo "Searched in: $OUTPUT_DIR/rootfs.ext4 or rootfs.ext2"
    exit 1
fi

# Check for QEMU
if ! command -v qemu-system-aarch64 &> /dev/null; then
    echo "Error: qemu-system-aarch64 not found"
    echo "Install: brew install qemu (macOS) or apt-get install qemu-system-arm (Linux)"
    exit 1
fi

echo "Starting NeuralOS in QEMU (headless)..."
echo "  Kernel:    $OUTPUT_DIR/Image"
echo "  Rootfs:    $ROOTFS"
echo "  Arch:      ARM64 (aarch64, Cortex-A57)"
echo "  Memory:    ${MEMORY}MB"
echo "  CPUs:      $CPUS"
echo "  SSH Port:  localhost:$SSH_PORT -> guest:22"
echo ""
echo "Press Ctrl+A then X to exit QEMU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run QEMU without display (compatible with QEMU 6.x+)
qemu-system-aarch64 \
    -machine virt \
    -cpu cortex-a57 \
    -m "$MEMORY" \
    -smp "$CPUS" \
    -kernel "$OUTPUT_DIR/Image" \
    -drive if=none,file="$ROOTFS",id=rootdisk,format=raw \
    -device virtio-blk-device,drive=rootdisk \
    -append "root=/dev/vda rw console=ttyAMA0 init=/sbin/init loglevel=5" \
    -device virtio-net-device,netdev=net0 \
    -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 \
    -nographic \
    -no-reboot

