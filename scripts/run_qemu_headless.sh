#!/bin/bash
#
# Run NeuralOS in QEMU (headless mode - no GUI)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILDROOT_DIR="$PROJECT_ROOT/buildroot-2024.02.9"
OUTPUT_DIR="$BUILDROOT_DIR/output/images"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          NeuralOS QEMU Boot (Headless Mode)                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if images exist
if [ ! -f "$OUTPUT_DIR/Image" ]; then
    echo "Error: Kernel image not found at $OUTPUT_DIR/Image"
    echo "Please run ./scripts/build_neuraos.sh first"
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/rootfs.ext4" ]; then
    echo "Error: Root filesystem not found at $OUTPUT_DIR/rootfs.ext4"
    echo "Please run ./scripts/build_neuraos.sh first"
    exit 1
fi

# Check for QEMU
if ! command -v qemu-system-aarch64 &> /dev/null; then
    echo "Error: qemu-system-aarch64 not found"
    echo "Install: brew install qemu (macOS) or apt-get install qemu-system-arm (Linux)"
    exit 1
fi

echo "Starting NeuralOS in QEMU (headless)..."
echo "Architecture: ARM64 (aarch64)"
echo "Memory: 1GB"
echo ""
echo "Press Ctrl+A then X to exit QEMU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run QEMU without display
qemu-system-aarch64 \
    -M virt \
    -cpu cortex-a57 \
    -m 1024 \
    -smp 2 \
    -kernel "$OUTPUT_DIR/Image" \
    -drive file="$OUTPUT_DIR/rootfs.ext4",if=virtio,format=raw \
    -append "root=/dev/vda rw console=ttyAMA0 init=/sbin/init" \
    -device virtio-net-pci,netdev=net0 \
    -netdev user,id=net0,hostfwd=tcp::2222-:22 \
    -nographic \
    -no-reboot

