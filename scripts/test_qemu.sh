#!/bin/bash
#
# Test NeuralOS in QEMU
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ROOTFS_DIR="$PROJECT_ROOT/output/rootfs"
INITRAMFS="$PROJECT_ROOT/output/initramfs.cpio.gz"

echo "NeuralOS QEMU Test"
echo "=================="
echo ""

# Check if rootfs exists
if [ ! -d "$ROOTFS_DIR" ]; then
    echo "Error: Rootfs not found. Run ./scripts/create_minimal_rootfs.sh first"
    exit 1
fi

# Create initramfs
echo "Creating initramfs..."
cd "$ROOTFS_DIR"
find . | cpio -H newc -o 2>/dev/null | gzip > "$INITRAMFS"
echo "Initramfs created: $INITRAMFS ($(du -h "$INITRAMFS" | cut -f1))"
echo ""

# Check for kernel
KERNEL=""
if [ -f "/boot/vmlinuz" ]; then
    KERNEL="/boot/vmlinuz"
elif [ -f "/boot/vmlinuz-$(uname -r)" ]; then
    KERNEL="/boot/vmlinuz-$(uname -r)"
else
    echo "Warning: No kernel found. QEMU will use built-in kernel."
    echo "For full boot test, you need a Linux kernel image."
    echo ""
fi

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64|amd64)
        QEMU_BIN="qemu-system-x86_64"
        QEMU_MACHINE="pc"
        QEMU_CPU="host"
        ;;
    aarch64|arm64)
        QEMU_BIN="qemu-system-aarch64"
        QEMU_MACHINE="virt"
        QEMU_CPU="cortex-a57"
        ;;
    armv7*|armhf)
        QEMU_BIN="qemu-system-arm"
        QEMU_MACHINE="virt"
        QEMU_CPU="cortex-a15"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

# Check if QEMU is available
if ! command -v $QEMU_BIN &> /dev/null; then
    echo "Error: $QEMU_BIN not found"
    echo "Install: sudo apt-get install qemu-system-arm qemu-system-aarch64"
    exit 1
fi

echo "Starting QEMU..."
echo "Architecture: $ARCH"
echo "QEMU: $QEMU_BIN"
echo "Machine: $QEMU_MACHINE"
echo ""
echo "Press Ctrl+A then X to exit QEMU"
echo "=================================="
echo ""

# QEMU command
if [ -n "$KERNEL" ]; then
    # Boot with real kernel
    $QEMU_BIN \
        -M $QEMU_MACHINE \
        -cpu $QEMU_CPU \
        -m 512 \
        -kernel "$KERNEL" \
        -initrd "$INITRAMFS" \
        -append "console=ttyAMA0 console=tty0 init=/sbin/init" \
        -nographic \
        -no-reboot
else
    # Test initramfs content only (no kernel boot)
    echo "Testing initramfs content (no kernel boot):"
    echo ""
    cd "$ROOTFS_DIR"
    echo "Directory structure:"
    ls -la
    echo ""
    echo "Init system:"
    if [ -f "sbin/init" ]; then
        file sbin/init
        echo ""
        echo "Init script content:"
        head -20 sbin/init
    else
        echo "Warning: No init found!"
    fi
fi

