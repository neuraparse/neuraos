#!/bin/bash
#
# NeuralOS Post-Image Script
# This script is executed after filesystem images are created
# to generate bootable SD card images and other deployment artifacts.
#

set -e

BOARD_DIR="$(dirname $0)/.."
IMAGES_DIR=$1

echo "NeuralOS: Running post-image script"
echo "Images directory: $IMAGES_DIR"

# Detect target architecture
if [ -f "$IMAGES_DIR/Image" ]; then
    KERNEL_IMAGE="Image"
    ARCH="arm64"
elif [ -f "$IMAGES_DIR/zImage" ]; then
    KERNEL_IMAGE="zImage"
    ARCH="arm"
elif [ -f "$IMAGES_DIR/bzImage" ]; then
    KERNEL_IMAGE="bzImage"
    ARCH="x86_64"
else
    echo "Warning: No kernel image found"
    KERNEL_IMAGE=""
    ARCH="unknown"
fi

echo "Detected architecture: $ARCH"
echo "Kernel image: $KERNEL_IMAGE"

# Create genimage configuration for SD card
if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "arm" ]; then
    cat > "$IMAGES_DIR/genimage.cfg" << EOF
# Genimage configuration for NeuralOS SD card image

image boot.vfat {
    vfat {
        files = {
            "$KERNEL_IMAGE",
            "bcm2711-rpi-4-b.dtb",
            "rpi-firmware"
        }
        file config.txt {
            image = "rpi-firmware/config.txt"
        }
    }
    size = 128M
}

image sdcard.img {
    hdimage {
        partition-table-type = "msdos"
    }

    partition boot {
        partition-type = 0xC
        bootable = "true"
        image = "boot.vfat"
    }

    partition rootfs {
        partition-type = 0x83
        image = "rootfs.ext4"
        size = 512M
    }
}
EOF

    # Create boot configuration for Raspberry Pi
    mkdir -p "$IMAGES_DIR/rpi-firmware"
    
    cat > "$IMAGES_DIR/rpi-firmware/config.txt" << 'EOF'
# NeuralOS Boot Configuration for Raspberry Pi

# Enable UART for serial console
enable_uart=1
uart_2ndstage=1

# GPU memory (minimal for headless)
gpu_mem=64

# Kernel
kernel=Image
arm_64bit=1

# Device tree
dtparam=i2c_arm=on
dtparam=spi=on

# Performance
arm_boost=1
over_voltage=2
arm_freq=1800

# Disable rainbow splash
disable_splash=1

# Enable camera (if needed)
# start_x=1
# gpu_mem=128
EOF

    cat > "$IMAGES_DIR/rpi-firmware/cmdline.txt" << 'EOF'
console=serial0,115200 console=tty1 root=/dev/mmcblk0p2 rootfstype=ext4 rootwait rw quiet init=/sbin/init
EOF

fi

# Create deployment package
DEPLOY_DIR="$IMAGES_DIR/deploy"
mkdir -p "$DEPLOY_DIR"

echo "Creating deployment package..."

# Copy images
if [ -n "$KERNEL_IMAGE" ] && [ -f "$IMAGES_DIR/$KERNEL_IMAGE" ]; then
    cp "$IMAGES_DIR/$KERNEL_IMAGE" "$DEPLOY_DIR/"
fi

if [ -f "$IMAGES_DIR/rootfs.squashfs" ]; then
    cp "$IMAGES_DIR/rootfs.squashfs" "$DEPLOY_DIR/"
fi

if [ -f "$IMAGES_DIR/rootfs.ext4" ]; then
    cp "$IMAGES_DIR/rootfs.ext4" "$DEPLOY_DIR/"
fi

if [ -f "$IMAGES_DIR/sdcard.img" ]; then
    cp "$IMAGES_DIR/sdcard.img" "$DEPLOY_DIR/"
fi

# Copy device tree blobs
if [ -d "$IMAGES_DIR" ]; then
    find "$IMAGES_DIR" -name "*.dtb" -exec cp {} "$DEPLOY_DIR/" \;
fi

# Create deployment README
cat > "$DEPLOY_DIR/README.txt" << EOF
NeuralOS Deployment Package
===========================

Version: 1.0.0-alpha
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Architecture: $ARCH

Files:
------
EOF

ls -lh "$DEPLOY_DIR" >> "$DEPLOY_DIR/README.txt"

cat >> "$DEPLOY_DIR/README.txt" << 'EOF'

Deployment Instructions:
------------------------

For SD Card (Raspberry Pi, BeagleBone, etc.):
  1. Insert SD card into your computer
  2. Identify the device (e.g., /dev/sdX)
  3. Flash the image:
     sudo dd if=sdcard.img of=/dev/sdX bs=4M status=progress conv=fsync
  4. Sync and eject:
     sync && sudo eject /dev/sdX

For Network Boot:
  1. Copy kernel and rootfs to TFTP server
  2. Configure U-Boot for network boot
  3. Boot device

For eMMC/Flash:
  1. Boot from SD card or USB
  2. Flash to eMMC:
     sudo dd if=sdcard.img of=/dev/mmcblk1 bs=4M status=progress conv=fsync

Default Credentials:
  Username: root
  Password: neuraos

First Boot:
  - System will auto-configure network (DHCP)
  - SSH server (dropbear) starts automatically
  - NPIE daemon starts automatically
  - Check status: npie-cli status

For more information: https://neuraparse.com
Support: info@neuraparse.com
EOF

# Create checksums
cd "$DEPLOY_DIR"
sha256sum * > SHA256SUMS 2>/dev/null || true
cd - > /dev/null

# Create compressed archive
ARCHIVE_NAME="neuraos-${ARCH}-$(date +%Y%m%d).tar.gz"
echo "Creating archive: $ARCHIVE_NAME"
tar -czf "$IMAGES_DIR/$ARCHIVE_NAME" -C "$IMAGES_DIR" deploy/

# Create build manifest
cat > "$IMAGES_DIR/build-manifest.txt" << EOF
NeuralOS Build Manifest
=======================

Build Information:
  Version: 1.0.0-alpha
  Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
  Architecture: $ARCH
  Kernel: Linux 6.12 LTS
  Buildroot: 2025.08 LTS

Components:
  - Base System: Buildroot with musl libc
  - Init System: NPI (NeuralOS Init)
  - AI Runtime: NPIE (NeuraParse Inference Engine)
  - Backends: LiteRT, ONNX Runtime, emlearn
  - Networking: eBPF/XDP support
  - Security: Secure boot ready, dm-verity support

Image Sizes:
EOF

if [ -f "$IMAGES_DIR/rootfs.squashfs" ]; then
    SIZE=$(du -h "$IMAGES_DIR/rootfs.squashfs" | cut -f1)
    echo "  Root FS (squashfs): $SIZE" >> "$IMAGES_DIR/build-manifest.txt"
fi

if [ -f "$IMAGES_DIR/rootfs.ext4" ]; then
    SIZE=$(du -h "$IMAGES_DIR/rootfs.ext4" | cut -f1)
    echo "  Root FS (ext4): $SIZE" >> "$IMAGES_DIR/build-manifest.txt"
fi

if [ -f "$IMAGES_DIR/sdcard.img" ]; then
    SIZE=$(du -h "$IMAGES_DIR/sdcard.img" | cut -f1)
    echo "  SD Card Image: $SIZE" >> "$IMAGES_DIR/build-manifest.txt"
fi

if [ -n "$KERNEL_IMAGE" ] && [ -f "$IMAGES_DIR/$KERNEL_IMAGE" ]; then
    SIZE=$(du -h "$IMAGES_DIR/$KERNEL_IMAGE" | cut -f1)
    echo "  Kernel: $SIZE" >> "$IMAGES_DIR/build-manifest.txt"
fi

cat >> "$IMAGES_DIR/build-manifest.txt" << 'EOF'

Deployment:
  - Flash sdcard.img to SD card or eMMC
  - Or use individual images for custom deployment

Documentation:
  - Getting Started: docs/getting_started.md
  - Architecture: docs/architecture_2025.md
  - API Reference: docs/api_reference.md

Support:
  - Email: info@neuraparse.com
  - GitHub: https://github.com/neuraparse/neuraos
EOF

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          NeuralOS Build Complete!                             ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files:"
echo "  - Deployment package: $DEPLOY_DIR/"
echo "  - Archive: $ARCHIVE_NAME"
echo "  - Manifest: build-manifest.txt"
echo ""
echo "Next steps:"
echo "  1. Flash to device: sudo dd if=sdcard.img of=/dev/sdX bs=4M"
echo "  2. Boot device and login (root/neuraos)"
echo "  3. Check status: npie-cli status"
echo ""

# Make script executable
chmod +x "$0"

echo "NeuralOS: Post-image script completed successfully"

