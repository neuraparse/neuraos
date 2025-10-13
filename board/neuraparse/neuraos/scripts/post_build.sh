#!/bin/bash
#
# NeuralOS Post-Build Script
# This script is executed after the root filesystem is built
# but before the filesystem images are created.
#

set -e

TARGET_DIR=$1

echo "NeuralOS: Running post-build script"
echo "Target directory: $TARGET_DIR"

# Create necessary directories
mkdir -p "$TARGET_DIR/opt/neuraparse"
mkdir -p "$TARGET_DIR/opt/neuraparse/models"
mkdir -p "$TARGET_DIR/opt/neuraparse/examples"
mkdir -p "$TARGET_DIR/etc/npi"
mkdir -p "$TARGET_DIR/etc/npi/services"
mkdir -p "$TARGET_DIR/var/log"
mkdir -p "$TARGET_DIR/var/run"

# Create NPI init configuration
cat > "$TARGET_DIR/etc/npi/init.conf" << 'EOF'
# NeuralOS Init Configuration
# Version: 1.0.0-alpha

[system]
hostname=neuraos
timezone=UTC
loglevel=info

[boot]
timeout=30
fastboot=true
quiet=false

[ai]
enable_npie=true
auto_load_models=true
default_backend=auto
default_accelerator=auto
EOF

# Create welcome message
cat > "$TARGET_DIR/etc/motd" << 'EOF'

╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║              Welcome to NeuralOS v1.0.0-alpha                 ║
║          AI-Native Embedded Operating System                  ║
║                                                               ║
║                    by NeuraParse™                             ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

System Information:
  - Kernel: Linux 6.12 LTS with PREEMPT_RT
  - AI Runtime: NeuraParse Inference Engine (NPIE)
  - Supported Backends: LiteRT, ONNX Runtime, emlearn
  - Hardware Acceleration: Auto-detected

Quick Start:
  npie-cli status              - Check NPIE status
  npie-cli hardware list       - List available accelerators
  npie-cli model load <path>   - Load AI model
  npie-cli inference run       - Run inference

Documentation: /opt/neuraparse/docs/
Examples: /opt/neuraparse/examples/

For support: info@neuraparse.com

EOF

# Create network configuration
cat > "$TARGET_DIR/etc/network/interfaces" << 'EOF'
# Network interfaces configuration

auto lo
iface lo inet loopback

auto eth0
iface eth0 inet dhcp
    hostname neuraos

# WiFi configuration (if available)
# auto wlan0
# iface wlan0 inet dhcp
#     wpa-conf /etc/wpa_supplicant.conf
EOF

# Create wpa_supplicant template
cat > "$TARGET_DIR/etc/wpa_supplicant.conf.template" << 'EOF'
# WPA Supplicant Configuration Template
# Copy this file to /etc/wpa_supplicant.conf and edit

ctrl_interface=/var/run/wpa_supplicant
ctrl_interface_group=0
update_config=1

network={
    ssid="YourNetworkName"
    psk="YourPassword"
    key_mgmt=WPA-PSK
}
EOF

# Create fstab
cat > "$TARGET_DIR/etc/fstab" << 'EOF'
# <file system> <mount point>   <type>  <options>       <dump>  <pass>
proc            /proc           proc    defaults        0       0
sysfs           /sys            sysfs   defaults        0       0
devtmpfs        /dev            devtmpfs defaults       0       0
tmpfs           /tmp            tmpfs   size=64M        0       0
tmpfs           /run            tmpfs   size=32M        0       0
tmpfs           /var/log        tmpfs   size=16M        0       0
EOF

# Create inittab for BusyBox init
cat > "$TARGET_DIR/etc/inittab" << 'EOF'
# /etc/inittab

::sysinit:/bin/mount -t proc proc /proc
::sysinit:/bin/mount -t sysfs sysfs /sys
::sysinit:/bin/mount -t devtmpfs devtmpfs /dev
::sysinit:/bin/mkdir -p /dev/pts /dev/shm
::sysinit:/bin/mount -a
::sysinit:/bin/hostname -F /etc/hostname

# Start system logging
::respawn:/sbin/syslogd -n
::respawn:/sbin/klogd -n

# Start NPIE daemon
::respawn:/usr/bin/npie-daemon

# Start getty on console
::respawn:/sbin/getty -L console 0 vt100

# Stuff to do before rebooting
::shutdown:/bin/umount -a -r
::shutdown:/sbin/swapoff -a
EOF

# Create profile for shell environment
cat > "$TARGET_DIR/etc/profile" << 'EOF'
# /etc/profile

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/root
export PS1='\u@\h:\w\$ '

# NeuralOS environment
export NEURAOS_VERSION="1.0.0-alpha"
export NPIE_CONFIG="/etc/npie/config.json"
export NPIE_MODELS="/opt/neuraparse/models"

# Enable colored output
alias ls='ls --color=auto'
alias ll='ls -lh'
alias la='ls -lah'

# AI shortcuts
alias npie='npie-cli'
alias npi-status='npie-cli status'
alias npi-models='npie-cli model list'

# Show welcome message
if [ -f /etc/motd ]; then
    cat /etc/motd
fi
EOF

# Create NPIE configuration
cat > "$TARGET_DIR/etc/npie/config.json" << 'EOF'
{
  "version": "1.0.0",
  "runtime": {
    "backend": "auto",
    "accelerator": "auto",
    "num_threads": 0,
    "enable_profiling": false,
    "enable_caching": true
  },
  "models": {
    "auto_load": true,
    "model_dir": "/opt/neuraparse/models",
    "cache_dir": "/var/cache/npie"
  },
  "logging": {
    "level": "info",
    "file": "/var/log/npie.log",
    "max_size": "10M",
    "rotate": 3
  },
  "hardware": {
    "detect_on_startup": true,
    "prefer_accelerator": true,
    "fallback_to_cpu": true
  },
  "security": {
    "verify_models": true,
    "allow_unsigned": false,
    "model_encryption": false
  }
}
EOF

# Set permissions
chmod 755 "$TARGET_DIR/etc/init.d"/* 2>/dev/null || true
chmod 644 "$TARGET_DIR/etc/fstab"
chmod 644 "$TARGET_DIR/etc/inittab"
chmod 644 "$TARGET_DIR/etc/profile"
chmod 600 "$TARGET_DIR/etc/wpa_supplicant.conf.template"

# Create version file
cat > "$TARGET_DIR/etc/neuraos-release" << EOF
NEURAOS_VERSION=1.0.0-alpha
NEURAOS_CODENAME=Pioneer
NEURAOS_BUILD_DATE=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
NEURAOS_KERNEL=$(uname -r 2>/dev/null || echo "6.12-neuraos")
NEURAOS_ARCH=$(uname -m 2>/dev/null || echo "unknown")
EOF

# Create README in /opt/neuraparse
cat > "$TARGET_DIR/opt/neuraparse/README.txt" << 'EOF'
NeuralOS - NeuraParse AI Runtime
=================================

This directory contains the NeuraParse AI runtime components:

/opt/neuraparse/
├── models/          - Pre-installed AI models
├── examples/        - Example applications
├── docs/            - Documentation
└── README.txt       - This file

Getting Started:
1. Check system status: npie-cli status
2. List models: npie-cli model list
3. Load a model: npie-cli model load /opt/neuraparse/models/vision/mobilenet_v2.tflite
4. Run inference: npie-cli inference run --input image.jpg

For more information, visit: https://neuraparse.com
Support: info@neuraparse.com
EOF

# Clean up unnecessary files
rm -rf "$TARGET_DIR/usr/share/man" 2>/dev/null || true
rm -rf "$TARGET_DIR/usr/share/doc" 2>/dev/null || true
rm -rf "$TARGET_DIR/usr/share/info" 2>/dev/null || true

# Strip binaries to reduce size
if [ -d "$TARGET_DIR/usr/bin" ]; then
    find "$TARGET_DIR/usr/bin" -type f -executable -exec strip --strip-all {} \; 2>/dev/null || true
fi

if [ -d "$TARGET_DIR/usr/sbin" ]; then
    find "$TARGET_DIR/usr/sbin" -type f -executable -exec strip --strip-all {} \; 2>/dev/null || true
fi

echo "NeuralOS: Post-build script completed successfully"

