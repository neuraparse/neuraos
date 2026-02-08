# Getting Started with NeuralOS v5.0.0

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Build Profiles](#build-profiles)
5. [Building Your First Image](#building-your-first-image)
6. [Deploying to Hardware](#deploying-to-hardware)
7. [Running Your First AI Model](#running-your-first-ai-model)
8. [Drone Quick Start](#drone-quick-start)
9. [Swarm Simulation](#swarm-simulation)
10. [Troubleshooting](#troubleshooting)

---

## Introduction

NeuralOS v5.0.0 is an AI-native embedded operating system designed for autonomous drones, robotics platforms, and edge AI devices. This guide will walk you through setting up your development environment, building system images, and deploying to hardware.

### What You'll Learn

- How to set up the NeuralOS build environment
- How to choose build profiles (minimal, drone, robot, full)
- How to configure and build NeuralOS for your target hardware
- How to deploy NeuralOS to your device
- How to run AI inference on NeuralOS
- How to set up drone communication (MAVLink, PX4)
- How to run swarm simulations

---

## System Requirements

### Host System

**Minimum Requirements:**
- **OS**: Linux (Ubuntu 22.04+, Debian 12+, Fedora 38+, or Arch Linux)
- **CPU**: 4 cores (8+ recommended)
- **RAM**: 8GB (16GB+ recommended)
- **Disk**: 50GB free space (100GB+ recommended for multiple builds)
- **Internet**: Broadband connection for downloading packages

**Recommended Setup:**
- **OS**: Ubuntu 24.04 LTS
- **CPU**: 8+ cores
- **RAM**: 32GB
- **Disk**: 200GB SSD
- **Internet**: High-speed connection

### Target Hardware

NeuralOS supports a wide range of embedded platforms:

**ARM Platforms:**
- Raspberry Pi 3/4/5
- NVIDIA Jetson Nano/Orin Nano
- BeagleBone AI-64
- NXP i.MX8M Plus/i.MX93
- Rockchip RK3588/RK3576
- Qualcomm RB5/RB6

**x86_64 Platforms:**
- Intel Atom/Celeron/Core
- AMD Ryzen Embedded
- Generic x86_64 PCs

**RISC-V Platforms (Experimental):**
- StarFive VisionFive 2
- SiFive HiFive Unmatched

---

## Installation

### Step 1: Clone the Repository

```bash
git clone --recursive https://github.com/neuraparse/neuraos.git
cd neuraos
```

The `--recursive` flag ensures that all submodules (including Buildroot) are cloned.

### Step 2: Run Setup Script

```bash
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

This script will:
- Detect your Linux distribution
- Install all required dependencies
- Clone Buildroot (2025.08 LTS)
- Set up Python virtual environment
- Create necessary build directories
- Install git hooks

**Note:** The script may ask for your sudo password to install system packages.

### Step 3: Verify Installation

```bash
make check-deps
```

This will verify that all required tools are installed and accessible.

---

## Build Profiles

NeuralOS v5.0.0 supports build profiles to tailor the image to your use case:

```bash
make profile-minimal    # Core OS only (NPI init, NPIE, BusyBox)
make profile-drone      # Drone flight stack (MAVLink, PX4, Swarm, Sensors)
make profile-robot      # Robotics platform (DDS, Sensors, Navigation)
make profile-full       # Everything (AI + Drone + Desktop + Quantum)
```

You can also use CMake directly with fine-grained options:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MAVLINK=ON \
  -DBUILD_SWARM=ON \
  -DBUILD_OFFBOARD=ON \
  -DBUILD_SENSORS=ON \
  -DBUILD_CAMERA=ON \
  -DBUILD_NETWORK=ON \
  -DBUILD_SECURITY=ON \
  -DBUILD_OTA=ON

cmake --build build --parallel $(nproc)
```

---

## Building Your First Image

### Step 1: Choose Your Target Platform

NeuralOS provides pre-configured defconfigs for popular platforms:

**For Raspberry Pi 4/5:**
```bash
make rpi4       # Raspberry Pi 4
make rpi5       # Raspberry Pi 5
```

**For NVIDIA Jetson Orin Nano:**
```bash
make jetson-orin-nano
```

**For Pixhawk 6X (FMUv6X):**
```bash
make pixhawk6x
```

**For x86_64 PC:**
```bash
make x86_64
```

### Step 2: Customize Configuration (Optional)

If you want to customize the build:

```bash
make menuconfig           # Configure Buildroot
make linux-menuconfig     # Configure Linux kernel
make busybox-menuconfig   # Configure BusyBox
```

### Step 3: Build the System

```bash
make all -j$(nproc)
```

This will:
- Build the cross-compilation toolchain
- Compile the Linux kernel (6.12 LTS with PREEMPT_RT)
- Build all system packages
- Compile NeuraParse Inference Engine (NPIE)
- Generate the root filesystem
- Create bootable images

**Build Time:**
- First build: 30-90 minutes (depending on your hardware)
- Incremental builds: 2-10 minutes

### Step 4: Locate Output Files

After a successful build, you'll find the following in `output/buildroot/images/`:

- `Image` or `zImage` - Linux kernel image
- `rootfs.squashfs` - Compressed root filesystem
- `rootfs.ext4` - Root filesystem (ext4 format)
- `sdcard.img` - Complete SD card image (for ARM boards)
- `*.dtb` - Device tree blobs (for ARM boards)

---

## Deploying to Hardware

### Method 1: SD Card (Raspberry Pi, BeagleBone, etc.)

1. **Insert SD card** into your host computer

2. **Identify the device:**
   ```bash
   lsblk
   ```
   Look for your SD card (e.g., `/dev/sdb`, `/dev/mmcblk0`)

3. **Flash the image:**
   ```bash
   make flash DEVICE=/dev/sdX
   ```
   Replace `/dev/sdX` with your actual device.

   **⚠️ WARNING:** This will erase all data on the device!

4. **Safely eject:**
   ```bash
   sync
   sudo eject /dev/sdX
   ```

### Method 2: Network Boot (Development)

For faster development cycles, you can use network boot:

1. **Set up TFTP server** on your host
2. **Configure U-Boot** for network boot
3. **Boot from network:**
   ```bash
   make netboot
   ```

See [Development Guide](development_guide.md) for detailed instructions.

### Method 3: eMMC/Flash (Production)

For production deployments to eMMC or flash storage:

```bash
make flash-emmc DEVICE=/dev/mmcblkX
```

---

## Running Your First AI Model

### Step 1: Boot NeuralOS

1. Insert the SD card into your device
2. Connect serial console (optional but recommended)
3. Power on the device

**Default Credentials:**
- Username: `root`
- Password: `neuraos`

### Step 2: Verify System Status

```bash
# Check NPIE status
npie-cli status

# Check available accelerators
npie-cli hardware list

# Check system resources
free -h
df -h
```

### Step 3: Load a Pre-installed Model

NeuralOS comes with several pre-trained models:

```bash
# List available models
ls /opt/neuraparse/models/

# Load MobileNetV2 for image classification
npie-cli model load /opt/neuraparse/models/vision/mobilenet_v2.tflite --name mobilenet
```

### Step 4: Run Inference

**Image Classification:**
```bash
# Capture image from camera
v4l2-ctl --device=/dev/video0 --stream-mmap --stream-to=image.jpg --stream-count=1

# Run inference
npie-cli inference run mobilenet --input image.jpg --output result.json

# View results
cat result.json
```

**Object Detection:**
```bash
# Load object detection model
npie-cli model load /opt/neuraparse/models/vision/ssd_mobilenet.tflite --name detector

# Run detection
npie-cli inference run detector --input image.jpg --output detections.json

# Visualize results
npie-visualize detections.json image.jpg output.jpg
```

### Step 5: Monitor Performance

```bash
# Enable profiling
npie-cli config set profiling.enabled true

# Run inference with metrics
npie-cli inference run mobilenet --input image.jpg --metrics

# View detailed metrics
npie-cli metrics show
```

---

## Drone Quick Start

### Connect to PX4 Autopilot

```bash
# MAVLink over UART (Pixhawk)
mavlink-router -e 127.0.0.1:14550 /dev/ttyS1:921600

# MAVLink over UDP (SITL)
mavlink-router -e 127.0.0.1:14550 0.0.0.0:14540
```

### PX4 Offboard Control

```c
#include "neuraos_offboard.h"

// Connect to PX4
neura_offboard_cfg_t cfg = {
    .mavlink_url = "udp://127.0.0.1:14540",
    .system_id = 1,
    .component_id = 191
};
neura_offboard_ctx_t *ctx = neura_offboard_create(&cfg);

// Take off and go to position
neura_offboard_arm(ctx);
neura_offboard_takeoff(ctx, 10.0f);  // 10m altitude
neura_offboard_goto_position(ctx, 47.397742, 8.545594, 10.0f, 0.0f);
```

### Sensor Fusion

```bash
# Monitor EKF state
npie-cli sensor fusion status

# View sensor data
npie-cli sensor imu read
npie-cli sensor gps read
npie-cli sensor baro read
```

### Remote ID (ASTM F3411)

Remote ID broadcasts automatically when the drone is armed. Configure in `/etc/neuraos/remote_id.conf`.

---

## Swarm Simulation

### Start PX4 SITL Swarm

```bash
# Start 5-vehicle swarm simulation
./scripts/neuraos_sim.sh swarm 5

# Monitor swarm status
npie-cli swarm status

# Set V-formation
npie-cli swarm formation V --spacing 10.0

# Start area scan mission
npie-cli swarm mission area-scan \
  --min-lat 47.3970 --min-lon 8.5450 \
  --max-lat 47.3980 --max-lon 8.5460 \
  --altitude 20.0

# Stop simulation
./scripts/neuraos_sim.sh stop
```

### Swarm Configuration

Edit `configs/neuraos_swarm.conf` to customize:
- Vehicle count and type
- Formation spacing and type
- Safety parameters (geofence, altitude limits)
- Communication ports (DDS, ZMQ)
- Leader election timeouts

---

## Troubleshooting

### Build Issues

**Problem:** Build fails with "No rule to make target"
```bash
# Solution: Clean and rebuild
make clean
make all
```

**Problem:** Out of disk space
```bash
# Solution: Clean downloads and build artifacts
make distclean
# Free up space, then rebuild
```

**Problem:** Buildroot configuration errors
```bash
# Solution: Reset to default configuration
make neuraos_defconfig
```

### Boot Issues

**Problem:** Device doesn't boot
- Check SD card is properly inserted
- Verify image was flashed correctly
- Check serial console for error messages
- Try re-flashing the image

**Problem:** Kernel panic on boot
- Check device tree blob is correct for your hardware
- Verify kernel configuration matches your board
- Check serial console output for details

### Runtime Issues

**Problem:** NPIE fails to load model
```bash
# Check model file exists and is readable
ls -lh /path/to/model.tflite

# Check NPIE logs
journalctl -u npie-daemon

# Try loading with verbose output
npie-cli model load /path/to/model.tflite --verbose
```

**Problem:** Inference is slow
```bash
# Check if hardware acceleration is enabled
npie-cli hardware list

# Enable GPU acceleration
npie-cli config set accelerator gpu

# Increase thread count
npie-cli config set threads 4
```

**Problem:** Out of memory during inference
```bash
# Check available memory
free -h

# Reduce model size (use quantized model)
npie-convert model.tflite --quantize int8 --output model_int8.tflite

# Reduce batch size
npie-cli config set batch_size 1
```

### Network Issues

**Problem:** No network connectivity
```bash
# Check network interfaces
ip addr show

# For WiFi, configure wpa_supplicant
wpa_passphrase "SSID" "password" > /etc/wpa_supplicant.conf
wpa_supplicant -B -i wlan0 -c /etc/wpa_supplicant.conf
dhclient wlan0
```

**Problem:** SSH not working
```bash
# Check if dropbear is running
ps aux | grep dropbear

# Start dropbear manually
/etc/init.d/S50dropbear start

# Check firewall rules
iptables -L
```

---

## Next Steps

Now that you have NeuralOS running, explore these topics:

1. **[API Reference](api_reference.md)** - Learn the NPIE API in detail
2. **[Development Guide](development_guide.md)** - Build custom applications
3. **[Hardware Support](hardware_support.md)** - Add support for new hardware
4. **[Performance Tuning](performance_tuning.md)** - Optimize for your use case
5. **[Security Best Practices](security.md)** - Secure your deployment

---

## Getting Help

- **GitHub Issues**: https://github.com/neuraparse/neuraos/issues
- **Email Support**: info@neuraparse.com

---

**Happy Building! 🚀**

