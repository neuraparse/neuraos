# NeuralOS by NeuraParse™
## *Next-Generation AI-Native Embedded Operating System*

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Version](https://img.shields.io/badge/version-1.0.0--alpha-orange.svg)](https://github.com/neuraparse/neuraos)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/neuraparse/neuraos)

---

## 🚀 Overview

**NeuralOS** is a cutting-edge, lightweight AI-native embedded operating system designed for edge AI devices, IoT systems, and intelligent embedded hardware. Built on modern 2025 technologies, it combines the power of Linux with state-of-the-art AI inference capabilities.

### Key Features

- **Ultra-Lightweight**: <64MB RAM, <256MB storage footprint
- **AI-First Design**: Native support for LiteRT, ONNX Runtime, and custom AI accelerators
- **Real-Time Capable**: Linux 6.12 LTS with PREEMPT_RT support
- **Multi-Architecture**: ARM (32/64-bit), x86_64, RISC-V
- **Hardware Acceleration**: NPU, GPU, TPU support (Mali, Qualcomm, Edge TPU, etc.)
- **Modern Networking**: eBPF/XDP for high-performance packet processing
- **Secure by Design**: Verified boot, encrypted models, minimal attack surface
- **WebAssembly Ready**: WASM runtime for portable AI applications

---

## 📋 System Architecture (2025 Edition)

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Base System** | Buildroot | 2025.08 LTS | Minimal Linux distribution builder |
| **Kernel** | Linux | 6.12 LTS | Real-time kernel with PREEMPT_RT |
| **AI Runtime** | LiteRT (TensorFlow Lite) | 2.18+ | Primary inference engine |
| **AI Runtime** | ONNX Runtime | 1.20+ | Cross-platform model support |
| **Classical ML** | emlearn | Latest | Lightweight ML algorithms |
| **Computer Vision** | OpenCV | 4.10+ | Minimal CV build |
| **C Library** | musl libc | 1.2.5+ | Lightweight standard library |
| **Init System** | Custom (npi) | 1.0 | Fast boot init system |
| **Networking** | eBPF/XDP | Kernel 6.12 | High-performance networking |
| **WASM Runtime** | WasmEdge | 0.14+ | WebAssembly for edge AI |
| **Container** | Podman Lite | 5.0+ | Lightweight containers |

### Performance Targets

- **Boot Time**: <5 seconds (power-on to login prompt) - Target: <3s for v1.0 final
- **Memory**:
  - Minimal boot: ~16MB
  - With services: ~22MB used (993MB total in QEMU)
  - Target: 32-64MB typical usage with AI workloads
- **Storage**:
  - Minimal rootfs: ~22MB used
  - Total image: 512MB (with room for models and data)
- **Inference Latency**: <50ms for MobileNet-class models (target)
- **Power**: 0.5-5W depending on workload (hardware dependent)

### Current System Specifications (v1.0.0-alpha)

**Kernel:**
- Linux 6.12.8 LTS with PREEMPT support
- Built: October 24, 2025
- Size: 8.6 MB (compressed kernel image)
- Features: PCI, VIRTIO, EXT4, networking, device management

**Rootfs:**
- Size: 512 MB (EXT4 filesystem)
- Used: ~22 MB (base system)
- Free: ~430 MB (for models, data, applications)
- C Library: musl libc 1.2.5
- Init: BusyBox init (custom NPI init planned)

**Included Packages:**
- BusyBox 1.36.1 (core utilities)
- Dropbear SSH server
- eudev 3.2.14 (device management)
- iptables (firewall)
- iproute2 (networking)
- GDB + GDB Server (debugging)
- util-linux libraries

**Network:**
- DHCP client (udhcpc)
- IPv4 + IPv6 support
- SSH server (Dropbear)
- Firewall (iptables)

**Boot Services:**
- syslogd (system logging)
- klogd (kernel logging)
- udevd (device management)
- networking (DHCP + routing)
- dropbear (SSH server)

---

## 🏗️ Project Structure

```
neuraos/
├── buildroot/                  # Buildroot submodule (2025.08 LTS)
├── configs/                    # Board and system configurations
│   ├── neuraos_defconfig      # Main NeuralOS configuration
│   ├── boards/                # Board-specific configs
│   │   ├── raspberrypi/
│   │   ├── jetson/
│   │   ├── generic_x86_64/
│   │   └── riscv/
│   └── kernel/                # Kernel configurations
│       └── neuraos_6.12_defconfig
├── board/                     # Board support files
│   └── neuraparse/
│       └── neuraos/
│           ├── rootfs_overlay/
│           ├── post_build.sh
│           └── post_image.sh
├── package/                   # Custom Buildroot packages
│   └── neuraparse/
│       ├── litert/           # LiteRT (TensorFlow Lite)
│       ├── onnxruntime/      # ONNX Runtime
│       ├── emlearn/          # emlearn package
│       ├── npie/             # NeuraParse Inference Engine
│       ├── wasmEdge/         # WebAssembly runtime
│       └── opencv-minimal/   # Minimal OpenCV
├── src/                       # Source code
│   ├── npie/                 # NeuraParse Inference Engine
│   │   ├── core/             # Core inference engine
│   │   ├── hal/              # Hardware Abstraction Layer
│   │   ├── scheduler/        # Inference scheduler
│   │   ├── memory/           # Memory manager
│   │   └── api/              # Public API
│   ├── npi/                  # NeuralOS init system
│   ├── drivers/              # Custom drivers
│   │   ├── npu/              # NPU drivers
│   │   └── accelerators/     # AI accelerator drivers
│   └── libs/                 # Shared libraries
├── tools/                     # Development tools
│   ├── npconvert/            # Model converter
│   ├── npprofiler/           # Performance profiler
│   ├── npsim/                # Simulator
│   ├── neuraos-build         # Build wrapper
│   ├── neuraos-flash         # Flash utility
│   └── neuraos-debug         # Debug utility
├── models/                    # Pre-trained AI models
│   ├── vision/
│   ├── nlp/
│   └── audio/
├── examples/                  # Example applications
│   ├── image_classification/
│   ├── object_detection/
│   ├── voice_recognition/
│   └── wasm_inference/
├── docs/                      # Documentation
│   ├── getting_started.md
│   ├── api_reference.md
│   ├── hardware_support.md
│   └── development_guide.md
├── tests/                     # Test suite
│   ├── unit/
│   ├── integration/
│   └── benchmarks/
├── scripts/                   # Build and utility scripts
│   ├── setup_environment.sh
│   ├── build_all.sh
│   └── create_sdk.sh
├── .github/                   # GitHub workflows
│   └── workflows/
│       ├── build.yml
│       └── test.yml
├── CMakeLists.txt            # Root CMake file
├── Makefile                  # Main Makefile
├── LICENSE                   # GPL v2 License
└── README.md                 # This file
```

---

## 🔧 Quick Start

### Prerequisites

**For Linux Users:**
- Ubuntu 22.04+ or Debian 12+ recommended
- 4GB+ RAM, 50GB+ free disk space
- Build tools: `build-essential`, `git`, `wget`, `cpio`, `unzip`, `rsync`, `bc`

**For macOS Users:**
- macOS 12+ (Monterey or later)
- Docker Desktop installed
- 8GB+ RAM, 50GB+ free disk space
- Homebrew (for QEMU installation)

### Build Instructions

#### Option 1: Native Linux Build

```bash
# Clone the repository
git clone https://github.com/neuraparse/neuraos.git
cd neuraos

# Install dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential wget cpio unzip rsync bc git \
    file python3 python3-dev libncurses5-dev libssl-dev

# Build the minimal system (no GUI)
./scripts/build_neuraos.sh

# Output will be in buildroot-2024.02.9/output/images/
```

#### Option 2: Docker Build (macOS or Linux)

```bash
# Clone the repository
git clone https://github.com/neuraparse/neuraos.git
cd neuraos

# Build Docker image
docker build -f Dockerfile.buildroot -t neuraos-buildroot .

# Run build in container
docker run -it --name neuraos-build neuraos-buildroot bash -c "./scripts/build_neuraos.sh"

# Copy images to host
docker cp neuraos-build:/neuraos/buildroot-2024.02.9/output/images ./neuraos-images

# Cleanup
docker stop neuraos-build
docker rm neuraos-build
```

### Testing with QEMU

#### Install QEMU

**Linux:**
```bash
sudo apt-get install qemu-system-aarch64
```

**macOS:**
```bash
brew install qemu
```

#### Run NeuralOS in QEMU

```bash
# Navigate to images directory
cd neuraos-images  # or buildroot-2024.02.9/output/images/

# Launch QEMU ARM64 virtual machine
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \
  -kernel Image \
  -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" \
  -nographic

# To exit QEMU: Press Ctrl+A, then X
```

### First Boot

```bash
# Default credentials
Username: root
Password: neuraos

# Check system information
uname -a
# Output: Linux neuraos 6.12.8-neuraos #1 SMP PREEMPT ... aarch64 GNU/Linux

# Check disk usage
df -h

# Check network
ifconfig

# Check memory
free -h

# Check running processes
ps aux
```

---

## 🎯 Supported Hardware & Installation

### Virtual Machines (Testing & Development)

#### QEMU ARM64 ✅ **Currently Supported**
- **Status**: Fully working
- **Architecture**: ARM64 (aarch64)
- **CPU**: Cortex-A57 emulation
- **RAM**: 512MB - 4GB
- **Storage**: virtio-blk
- **Network**: virtio-net
- **Use Case**: Development, testing, CI/CD

**Installation:**
```bash
# Already covered in Quick Start section above
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \
  -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" -nographic
```

---

### ARM Platforms

#### Raspberry Pi 4/5 🔄 **Planned**
- **Status**: In development
- **Architecture**: ARM64 (Cortex-A72/A76)
- **RAM**: 2GB/4GB/8GB
- **Storage**: microSD, USB, NVMe (Pi 5)
- **AI Accelerator**: Coral TPU via USB (optional)

**Installation (Coming Soon):**
```bash
# Flash to SD card
sudo dd if=neuraos-rpi4.img of=/dev/sdX bs=4M status=progress
sync

# Or use Raspberry Pi Imager
# Select "Custom OS" -> neuraos-rpi4.img
```

#### NVIDIA Jetson Nano/Orin Nano 🔄 **Planned**
- **Status**: Planned for v1.1
- **Architecture**: ARM64 (Cortex-A57/A78AE)
- **RAM**: 4GB/8GB
- **AI Accelerator**: NVIDIA GPU (128/1024 CUDA cores)
- **Use Case**: Computer vision, deep learning

**Installation (Coming Soon):**
```bash
# Flash using NVIDIA SDK Manager
# Or use custom flash script
sudo ./tools/neuraos-flash-jetson --device nano --image neuraos-jetson.img
```

#### BeagleBone AI-64 📋 **Future**
- **Status**: Planned for v2.0
- **Architecture**: ARM64 (Cortex-A72)
- **AI Accelerator**: TI C7x DSP + Deep Learning Accelerators

#### NXP i.MX8M Plus 📋 **Future**
- **Status**: Planned for v2.0
- **AI Accelerator**: NPU (2.3 TOPS)
- **Use Case**: Industrial IoT, automotive

#### Rockchip RK3588/RK3576 📋 **Future**
- **Status**: Planned for v2.0
- **AI Accelerator**: NPU (6 TOPS)
- **Use Case**: Edge AI servers, NAS

---

### x86_64 Platforms

#### Generic x86_64 PC 🔄 **Planned**
- **Status**: Planned for v1.1
- **Architecture**: x86_64
- **RAM**: 2GB+
- **Storage**: SATA, NVMe, USB
- **Use Case**: Development, edge servers

**Installation (Coming Soon):**
```bash
# Create bootable USB
sudo dd if=neuraos-x86_64.iso of=/dev/sdX bs=4M status=progress

# Or use Ventoy/Rufus
# Boot from USB and install to disk
```

#### Intel Atom/Celeron/Core 📋 **Future**
- **Status**: Planned for v1.2
- **AI Accelerator**: Intel Neural Compute Stick 2 (optional)

---

### RISC-V Platforms (Experimental)

#### StarFive VisionFive 2 📋 **Future**
- **Status**: Experimental (v2.0+)
- **Architecture**: RISC-V (RV64GC)
- **Use Case**: Research, education

---

### AI Accelerators Support

| Accelerator | Status | Interface | Performance |
|-------------|--------|-----------|-------------|
| **Google Coral Edge TPU** | 🔄 Planned v1.1 | USB/PCIe | 4 TOPS |
| **ARM Mali GPU** | 📋 Future v2.0 | SoC | Varies |
| **Qualcomm Hexagon NPU** | 📋 Future v2.0 | SoC | Varies |
| **Intel NCS2** | 📋 Future v1.2 | USB | 1 TOPS |
| **Hailo-8** | 📋 Future v2.0 | PCIe/M.2 | 26 TOPS |
| **NVIDIA GPU** | 🔄 Planned v1.1 | SoC/PCIe | Varies |

**Legend:**
- ✅ Fully working
- 🔄 In development
- 📋 Planned
- ❌ Not supported

---

## 📚 Documentation

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [Hardware Support](docs/hardware_support.md)
- [Development Guide](docs/development_guide.md)
- [Performance Tuning](docs/performance_tuning.md)
- [Security Best Practices](docs/security.md)

---

## 🛠️ Development

### Building Custom Packages

NeuralOS uses Buildroot's BR2_EXTERNAL mechanism for custom packages:

```bash
# Package structure
package/neuraparse/
├── npie/              # NeuraParse Inference Engine
│   ├── Config.in      # Buildroot config
│   ├── npie.mk        # Build recipe
│   └── npie.hash      # Package checksums
└── npi-init/          # Custom init system
    ├── Config.in
    ├── npi-init.mk
    └── npi-init.hash

# To add a new package:
# 1. Create package directory
mkdir -p package/neuraparse/mypackage

# 2. Create Config.in
cat > package/neuraparse/mypackage/Config.in << 'EOF'
config BR2_PACKAGE_MYPACKAGE
    bool "mypackage"
    help
      My custom package description
EOF

# 3. Create mypackage.mk (Buildroot recipe)
# 4. Add to package/neuraparse/Config.in
# 5. Rebuild
make mypackage-rebuild
```

### Kernel Configuration

```bash
# Modify kernel config
cd buildroot-2024.02.9
make linux-menuconfig

# Save changes
make linux-update-defconfig

# Copy to project
cp output/build/linux-6.12.8/.config ../configs/kernel/neuraos_defconfig
```

### Debugging

**QEMU with GDB:**
```bash
# Terminal 1: Start QEMU with GDB server
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \
  -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" \
  -nographic -s -S

# Terminal 2: Connect GDB
gdb-multiarch Image
(gdb) target remote :1234
(gdb) continue
```

**SSH Access:**
```bash
# QEMU with port forwarding
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \
  -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -nographic

# Connect from host
ssh -p 2222 root@localhost
# Password: neuraos
```

### Testing

```bash
# Run unit tests (when available)
make test

# Run integration tests
./tests/integration/run_all.sh

# Benchmark performance
./tests/benchmarks/run_benchmarks.sh
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas Needing Help

1. **Hardware Support**
   - Raspberry Pi 4/5 board support
   - Jetson Nano support
   - Device tree configurations

2. **AI/ML Integration**
   - LiteRT package completion
   - ONNX Runtime integration
   - Model optimization tools

3. **Documentation**
   - Hardware setup guides
   - API documentation
   - Tutorial videos

4. **Testing**
   - Hardware testing on real devices
   - Performance benchmarks
   - Bug reports

### Contribution Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your fork (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- **C/C++**: Follow Linux kernel coding style
- **Python**: PEP 8
- **Shell**: ShellCheck compliant
- **Commit messages**: Conventional Commits format

For detailed guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon).

---

## 📄 License

NeuralOS is licensed under GPL v2. See [LICENSE](LICENSE) for details.

Third-party components retain their original licenses:
- Buildroot: GPL v2
- Linux Kernel: GPL v2
- LiteRT: Apache 2.0
- ONNX Runtime: MIT
- emlearn: MIT
- OpenCV: Apache 2.0

---

## 🌟 Roadmap & Current Status

### Version 1.0.0-alpha (Current - October 2025)

**✅ Completed:**
- Buildroot 2024.02.9 foundation
- Linux 6.12.8 LTS with PREEMPT support
- ARM64 (aarch64) architecture support
- QEMU ARM64 virtual machine support
- Minimal rootfs (512MB EXT4)
- Basic networking (DHCP, SSH via Dropbear)
- Device management (eudev)
- Kernel configuration with PCI + VIRTIO support

**🔄 In Progress:**
- NPIE (NeuraParse Inference Engine) integration
- NPI custom init system
- LiteRT (TensorFlow Lite) package
- ONNX Runtime package
- emlearn integration

**📋 Planned for v1.0.0 Final:**
- Raspberry Pi 4/5 support
- Basic AI model inference
- Performance benchmarks
- Documentation completion
- Example applications

---

### Version 1.1.0 (Q1 2026)

**Planned Features:**
- 📋 WebAssembly runtime (WasmEdge) integration
- 📋 Google Coral Edge TPU support
- 📋 NVIDIA Jetson Nano/Orin support
- 📋 x86_64 platform support
- 📋 Web management interface
- 📋 Python SDK for model deployment
- 📋 GUI support (Wayland + Qt5)
- 📋 Enhanced networking (eBPF/XDP)

---

### Version 1.2.0 (Q2 2026)

**Planned Features:**
- 📋 Intel NCS2 support
- 📋 OpenCV integration
- 📋 Video streaming pipeline
- 📋 Model optimization tools
- 📋 OTA update system
- 📋 Container support (Podman Lite)

---

### Version 2.0.0 (Q3 2026)

**Long-term Goals:**
- 🚀 Distributed AI inference
- 🚀 Kubernetes integration
- 🚀 Real-time OS variant (PREEMPT_RT full)
- 🚀 Automotive/Industrial certifications
- 🚀 Federated learning support
- 🚀 Multi-NPU orchestration

---

## 📝 Known Issues & Limitations

### Current Limitations (v1.0.0-alpha)

1. **No GUI Support Yet**
   - Issue: Qt5/Wayland packages marked as legacy in Buildroot 2024.02.9
   - Workaround: Using minimal console-only build
   - Fix: Planned for v1.1.0 with updated packages

2. **NPIE Not Integrated**
   - Status: Package structure exists but not built yet
   - Reason: Focusing on stable base system first
   - Timeline: v1.0.0 final

3. **Limited Hardware Support**
   - Currently: QEMU ARM64 only
   - Next: Raspberry Pi 4/5 (v1.0.0 final)
   - Future: Jetson, x86_64, RISC-V

4. **No AI Accelerator Support**
   - Status: Software stack only
   - Planned: Coral TPU (v1.1.0), NVIDIA GPU (v1.1.0)

### Build Requirements

- **Disk Space**: ~30GB for full build
- **RAM**: 4GB minimum, 8GB recommended
- **Build Time**:
  - First build: ~30-45 minutes (depends on CPU)
  - Incremental: ~5-10 minutes
- **Network**: Required for downloading packages (~2GB)

---

## 🔧 Troubleshooting

### Build Issues

**Problem: Docker disk space error**
```bash
# Solution: Clean Docker cache
docker system prune -a -f
```

**Problem: Kernel panic - /dev/vda not found**
```bash
# Solution: Ensure PCI support is enabled in kernel config
# configs/kernel/neuraos_defconfig should have:
CONFIG_PCI=y
CONFIG_PCI_HOST_GENERIC=y
CONFIG_VIRTIO_PCI=y
```

**Problem: Build fails with "BR2_LEGACY=y"**
```bash
# Solution: Use minimal config without GUI
# Use configs/neuraos_minimal_defconfig instead
```

### Runtime Issues

**Problem: Cannot login**
```bash
# Default credentials:
Username: root
Password: neuraos
```

**Problem: No network**
```bash
# Check if DHCP is running
ps aux | grep udhcpc

# Manually request IP
udhcpc -i eth0
```

**Problem: QEMU won't start**
```bash
# Ensure QEMU is installed
qemu-system-aarch64 --version

# Check image paths are correct
ls -lh Image rootfs.ext4
```

---

## 📖 Quick Reference

### Essential Commands

**System Information:**
```bash
uname -a                    # Kernel version
cat /proc/cpuinfo          # CPU info
free -h                    # Memory usage
df -h                      # Disk usage
ps aux                     # Running processes
```

**Networking:**
```bash
ifconfig                   # Network interfaces
ip addr                    # IP addresses
ping 8.8.8.8              # Test connectivity
netstat -tulpn            # Open ports
```

**Package Management:**
```bash
# NeuralOS uses a read-only rootfs by default
# To install packages, rebuild the system with desired packages
```

**File System:**
```bash
ls -lh /                   # Root directory
du -sh /usr               # Directory size
mount                     # Mounted filesystems
```

**Logs:**
```bash
dmesg                     # Kernel messages
cat /var/log/messages     # System log
```

### Build Artifacts

After successful build, you'll find:

```
buildroot-2024.02.9/output/images/
├── Image              # Linux kernel (8.6 MB)
├── rootfs.ext2        # Root filesystem (512 MB)
├── rootfs.ext4        # Symlink to rootfs.ext2
├── rootfs.tar         # Compressed rootfs (22 MB)
└── u-boot.bin         # U-Boot bootloader (1.0 MB)
```

### Configuration Files

```
configs/
├── neuraos_minimal_defconfig       # Minimal build (no GUI)
├── neuraos_defconfig              # Full build (with GUI - WIP)
└── kernel/
    └── neuraos_defconfig          # Kernel configuration
```

### Useful QEMU Options

```bash
# Basic boot
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \
  -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" -nographic

# With SSH port forwarding
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 \
  -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 -nographic

# With more RAM
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 2048 \
  -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" -nographic

# With SMP (multiple CPUs)
qemu-system-aarch64 -M virt -cpu cortex-a57 -m 1024 -smp 4 \
  -kernel Image -drive file=rootfs.ext4,if=virtio,format=raw \
  -append "root=/dev/vda console=ttyAMA0" -nographic
```

---

## 📞 Contact & Support

- **Website**: https://neuraparse.com
- **Email**: info@neuraparse.com
- **GitHub**: https://github.com/neuraparse/neuraos
- **Issues**: https://github.com/neuraparse/neuraos/issues
- **Discussions**: https://github.com/neuraparse/neuraos/discussions

### Community

- **Discord**: Coming soon
- **Forum**: Coming soon
- **Twitter/X**: @neuraparse (coming soon)

---

## 🙏 Acknowledgments

NeuralOS is built on the shoulders of giants:

- **Buildroot** - The embedded Linux build system
- **Linux Kernel** - The world's most advanced operating system kernel
- **TensorFlow Lite** - Google's lightweight ML framework
- **ONNX Runtime** - Microsoft's cross-platform inference engine
- **QEMU** - The amazing emulator that makes development possible
- **musl libc** - Lightweight and fast C library
- **BusyBox** - The Swiss Army knife of embedded Linux

Special thanks to all open-source contributors who make projects like this possible.

---

**Built with ❤️ by NeuraParse Team**

*NeuralOS - Bringing AI to the Edge, One Device at a Time*

