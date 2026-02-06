# NeuralOS

**AI-Native Embedded Linux for Edge Computing, Robotics & Quantum Applications**

[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-green.svg)](https://github.com/neuraparse/neuraos/releases)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/neuraparse/neuraos/actions)

---

## Overview

NeuralOS is a lightweight, production-ready embedded Linux distribution optimized for AI inference, robotics, drone systems, and quantum computing research. Built on Buildroot 2024.02.9 with Linux 6.12 LTS kernel featuring native PREEMPT_RT support.

### Key Specifications

| Component | Details |
|-----------|---------|
| **Kernel** | Linux 6.12.57 LTS (PREEMPT_RT) |
| **Architectures** | x86_64, ARM64 |
| **Memory** | 64MB minimum, 1GB recommended |
| **Storage** | 512MB rootfs |
| **Boot Time** | <5 seconds |

---

## Included Packages

### AI/ML Stack
- **llama.cpp** - LLM inference engine
- **emlearn** - Embedded ML algorithms
- **NumPy** - Scientific computing
- **Python 3.11** - Runtime environment

### Robotics & Drones
- **Fast-DDS 3.4.1** - Real-time pub/sub (ROS2 middleware)
- **MAVLink 2.0** - Drone communication protocol
- **FastCDR** - Serialization library

### Quantum Computing
- **QuEST** - Quantum circuit simulator

### System
- **Dropbear** - SSH server
- **BusyBox** - Core utilities
- **iptables** - Firewall
- **chrony** - NTP client

---

## Quick Start

### Prerequisites
- Docker (recommended) or Linux build environment
- 30GB free disk space
- 4GB RAM minimum

### Build (Docker)

```bash
git clone https://github.com/neuraparse/neuraos.git
cd neuraos

# Build x86_64 image
docker build -f Dockerfile.x86_64 -t neuraos-builder .
docker run --name neuraos-build neuraos-builder
docker cp neuraos-build:/neuraos/buildroot-2024.02.9/output/images ./neuraos-images
```

### Run with QEMU/KVM

```bash
# x86_64 with KVM acceleration
qemu-system-x86_64 -enable-kvm -cpu host -m 1024 -smp 2 \
  -kernel neuraos-images/bzImage \
  -drive file=neuraos-images/rootfs.ext2,format=raw,if=virtio \
  -append "root=/dev/vda rw console=ttyS0" \
  -netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::8080-:8080 \
  -device virtio-net-pci,netdev=net0 -nographic

# SSH access
ssh -p 2222 root@localhost  # Password: neuraos
```

### Web Control Panel

Access system metrics and terminal via web browser:
```
http://localhost:8080
```

---

## Project Structure

```
neuraos/
├── board/                    # Board support files
│   └── neuraparse/neuraos/
│       ├── rootfs_overlay/   # Filesystem overlay
│       └── scripts/          # Build scripts
├── configs/                  # Buildroot configurations
│   ├── neuraos_x86_64_defconfig
│   └── kernel/
├── package/neuraparse/       # Custom packages
│   ├── llama-cpp/
│   ├── fast-dds/
│   ├── quest/
│   └── ...
├── scripts/                  # Utility scripts
│   └── run_qemu_kvm.sh
├── Dockerfile.x86_64         # Docker build environment
└── Config.in                 # Package menu
```

---

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| x86_64 (KVM) | Tested | Production ready |
| ARM64 (QEMU) | Tested | Development/CI |
| Raspberry Pi 4/5 | Planned | Q2 2026 |
| NVIDIA Jetson | Planned | Q3 2026 |

---

## Development

### Adding Custom Packages

```bash
# Create package directory
mkdir -p package/neuraparse/mypackage

# Add Config.in and mypackage.mk files
# See existing packages for examples

# Rebuild
make mypackage-rebuild
```

### Kernel Configuration

```bash
cd buildroot-2024.02.9
make linux-menuconfig
make linux-update-defconfig
```

---

## Performance

Benchmarks on AMD EPYC 9355P:

| Test | Result |
|------|--------|
| Matrix Operations | 4.55 GFLOPS |
| Neural Network (MLP) | 5,325 inf/sec |
| Memory Bandwidth | 28.37 GB/s |

---

## Security

- Minimal attack surface (<100MB footprint)
- SSH with key-based authentication
- iptables firewall enabled by default
- No telemetry or cloud dependencies
- Offline operation capable

---

## License

GPL v2. See [LICENSE](LICENSE) for details.

Third-party components retain their original licenses (Apache 2.0, MIT, etc.).

---

## Links

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/neuraparse/neuraos/issues)
- **Website**: [neuraparse.com](https://neuraparse.com)

---

*Built by NeuraParse Team - 2026*
