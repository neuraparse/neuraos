<h1 align="center">NeuralOS</h1>

<h3 align="center">AI-Native Embedded Linux for Drones, Robotics, Edge AI & Quantum</h3>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Proprietary-red.svg?style=for-the-badge&logo=lock" alt="License"/></a>
  <a href="https://github.com/neuraparse/neuraos/releases"><img src="https://img.shields.io/badge/Version-5.0.0-green.svg?style=for-the-badge&logo=v" alt="Version"/></a>
  <a href="https://github.com/neuraparse/neuraos/actions"><img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg?style=for-the-badge&logo=githubactions" alt="Build"/></a>
  <a href="https://github.com/neuraparse/neuraos/stargazers"><img src="https://img.shields.io/github/stars/neuraparse/neuraos?style=for-the-badge&logo=github" alt="Stars"/></a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-drone--robotics-stack">Drone Stack</a> •
  <a href="#-desktop-shell">Desktop Shell</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-packages">Packages</a> •
  <a href="#-benchmarks">Benchmarks</a> •
  <a href="#-docs">Docs</a>
</p>

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 Ultra-Lightweight
```
┌─────────────────────────┐
│  Memory:     64MB min   │
│  Storage:    512MB      │
│  Boot Time:  <5 sec     │
└─────────────────────────┘
```
Minimal footprint for resource-constrained edge devices.

</td>
<td width="50%">

### ⚡ Real-Time Performance
```
┌─────────────────────────┐
│  Kernel:  6.12 LTS      │
│  RT:      PREEMPT_RT    │
│  Latency: <1ms          │
└─────────────────────────┘
```
Native real-time support for robotics and drones.

</td>
</tr>
<tr>
<td width="50%">

### 🤖 AI-First Design
```
┌─────────────────────────┐
│  LLM:     llama.cpp     │
│  Vision:  NCNN+OpenCV   │
│  Speech:  whisper.cpp   │
│  ML:      12 backends   │
└─────────────────────────┘
```
Built-in AI inference with LLM, vision, speech & quantum computing.

</td>
<td width="50%">

### 🔒 Security-Hardened
```
┌─────────────────────────┐
│  Secure Boot, dm-verity │
│  AES-256-GCM Models     │
│  WireGuard VPN          │
│  RAUC OTA               │
└─────────────────────────┘
```
Zero telemetry, minimal attack surface.

</td>
</tr>
</table>

---

## 🚁 Drone & Robotics Stack

| Subsystem | Details |
|-----------|---------|
| MAVLink 2.0 | Full protocol with message signing, UART/UDP transport |
| PX4 Offboard | Position, velocity, attitude command modes |
| Swarm Engine | V/Line/Circle/Grid formations, Raft leader election, DDS+ZMQ |
| Sensor Fusion | 16-state EKF (IMU, GPS, Baro, Mag) |
| Camera | V4L2 abstraction with mmap zero-copy |
| Remote ID | ASTM F3411 compliance |
| WiFi Mesh | 802.11s + BATMAN-adv |
| eBPF QoS | XDP packet classification, MAVLink priority |
| Cellular | 4G/5G modem via QMI/MBIM |
| WireGuard | VPN tunnel management |
| Telemetry | Delta + LZ4 compression |

---

## 🖥️ Desktop Shell

NeuralOS includes a Qt5 QML desktop environment with glassmorphism design, floating dock taskbar, and **33 built-in applications**.

**Shell Components:**
- Floating dock taskbar with app indicators
- Start menu with search, pinned apps, and app grid
- Notification center with quick toggles and sliders
- **Command Palette** (Ctrl+K) — Natural language OS control
- Desktop widgets (clock, system stats, weather, media, calendar)
- Window manager with drag, resize, minimize, maximize, close

**Applications:**

| Category | Apps |
|----------|------|
| System | System Monitor, Terminal, File Manager, Settings, Task Manager, Package Manager, Network Center, **Ecosystem** |
| AI & ML | Neural Studio, AI Agent Hub, AI Assistant, NPU Control Center, **AI Bus**, **AI Memory**, **Automation Studio**, **MCP Hub**, **Knowledge Base** |
| Utilities | Calculator, Text Editor, Notes, Calendar, Clock, Weather, Photos |
| Media | Music Player, Video Player, Image Viewer |
| Internet | Web Browser, App Store |
| Defense | Drone Command Center, Defense Monitor |
| Quantum | **Quantum Lab** (full statevector simulation, 13 gates, 7 presets) |

**AI-Inspired Features** (inspired by Archon OS, Kuse AI, Fuchsia, WarmWind, Bytebot, Steve OS, CosmOS):

| Feature | Inspiration | Description |
|---------|-------------|-------------|
| AI Bus Orchestration | CosmOS | Multi-model pipeline coordination and agent management |
| Shared AI Memory | Steve OS | Cross-application contextual AI memory system |
| Command Palette | Bytebot OS | Natural language OS control (Ctrl+K) |
| Automation Studio | WarmWind OS | Record and replay workflow automations |
| MCP Hub | Archon OS | Model Context Protocol server for AI assistant connectivity |
| Knowledge Base | Archon + Kuse AI | RAG-based document indexing and AI-powered search |
| Ecosystem Manager | CosmOS + Fuchsia | Multi-device management and task distribution |

```bash
# Build the desktop shell (requires Qt5)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run
./src/dashboard/neuraos-dashboard
```

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NEURAOS v5.0.0                                  │
├────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │   LLM    │ │  Vision  │ │  Speech  │ │ Quantum  │ │ Desktop  │    │
│  │llama.cpp │ │NCNN+OCV  │ │whisper   │ │  QuEST   │ │ Qt5 QML  │    │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘    │
│       └─────────────┴────────────┴────────────┴────────────┘          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │      NPIE v2.0.0 - NeuraParse Inference Engine (12 backends)    │  │
│  │  LiteRT│ONNX│emlearn│Wasm│NCNN│ExecuTorch│OpenVINO│SD│MLC│QuEST│  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ MAVLink 2.0 │ Swarm Engine │ PX4 Offboard │ Sensor Fusion(EKF) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ WiFi Mesh │ eBPF QoS │ Cellular │ WireGuard │ Remote ID │ OTA  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Secure Boot │ dm-verity │ AES-256-GCM │ NPU/GPU Drivers        │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │   NPI Init │ cgroups v2 │ OverlayFS │ Watchdog │ BusyBox       │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              Linux 6.12 LTS Kernel (PREEMPT_RT)                 │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Packages

<table>
<tr>
<th align="center">🤖 AI/ML & LLM</th>
<th align="center">⚛️ Quantum</th>
<th align="center">🚁 Robotics</th>
<th align="center">🌐 Network</th>
<th align="center">🛡️ System</th>
</tr>
<tr>
<td>

| Package | Version |
|---------|---------|
| NPIE | 2.0.0 |
| llama.cpp | b7951 |
| whisper.cpp | 1.8.3 |
| stable-diffusion.cpp | 0.4.2 |
| NCNN | 1.0.20260114 |
| ONNX Runtime | 1.24.1 |
| LiteRT | 2.1.2 |
| emlearn | 0.23.1 |
| OpenVINO | 2025.4.1 |
| ExecuTorch | 1.1.0 |
| WasmEdge | 0.16.0 |
| MediaPipe | 0.10.32 |
| Apache TVM | 0.22.0 |

</td>
<td>

| Package | Version |
|---------|---------|
| QuEST | 4.2.0 |
| Qulacs | 0.6.12 |
| Stim (QEC) | 1.15.0 |
| PennyLane | 0.44.0 |
| CUDA Quantum* | 0.10.0 |

</td>
<td>

| Package | Version |
|---------|---------|
| PX4 Autopilot | 1.16.0 |
| ArduPilot | Copter-4.6.3 |
| MAVSDK | 3.14.0 |
| Fast-DDS | 3.4.2 |
| MAVLink | 2.0 |
| Micro-XRCE-DDS | 2.4.3 |
| ZeroMQ | 4.3.6 |
| gRPC | 1.78.0 |

</td>
<td>

| Package | Version |
|---------|---------|
| BATMAN-adv | 2025.2 |
| libbpf | 1.6.2 |
| WireGuard | 1.0.20250521 |
| libsodium | 1.0.21 |
| RAUC | 1.15.1 |

</td>
<td>

| Package | Version |
|---------|---------|
| Kernel | 6.12.57 |
| BusyBox | 1.36.1 |
| Dropbear | 2024.86 |
| iptables | 1.8.10 |
| Qt5 | 5.15.x |
| Python | 3.11.10 |
| NumPy | 1.25.0 |

</td>
</tr>
</table>

<sub>* Disabled by default due to complex dependencies</sub>

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
Docker >= 20.0
QEMU >= 6.0 (for testing)

# System
RAM: 4GB minimum
Disk: 30GB free space
```

### 📥 Build

```bash
# Clone repository
git clone https://github.com/neuraparse/neuraos.git
cd neuraos

# Build with Docker
docker build -f Dockerfile.x86_64 -t neuraos-builder .
docker run --name neuraos-build neuraos-builder
docker cp neuraos-build:/neuraos/buildroot-2025.08/output/images ./neuraos-images

# Or build native components with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 🚁 Drone Simulation

```bash
# Drone simulation (PX4 SITL)
./scripts/neuraos_sim.sh start

# 5-vehicle swarm simulation
./scripts/neuraos_sim.sh swarm 5
```

### 🏗️ Build Profiles

```bash
make profile-minimal    # Core OS only
make profile-drone      # Drone flight stack
make profile-robot      # Robotics platform
make profile-full       # Everything
```

### ▶️ Run

```bash
# Quick start (VM + Web Dashboard)
./scripts/start_neuraos.sh x86_64    # or: arm64

# Or launch manually with KVM
qemu-system-x86_64 -enable-kvm -cpu host -m 1024 -smp 2 \
  -kernel neuraos-images-x86_64/bzImage \
  -drive file=neuraos-images-x86_64/rootfs.ext2,format=raw,if=virtio \
  -append "root=/dev/vda rw console=ttyS0" \
  -netdev user,id=net0,hostfwd=tcp::2222-:22,hostfwd=tcp::8080-:8080 \
  -device virtio-net-pci,netdev=net0 -nographic

# SSH access
ssh -p 2222 root@localhost
# Password: neuraos

# Stop everything
./scripts/stop_neuraos.sh
```

### 🌐 Web Dashboard

```
VM API:    http://localhost:8080
Dashboard: http://localhost:8082
```

---

## 📊 Benchmarks

<table>
<tr>
<td align="center">
<h3>4.55</h3>
<sub>GFLOPS</sub>
<br/>
<b>Matrix Ops</b>
</td>
<td align="center">
<h3>5,325</h3>
<sub>inf/sec</sub>
<br/>
<b>Neural Network</b>
</td>
<td align="center">
<h3>28.37</h3>
<sub>GB/s</sub>
<br/>
<b>Memory BW</b>
</td>
<td align="center">
<h3><5</h3>
<sub>seconds</sub>
<br/>
<b>Boot Time</b>
</td>
</tr>
</table>

<sub>Tested on AMD EPYC 9355P 32-Core • 1GB RAM • KVM</sub>

---

## 🖥️ Supported Platforms

| Platform | Status | Architecture | Notes |
|:--------:|:------:|:------------:|:------|
| <img src="https://img.shields.io/badge/x86__64-KVM-blue?style=flat-square&logo=intel" /> | ✅ Ready | x86_64 | Production |
| <img src="https://img.shields.io/badge/ARM64-QEMU-orange?style=flat-square&logo=arm" /> | ✅ Ready | aarch64 | Development |
| <img src="https://img.shields.io/badge/RPi_4/5-Ready-green?style=flat-square&logo=raspberrypi" /> | ✅ Ready | ARM64 | Cortex-A72/A76 |
| <img src="https://img.shields.io/badge/Pixhawk_6X-Ready-green?style=flat-square&logo=ardupilot" /> | ✅ Ready | ARM64 | STM32H7 FMU |
| <img src="https://img.shields.io/badge/Jetson_Orin_Nano-Ready-green?style=flat-square&logo=nvidia" /> | ✅ Ready | ARM64 | GPU Accel |

---

## 📁 Project Structure

```
neuraos/
├── 📂 src/
│   ├── 📂 npie/               # NPIE v2.0.0 inference engine
│   │   ├── 📂 api/            # Public C API (LLM, Speech, Quantum)
│   │   └── 📂 core/backends/  # 12 backend implementations (.cpp/.c)
│   ├── 📂 drivers/npu/        # NPU driver (hw + simulated)
│   ├── 📂 drivers/accelerators/# GPU/Vulkan acceleration
│   ├── 📂 mavlink/            # MAVLink 2.0 protocol
│   ├── 📂 offboard/           # PX4 offboard control
│   ├── 📂 swarm/              # Swarm coordination
│   ├── 📂 sensors/            # Sensor drivers + EKF fusion
│   ├── 📂 camera/             # V4L2 camera
│   ├── 📂 network/            # WiFi mesh, eBPF, cellular, VPN, Remote ID
│   ├── 📂 security/           # Secure boot, model encryption
│   ├── 📂 power/              # Power/thermal management
│   ├── 📂 ota/                # RAUC OTA manager
│   ├── 📂 npi/                # NPI init system
│   └── 📂 dashboard/          # Qt5 QML desktop shell (33 apps)
│       ├── 📂 backend/        # 14 C++ backend managers
│       ├── 📂 qml/apps/       # Application QML files
│       ├── 📂 qml/shell/      # Shell components (dock, start menu, etc.)
│       └── 📂 qml/components/ # Reusable UI components
├── 📂 tools/
│   └── 📂 npie_bench/         # Multi-backend benchmark tool
├── 📂 web/
│   ├── 📂 frontend/           # React + Vite + TypeScript
│   └── 📂 backend/            # Node.js + Express API
├── 📂 configs/
│   ├── 📄 neuraos_defconfig   # ARM64 (aarch64)
│   ├── 📄 neuraos_x86_64_defconfig
│   ├── 📂 kernel/
│   └── 📂 boards/             # BSP configs (RPi, Jetson, Pixhawk)
├── 📂 package/neuraparse/     # 46 custom Buildroot packages
├── 📂 scripts/
│   ├── 🔧 start_neuraos.sh   # Quick start (VM + Dashboard)
│   ├── 🔧 stop_neuraos.sh    # Stop all services
│   ├── 🔧 run_qemu_kvm.sh    # x86_64 with KVM
│   └── 🔧 run_qemu_headless.sh # ARM64 headless
├── 📂 docs/                   # Architecture, API, guides
├── 📂 tests/                  # 25 tests (10 unit + 8 integration + 7 driver)
├── 🐳 Dockerfile.x86_64
└── 📄 CMakeLists.txt
```

---

## 🔐 Security

| Feature | Status |
|---------|:------:|
| Minimal Attack Surface (<100MB) | ✅ |
| iptables Firewall | ✅ |
| SSH Key Authentication | ✅ |
| No Telemetry | ✅ |
| Offline Operation | ✅ |
| Read-Only Rootfs (optional) | ✅ |
| Model Encryption (AES-256-GCM) | ✅ |
| eBPF Security Policies | ✅ |
| Secure Boot (FIT image) | ✅ |
| dm-verity Root FS | ✅ |
| WireGuard VPN | ✅ |
| RAUC OTA Rollback | ✅ |

---

## 🛠️ Development Tools

| Tool | Description |
|------|-------------|
| `npie-bench` | Multi-backend inference benchmark (JSON + human-readable) |
| `npconvert` | Model format converter |
| `npprofiler` | Performance profiler |
| `npsim` | System simulator |
| `npie-cli` | NPIE command-line interface |

```bash
# Build native components & run tests
make npie && make tools && make test

# Benchmark all backends
npie-bench --iterations 500 --json
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [Architecture (2026)](docs/architecture_2025.md) | System architecture & technology stack |
| [Getting Started](docs/getting_started.md) | First steps guide |
| [API Reference](docs/api_reference.md) | NPIE v2.0.0 API (Core, LLM, Speech, Quantum) |
| [GUI Build Guide](docs/GUI_BUILD.md) | Desktop shell build instructions |
| [Security](docs/security.md) | Security features & best practices |
| [Hardware Support](docs/hardware_support.md) | Supported devices |
| [Package Guide](docs/packages.md) | Adding custom packages |

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Fork & Clone
git clone https://github.com/YOUR_USERNAME/neuraos.git

# Create branch
git checkout -b feature/amazing-feature

# Commit & Push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# Open Pull Request
```

---

## 📜 License

<table>
<tr>
<td>
<img src="https://www.gnu.org/graphics/gplv2-88x31.png" alt="GPL v2"/>
</td>
<td>
This project is licensed under the <b>GNU General Public License v2</b>.<br/>
See <a href="LICENSE">LICENSE</a> for details.
</td>
</tr>
</table>

---

<p align="center">
  <a href="https://neuraparse.com">
    <img src="https://img.shields.io/badge/Website-neuraparse.com-blue?style=for-the-badge&logo=google-chrome" alt="Website"/>
  </a>
  <a href="https://github.com/neuraparse/neuraos">
    <img src="https://img.shields.io/badge/GitHub-neuraparse/neuraos-black?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  <a href="https://github.com/neuraparse/neuraos/issues">
    <img src="https://img.shields.io/badge/Issues-Report%20Bug-red?style=for-the-badge&logo=github" alt="Issues"/>
  </a>
</p>

<p align="center">
  <sub>Built with ❤️ by <b>NeuraParse Team</b> • 2026</sub>
</p>
