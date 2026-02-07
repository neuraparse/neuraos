<h1 align="center">NeuralOS</h1>

<h3 align="center">AI-Native Embedded Linux for Edge Computing, Robotics & Quantum</h3>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL%20v2-blue.svg?style=for-the-badge&logo=gnu" alt="License"/></a>
  <a href="https://github.com/neuraparse/neuraos/releases"><img src="https://img.shields.io/badge/Version-4.0.0-green.svg?style=for-the-badge&logo=v" alt="Version"/></a>
  <a href="https://github.com/neuraparse/neuraos/actions"><img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg?style=for-the-badge&logo=githubactions" alt="Build"/></a>
  <a href="https://github.com/neuraparse/neuraos/stargazers"><img src="https://img.shields.io/github/stars/neuraparse/neuraos?style=for-the-badge&logo=github" alt="Stars"/></a>
</p>

<p align="center">
  <a href="#-features">Features</a> •
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
│  ML:      emlearn       │
│  Python:  3.11 + NumPy  │
└─────────────────────────┘
```
Built-in AI inference engines for edge deployment.

</td>
<td width="50%">

### 🔒 Security-Hardened
```
┌─────────────────────────┐
│  Firewall:  iptables    │
│  SSH:       Dropbear    │
│  Offline:   100%        │
└─────────────────────────┘
```
Zero telemetry, minimal attack surface.

</td>
</tr>
</table>

---

## 🖥️ Desktop Shell

NeuralOS includes a Qt5 QML desktop environment with glassmorphism design, floating dock taskbar, and 26 built-in applications.

**Shell Components:**
- Floating dock taskbar with app indicators
- Start menu with search, pinned apps, and app grid
- Notification center with quick toggles and sliders
- Desktop widgets (clock, system stats, weather, media, calendar)
- Window manager with drag, resize, minimize, maximize, close

**Applications:**

| Category | Apps |
|----------|------|
| System | System Monitor, Terminal, File Manager, Settings, Task Manager, Package Manager, Network Center |
| AI & ML | Neural Studio, AI Agent Hub, AI Assistant, NPU Control Center |
| Utilities | Calculator, Text Editor, Notes, Calendar, Clock, Weather, Photos |
| Media | Music Player, Video Player, Image Viewer |
| Internet | Web Browser, App Store |
| Defense | Drone Command Center, Defense Monitor |
| Quantum | Quantum Lab |

```bash
# Build the desktop shell (requires Qt5)
cd src/dashboard
cmake -B build && cmake --build build

# Run
./build/neuraos-dashboard
```

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          NEURAOS v4.0                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │     LLM     │  │  Robotics   │  │   Quantum   │  │   Vision   │ │
│  │  llama.cpp  │  │  Fast-DDS   │  │    QuEST    │  │   OpenCV   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │
│         │                │                │               │        │
│         └────────────────┴────────────────┴───────────────┘        │
│                          │                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         NPIE - NeuraParse Inference Engine                  │   │
│  │    LiteRT │ ONNX Runtime │ emlearn │ WasmEdge              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │       NPU / GPU Drivers  │  Python 3.11 + NumPy            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           NPI Init  │  BusyBox + Core Utils                 │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                         │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │             Linux 6.12 LTS Kernel (PREEMPT_RT)              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Packages

<table>
<tr>
<th align="center">🤖 AI/ML</th>
<th align="center">🚁 Robotics</th>
<th align="center">⚛️ Quantum</th>
<th align="center">🛡️ System</th>
</tr>
<tr>
<td>

| Package | Version |
|---------|---------|
| NPIE | 1.0.0 |
| llama.cpp | b7746 |
| emlearn | 0.21.1 |
| NumPy | 1.25.0 |
| Python | 3.11.10 |

</td>
<td>

| Package | Version |
|---------|---------|
| Fast-DDS | 3.4.1 |
| MAVLink | 2.0.0 |
| FastCDR | 2.2.0 |
| Asio | 1.30.2 |

</td>
<td>

| Package | Version |
|---------|---------|
| QuEST | 4.2.0 |
| Qiskit* | 0.14.2 |
| PennyLane* | 0.38.0 |

</td>
<td>

| Package | Version |
|---------|---------|
| Kernel | 6.12.57 |
| BusyBox | 1.36.1 |
| Dropbear | 2024.86 |
| iptables | 1.8.10 |

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
| <img src="https://img.shields.io/badge/RPi_4/5-Planned-yellow?style=flat-square&logo=raspberrypi" /> | 🔄 Q2 2026 | ARM64 | Cortex-A72/A76 |
| <img src="https://img.shields.io/badge/Jetson-Planned-yellow?style=flat-square&logo=nvidia" /> | 🔄 Q3 2026 | ARM64 | GPU Accel |

---

## 📁 Project Structure

```
neuraos/
├── 📂 src/
│   ├── 📂 npie/               # NPIE inference engine (4 backends)
│   ├── 📂 drivers/npu/        # NPU driver (hw + simulated)
│   ├── 📂 drivers/accelerators/# GPU acceleration
│   ├── 📂 npi/                # NPI init system
│   └── 📂 dashboard/          # Qt5 QML desktop shell (26 apps)
├── 📂 tools/
│   └── 📂 npie_bench/         # Multi-backend benchmark tool
├── 📂 web/
│   ├── 📂 frontend/           # React + Vite + TypeScript
│   └── 📂 backend/            # Node.js + Express API
├── 📂 configs/
│   ├── 📄 neuraos_defconfig   # ARM64 (aarch64)
│   ├── 📄 neuraos_x86_64_defconfig
│   └── 📂 kernel/
├── 📂 package/neuraparse/     # 44 custom Buildroot packages
├── 📂 scripts/
│   ├── 🔧 start_neuraos.sh   # Quick start (VM + Dashboard)
│   ├── 🔧 stop_neuraos.sh    # Stop all services
│   ├── 🔧 run_qemu_kvm.sh    # x86_64 with KVM
│   └── 🔧 run_qemu_headless.sh # ARM64 headless
├── 📂 tests/                  # Unit, integration & benchmarks
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
| [Getting Started](docs/getting_started.md) | First steps guide |
| [Hardware Support](docs/hardware_support.md) | Supported devices |
| [Package Guide](docs/packages.md) | Adding custom packages |
| [API Reference](docs/api_reference.md) | Web API documentation |

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

